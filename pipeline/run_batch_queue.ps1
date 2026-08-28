[CmdletBinding()]
param(
    [string[]]$VideoPaths,
    [ValidateSet("Auto", "Object", "Walkthrough", "House", "AerialExterior")]
    [string]$CaptureType = "Auto",
    [ValidateSet("Preview", "Final", "Custom")]
    [string]$Mode = "Final",
    [ValidateSet("Brush", "Spirula", "3DGUT-MCMC")]
    [string]$Trainer = "Brush",
    [int]$SelectedFrames = 1200,
    [int]$TrainingSteps = 40000,
    [int]$CandidateMultiplier = 8,
    [double]$MaxCumulativeFlow = 0.0125,
    [int]$MaxLongSide = 0,
    [int]$TrainingMaxResolution = 1920,
    [string]$OutputRoot,
    [string]$ConfigPath,
    [string]$QueuePath,
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$NoAutoRotate,
    [switch]$NoBlender
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not (Get-Command Import-PowerShellDataFile -ErrorAction SilentlyContinue)) {
    Import-Module -Name (Join-Path $PSHOME "Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1") -Force
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CoreRunner = Join-Path $PSScriptRoot "run_video_to_splat.ps1"
$AutoRunner = Join-Path $PSScriptRoot "run_auto_video_to_splat.ps1"
$SafetyHelper = Join-Path $PSScriptRoot "production_safety.ps1"
. $SafetyHelper

if (-not $ConfigPath) { $ConfigPath = Join-Path $ProjectRoot "splatitup.local.psd1" }
$Config = if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    Import-PowerShellDataFile -LiteralPath $ConfigPath
} else { @{} }
if (-not $OutputRoot) {
    $OutputRoot = if ($Config.ContainsKey("OutputRoot") -and $Config.OutputRoot) {
        [string]$Config.OutputRoot
    } else { Join-Path $ProjectRoot "runs" }
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Get-QueueRunName([string]$Path, [string]$Sha256) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Path).ToLowerInvariant()
    $name = ($name -replace "[^a-z0-9]+", "-").Trim([char[]]"-")
    if (-not $name) { $name = "gaussian-run" }
    return "$name-$($Sha256.Substring(0, 8))"
}

function Save-Queue {
    $script:Queue.updated_utc = [DateTime]::UtcNow.ToString("o")
    Write-SplatJsonAtomic -Path $QueuePath -Value $script:Queue -Depth 16
}

$queueRoot = Join-Path $OutputRoot ".queues"
New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
if (-not $QueuePath) {
    $QueuePath = Join-Path $queueRoot "batch-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))-$([guid]::NewGuid().ToString('N').Substring(0,8)).json"
}
$QueuePath = [System.IO.Path]::GetFullPath($QueuePath)
$queuePrefix = $queueRoot.TrimEnd([char[]]"\/") + [System.IO.Path]::DirectorySeparatorChar
if (-not $QueuePath.StartsWith($queuePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "QueuePath must stay inside $queueRoot"
}

if ($Resume) {
    if (-not (Test-Path -LiteralPath $QueuePath -PathType Leaf)) { throw "Queue not found: $QueuePath" }
    $script:Queue = Get-Content -LiteralPath $QueuePath -Raw | ConvertFrom-Json
    if ([int]$script:Queue.schema_version -ne 1) { throw "Unsupported queue schema" }
    if ([string]$script:Queue.output_root -ne $OutputRoot) { throw "Queue output root does not match the current configuration" }
    foreach ($job in $script:Queue.jobs) {
        if ([string]$job.status -eq "RUNNING") { $job.status = "PENDING" }
        if ($RetryFailed -and [string]$job.status -in @("FAILED", "REVIEW_REQUIRED")) { $job.status = "PENDING" }
    }
} else {
    if (Test-Path -LiteralPath $QueuePath) { throw "Queue already exists: $QueuePath" }
    $supported = @(".mov", ".mp4", ".m4v", ".avi", ".mkv")
    $resolvedVideos = @($VideoPaths | ForEach-Object {
        $path = [System.IO.Path]::GetFullPath($_)
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Video not found: $path" }
        if ([System.IO.Path]::GetExtension($path).ToLowerInvariant() -notin $supported) { throw "Unsupported video: $path" }
        $path
    })
    if ($resolvedVideos.Count -lt 1) { throw "At least one video is required" }
    if (@($resolvedVideos | Select-Object -Unique).Count -ne $resolvedVideos.Count) { throw "The queue contains a duplicate video path" }

    $jobs = @()
    $sourceHashes = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    for ($index = 0; $index -lt $resolvedVideos.Count; $index++) {
        $video = $resolvedVideos[$index]
        Write-Host "[queue] Hashing $($index + 1)/$($resolvedVideos.Count): $video"
        $sourceSha = (Get-FileHash -LiteralPath $video -Algorithm SHA256).Hash.ToLowerInvariant()
        if (-not $sourceHashes.Add($sourceSha)) { throw "The queue contains duplicate video content: $video" }
        $jobs += [ordered]@{
            index = $index + 1
            video_path = $video
            source_sha256 = $sourceSha
            run_name = Get-QueueRunName $video $sourceSha
            run_root = Join-Path $OutputRoot (Get-QueueRunName $video $sourceSha)
            status = "PENDING"
            attempts = 0
            started_utc = $null
            completed_utc = $null
            exit_code = $null
            last_error = $null
        }
    }
    $script:Queue = [ordered]@{
        schema_version = 1
        queue_id = [guid]::NewGuid().ToString()
        status = "PENDING"
        created_utc = [DateTime]::UtcNow.ToString("o")
        updated_utc = [DateTime]::UtcNow.ToString("o")
        output_root = $OutputRoot
        settings = [ordered]@{
            capture_type = $CaptureType
            mode = $Mode
            trainer = $Trainer
            selected_frames = $SelectedFrames
            training_steps = $TrainingSteps
            candidate_multiplier = $CandidateMultiplier
            max_cumulative_flow = $MaxCumulativeFlow
            max_long_side = $MaxLongSide
            training_max_resolution = $TrainingMaxResolution
            no_auto_rotate = [bool]$NoAutoRotate
            no_blender = [bool]$NoBlender
            config_path = [System.IO.Path]::GetFullPath($ConfigPath)
        }
        jobs = $jobs
    }
    Save-Queue
}

$pendingCount = @($script:Queue.jobs | Where-Object { [string]$_.status -eq "PENDING" }).Count
$minimumFreeGB = Get-SplatProductionValue -Config $Config -Name "MinimumFreeSpaceGB" -Default 20
$perJobReserveGB = Get-SplatProductionValue -Config $Config -Name "PerQueuedJobReserveGB" -Default 12
$requiredFreeGB = [math]::Max($minimumFreeGB, $pendingCount * $perJobReserveGB)
Assert-SplatDiskCapacity -Path $OutputRoot -RequiredGB $requiredFreeGB | Out-Null

$queueLock = $null
$batchLock = $null
try {
    $batchLock = Enter-SplatExclusiveLock -Path (Join-Path $OutputRoot ".splatitup.batch.lock") -Metadata @{
        kind = "batch"
        queue_path = $QueuePath
    }
    $queueLock = Enter-SplatExclusiveLock -Path "$QueuePath.lock" -Metadata @{
        kind = "queue"
        queue_path = $QueuePath
    }
    $script:Queue.status = "RUNNING"
    Save-Queue
    Write-Host "QUEUE: $QueuePath"

    foreach ($job in $script:Queue.jobs) {
        if ([string]$job.status -ne "PENDING") { continue }
        $job.status = "RUNNING"
        $job.attempts = [int]$job.attempts + 1
        $job.started_utc = [DateTime]::UtcNow.ToString("o")
        $job.completed_utc = $null
        $job.exit_code = $null
        $job.last_error = $null
        Save-Queue

        $settings = $script:Queue.settings
        $runner = if ([string]$settings.capture_type -eq "Auto") { $AutoRunner } else { $CoreRunner }
        $arguments = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner,
            "-VideoPath", [string]$job.video_path,
            "-RunName", [string]$job.run_name,
            "-SelectedFrames", [string]$settings.selected_frames,
            "-TrainingSteps", [string]$settings.training_steps,
            "-Trainer", [string]$settings.trainer,
            "-CandidateMultiplier", [string]$settings.candidate_multiplier,
            "-MaxCumulativeFlow", [string]$settings.max_cumulative_flow,
            "-MaxLongSide", [string]$settings.max_long_side,
            "-TrainingMaxResolution", [string]$settings.training_max_resolution,
            "-OutputRoot", $OutputRoot,
            "-ConfigPath", [string]$settings.config_path,
            "-ToStage", "view",
            "-NoBrowser"
        )
        if ([string]$settings.capture_type -eq "Auto") {
            $arguments += @("-AutoPreset", [string]$settings.mode)
        } else {
            $arguments += @("-SceneType", [string]$settings.capture_type)
        }
        if ([bool]$settings.no_auto_rotate) { $arguments += "-NoAutoRotate" }
        if ([bool]$settings.no_blender) { $arguments += "-NoBlender" }

        Write-Host "[queue] Starting $($job.index)/$($script:Queue.jobs.Count): $($job.run_name)"
        & powershell.exe @arguments
        $job.exit_code = $LASTEXITCODE
        $job.completed_utc = [DateTime]::UtcNow.ToString("o")
        if ($LASTEXITCODE -eq 0) {
            $job.status = "COMPLETE"
        } elseif ($LASTEXITCODE -eq 2) {
            $job.status = "REVIEW_REQUIRED"
            $job.last_error = "The pipeline stopped at a fail-closed review gate."
        } else {
            $job.status = "FAILED"
            $job.last_error = "The pipeline exited with code $LASTEXITCODE. Review the run logs before retrying."
        }
        Save-Queue
    }

    $failed = @($script:Queue.jobs | Where-Object { [string]$_.status -eq "FAILED" }).Count
    $review = @($script:Queue.jobs | Where-Object { [string]$_.status -eq "REVIEW_REQUIRED" }).Count
    $script:Queue.status = if ($failed -gt 0) { "COMPLETE_WITH_FAILURES" } elseif ($review -gt 0) { "COMPLETE_WITH_REVIEW" } else { "COMPLETE" }
    Save-Queue
    Write-Host "[queue] $($script:Queue.status): $QueuePath"
    if ($failed -gt 0) { exit 1 }
    if ($review -gt 0) { exit 2 }
    exit 0
} finally {
    Exit-SplatExclusiveLock $queueLock
    Exit-SplatExclusiveLock $batchLock
}
