[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$VideoPath,

    [string]$RunName,
    [Parameter(Mandatory = $true)]
    [ValidateSet("Object", "Walkthrough", "House", "AerialExterior")]
    [string]$SceneType,
    [int]$SelectedFrames = 300,
    [int]$TrainingSteps = 30000,
    [ValidateSet("Brush", "Spirula", "3DGUT", "3DGUT-MCMC")]
    [string]$Trainer = "Brush",
    [switch]$AdaptiveExtraction,
    [int]$CandidateMultiplier = 2,
    [double]$MaxCumulativeFlow = 0.0,
    [int]$MaxLongSide = 0,
    [int]$TrainingMaxResolution = 1920,
    [int]$BrushMaxSplats = 10000000,
    [double]$BrushScaleLossWeight = 1e-8,
    [int]$EvalSplitEvery = 10,
    [switch]$NoAutoRotate,
    [ValidateSet("extract", "select", "solve", "train", "blender", "view")]
    [string]$FromStage = "extract",
    [ValidateSet("extract", "select", "solve", "train", "blender", "view")]
    [string]$ToStage = "view",
    [switch]$RevalidateSolve,
    [string]$ShotManifestPath,
    [string]$OutputRoot,
    [string]$ConfigPath,
    [int]$ViewerPort = 3010,
    [switch]$NoBlender,
    [switch]$OpenBlender,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if (-not (Get-Command Import-PowerShellDataFile -ErrorAction SilentlyContinue)) {
    Import-Module -Name (Join-Path $PSHOME "Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1") -Force
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Stages = @("extract", "select", "solve", "train", "blender", "view")
$SelectorVersion = "2.4-max-cumulative-flow"
$GateVersion = "2.9-shot-edge-redundancy"
$TrainingSeed = 42
$AerialDiagnosticSteps = 18000
$AerialDiagnosticGrowthStopIter = 15000
$AerialDiagnosticMaxSplats = 6000000
$AerialDiagnosticMaxResolution = 1920
$AerialDiagnosticEvalSplitEvery = 10
$ThreeDGrutSmokeSteps = 1000
$ThreeDGrutSmokeMaxSplats = 250000
$ThreeDGrutDiagnosticSteps = 7000
$ThreeDGrutDiagnosticMaxSplats = 1000000
$ThreeDGrutEnvironmentFingerprintVersion = "importlib-metadata-runtime-v3-utf8"
$ThreeDGrutRuntimeHelperPath = Join-Path $PSScriptRoot "three_dgrut_runtime.ps1"
. $ThreeDGrutRuntimeHelperPath
$ProductionSafetyPath = Join-Path $PSScriptRoot "production_safety.ps1"
. $ProductionSafetyPath
$FromIndex = [array]::IndexOf($Stages, $FromStage)
$ToIndex = [array]::IndexOf($Stages, $ToStage)
if ($FromIndex -gt $ToIndex) { throw "FromStage must come before ToStage" }
$UsesChronologicalMatching = $SceneType -in @("Walkthrough", "House", "AerialExterior")
$UsesIncrementalMapper = $SceneType -in @("Walkthrough", "House")
$RequiresAerialDiagnostic = $Trainer -eq "Brush" -and $SceneType -eq "AerialExterior" -and $TrainingSteps -ge 40000
$RequiresThreeDGrutAerialStages = $Trainer -eq "3DGUT-MCMC" -and $SceneType -eq "AerialExterior"
if ($Trainer -eq "3DGUT") {
    throw "EXPERIMENTAL_TRAINER_UNSUPPORTED: Plain 3DGUT has no measured staged quality gate and cannot run or publish. Use Brush."
}
if ($Trainer -eq "3DGUT-MCMC" -and -not $RequiresThreeDGrutAerialStages) {
    throw "EXPERIMENTAL_TRAINER_UNSUPPORTED: 3DGUT-MCMC is limited to the AerialExterior 1K visual-review smoke. Use Brush for $SceneType."
}

function Test-Stage([string]$Name) {
    $index = [array]::IndexOf($Stages, $Name)
    return $index -ge $FromIndex -and $index -le $ToIndex
}

function Assert-ApprovedTrainerForDownstream([string]$Stage) {
    if ($Trainer -notin @("Brush", "Spirula")) {
        throw "THREEDGRUT_VISUAL_APPROVAL_REQUIRED: $Trainer has no approved training evidence and cannot enter $Stage. Run the AerialExterior 1K smoke for visual review or use Brush."
    }
}

if ($Trainer -notin @("Brush", "Spirula") -and -not (Test-Stage "train")) {
    if (Test-Stage "blender") { Assert-ApprovedTrainerForDownstream "Blender handoff" }
    if (Test-Stage "view") { Assert-ApprovedTrainerForDownstream "viewer handoff" }
}

if ($RevalidateSolve -and $FromStage -ne "solve") {
    throw "RevalidateSolve is validation-only and requires -FromStage solve."
}

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $ProjectRoot "splatitup.local.psd1"
}
$Config = @{}
if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    $Config = Import-PowerShellDataFile -LiteralPath $ConfigPath
}

function Resolve-ExecutablePath {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$CommandName,
        [string[]]$Candidates = @(),
        [switch]$Required
    )

    $paths = [System.Collections.Generic.List[string]]::new()
    if ($Config.ContainsKey("Tools") -and $Config.Tools.ContainsKey($Name) -and $Config.Tools[$Name]) {
        $paths.Add([string]$Config.Tools[$Name])
    }
    $environmentValue = [Environment]::GetEnvironmentVariable("SPLATITUP_" + $Name.ToUpperInvariant())
    if ($environmentValue) { $paths.Add($environmentValue) }
    foreach ($candidate in $Candidates) {
        if ($candidate) { $paths.Add($candidate) }
    }

    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            return (Get-Item -LiteralPath $path).FullName
        }
    }
    if ($CommandName) {
        $command = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) { return $command.Source }
    }
    if ($Required) {
        throw "Required tool '$Name' was not found. Copy splatitup.config.example.psd1 to splatitup.local.psd1 and set its path."
    }
    return $null
}

function Resolve-DirectoryPath {
    param([string]$Name, [string[]]$Candidates = @())
    $paths = [System.Collections.Generic.List[string]]::new()
    if ($Config.ContainsKey("Tools") -and $Config.Tools.ContainsKey($Name) -and $Config.Tools[$Name]) {
        $paths.Add([string]$Config.Tools[$Name])
    }
    $environmentValue = [Environment]::GetEnvironmentVariable("SPLATITUP_" + $Name.ToUpperInvariant())
    if ($environmentValue) { $paths.Add($environmentValue) }
    foreach ($candidate in $Candidates) {
        if ($candidate) { $paths.Add($candidate) }
    }
    foreach ($path in $paths) {
        if (Test-Path -LiteralPath $path -PathType Container) {
            return (Get-Item -LiteralPath $path).FullName
        }
    }
    return $null
}

function Get-ThreeDGrutEnvironmentFingerprint {
    param(
        [Parameter(Mandatory = $true)][string]$PythonPath,
        [Parameter(Mandatory = $true)][string[]]$RequiredModules,
        [Parameter(Mandatory = $true)][string[]]$RuntimeFingerprintRows
    )
    $moduleLiteral = ($RequiredModules | ForEach-Object { "'$_'" }) -join ","
    $fingerprintCode = "import importlib.metadata as m, importlib.util as u, sys; required=($moduleLiteral,); assert all(u.find_spec(name) is not None for name in required); rows=['python=='+sys.version.replace(chr(10),' ')]; rows.extend(sorted('{}=={}'.format(d.metadata.get('Name',''),d.version) for d in m.distributions() if d.metadata.get('Name'))); print(chr(10).join(rows))"
    $lines = @(& $PythonPath -c $fingerprintCode 2>$null)
    if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 2) {
        throw "3DGRUT Python environment fingerprint failed. The official uv environment must be installed before footage processing starts."
    }
    $fingerprintLines = @($lines) + @($RuntimeFingerprintRows | Sort-Object)
    $bytes = [Text.Encoding]::UTF8.GetBytes(($fingerprintLines -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

$PythonPath = Resolve-ExecutablePath -Name "Python" -CommandName "python.exe" -Candidates @(
    (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
) -Required
$FfmpegPath = Resolve-ExecutablePath -Name "Ffmpeg" -CommandName "ffmpeg.exe" -Candidates @(
    (Join-Path $ProjectRoot "tools\ffmpeg\bin\ffmpeg.exe")
) -Required
$FfprobePath = Resolve-ExecutablePath -Name "Ffprobe" -CommandName "ffprobe.exe" -Candidates @(
    (Join-Path $ProjectRoot "tools\ffmpeg\bin\ffprobe.exe")
) -Required
$ColmapPath = Resolve-ExecutablePath -Name "Colmap" -CommandName "colmap.exe" -Candidates @(
    (Join-Path $ProjectRoot "tools\colmap\bin\colmap.exe")
) -Required
$BrushPath = Resolve-ExecutablePath -Name "Brush" -Candidates @(
    (Join-Path $ProjectRoot "tools\brush\brush_app.exe")
) -Required:($Trainer -eq "Brush" -and (Test-Stage "train"))
$SpirulaPath = Resolve-ExecutablePath -Name "Spirula" -Candidates @(
    (Join-Path $ProjectRoot "tools\spirula\v2026.8.28\bin\spirula.exe")
) -Required:($Trainer -eq "Spirula" -and (Test-Stage "train"))
$SuperSplatDist = Resolve-DirectoryPath -Name "SuperSplatDist" -Candidates @(
    (Join-Path $ProjectRoot "tools\supersplat\dist")
)
$VocabTreePath = Resolve-ExecutablePath -Name "VocabTree" -Candidates @(
    (Join-Path (Split-Path -Parent $ColmapPath) "..\vocab_tree_faiss_flickr100K_words32K.bin")
) -Required:($UsesChronologicalMatching -and (Test-Stage "solve"))
$BlenderBuilderPath = Resolve-ExecutablePath -Name "BlenderBuilder" -Candidates @(
    "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
) -Required:(-not $NoBlender -and (Test-Stage "blender"))
$BlenderOpenPath = Resolve-ExecutablePath -Name "BlenderOpen" -Candidates @(
    "C:\Users\mat\Applications\blender-5.2.0-windows-x64\blender.exe",
    "C:\Users\mat\AppData\Local\Programs\Blender Foundation\Blender 5.2 LTS Release Candidate\blender.exe"
) -Required:(-not $NoBlender -and (Test-Stage "blender"))

if (-not $OutputRoot) {
    if ($Config.ContainsKey("OutputRoot") -and $Config.OutputRoot) {
        $OutputRoot = [string]$Config.OutputRoot
    } else {
        $OutputRoot = Join-Path $ProjectRoot "runs"
    }
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$MinimumFreeSpaceGB = Get-SplatProductionValue -Config $Config -Name "MinimumFreeSpaceGB" -Default 20

$ThreeDGrutRepo = $null
$ThreeDGrutPython = $null
$ThreeDGrutRuntime = $null
$ThreeDGrutConfigName = if ($Trainer -eq "3DGUT-MCMC") { "apps/colmap_3dgut_mcmc.yaml" } else { "apps/colmap_3dgut.yaml" }
$ThreeDGrutModelConfig = $null
$ThreeDGrutEnvironmentSha256 = $null
$ThreeDGrutMcmcMaxSplats = 1000000
if ($Config.ContainsKey("ThreeDGRUT") -and $Config.ThreeDGRUT.ContainsKey("McmcMaxSplats") -and $Config.ThreeDGRUT.McmcMaxSplats) {
    $ThreeDGrutMcmcMaxSplats = [int]$Config.ThreeDGRUT.McmcMaxSplats
}
if ($Trainer -in @("3DGUT", "3DGUT-MCMC")) {
    if (-not $Config.ContainsKey("ThreeDGRUT")) {
        throw "3DGRUT is not configured. Add ThreeDGRUT.Repo and ThreeDGRUT.Python to splatitup.local.psd1."
    }
    $ThreeDGrutRepo = [string]$Config.ThreeDGRUT.Repo
    $ThreeDGrutPython = [string]$Config.ThreeDGRUT.Python
    if (-not $ThreeDGrutRepo -or -not (Test-Path -LiteralPath $ThreeDGrutRepo -PathType Container)) {
        throw "3DGRUT.Repo is missing or is not a directory: $ThreeDGrutRepo"
    }
    if (-not $ThreeDGrutPython -or -not (Test-Path -LiteralPath $ThreeDGrutPython -PathType Leaf)) {
        throw "3DGRUT.Python is missing or is not a file: $ThreeDGrutPython"
    }
    $ThreeDGrutRepo = (Get-Item -LiteralPath $ThreeDGrutRepo).FullName
    $ThreeDGrutPython = (Get-Item -LiteralPath $ThreeDGrutPython).FullName
    if (-not (Test-Path -LiteralPath (Join-Path $ThreeDGrutRepo "train.py") -PathType Leaf)) {
        throw "3DGRUT.Repo does not contain the official train.py entrypoint: $ThreeDGrutRepo"
    }
    $ThreeDGrutModelConfig = Join-Path $ThreeDGrutRepo (Join-Path "configs" $ThreeDGrutConfigName)
    if (-not (Test-Path -LiteralPath $ThreeDGrutModelConfig -PathType Leaf)) {
        throw "3DGRUT official config is missing: $ThreeDGrutModelConfig"
    }
    $ThreeDGrutRuntime = Initialize-ThreeDGrutRuntime -ThreeDGrutConfig $Config.ThreeDGRUT
    $ThreeDGrutRepo = [string]$ThreeDGrutRuntime.Repo
    $ThreeDGrutPython = [string]$ThreeDGrutRuntime.Python
    $ThreeDGrutEnvironmentSha256 = Get-ThreeDGrutEnvironmentFingerprint `
        -PythonPath $ThreeDGrutPython `
        -RequiredModules @($ThreeDGrutRuntime.RequiredModules) `
        -RuntimeFingerprintRows @($ThreeDGrutRuntime.FingerprintRows)
}

$SpirulaFloaterSuppression = "mild"
$SpirulaDistractionRobustness = if ($SceneType -eq "Object") { "off" } else { "mild" }
$SpirulaSaveFullCheckpoint = $false
if ($Config.ContainsKey("Spirula")) {
    if ($Config.Spirula.ContainsKey("FloaterSuppression") -and $Config.Spirula.FloaterSuppression) {
        $SpirulaFloaterSuppression = [string]$Config.Spirula.FloaterSuppression
    }
    if ($Config.Spirula.ContainsKey("DistractionRobustness") -and $Config.Spirula.DistractionRobustness) {
        $SpirulaDistractionRobustness = [string]$Config.Spirula.DistractionRobustness
    }
    if ($Config.Spirula.ContainsKey("SaveFullCheckpoint")) {
        $SpirulaSaveFullCheckpoint = [bool]$Config.Spirula.SaveFullCheckpoint
    }
}
if ($SpirulaFloaterSuppression -notin @("off", "mild", "strong")) {
    throw "Spirula.FloaterSuppression must be off, mild, or strong."
}
if ($SpirulaDistractionRobustness -notin @("off", "mild", "strong")) {
    throw "Spirula.DistractionRobustness must be off, mild, or strong."
}

$VideoPath = [System.IO.Path]::GetFullPath($VideoPath)
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) { throw "Video not found: $VideoPath" }
if ($SelectedFrames -lt 2) { throw "SelectedFrames must be at least 2" }
if ($TrainingSteps -lt 1) { throw "TrainingSteps must be positive" }
if ($CandidateMultiplier -lt 1) { throw "CandidateMultiplier must be at least 1" }
if ($MaxCumulativeFlow -lt 0.0) { throw "MaxCumulativeFlow cannot be negative" }
if ($MaxLongSide -lt 0) { throw "MaxLongSide cannot be negative" }
if ($TrainingMaxResolution -lt 320) { throw "TrainingMaxResolution must be at least 320" }
if ($BrushMaxSplats -lt 10000) { throw "BrushMaxSplats must be at least 10000" }
if ($ThreeDGrutMcmcMaxSplats -lt 10000) { throw "ThreeDGRUT.McmcMaxSplats must be at least 10000" }
if ([double]::IsNaN($BrushScaleLossWeight) -or [double]::IsInfinity($BrushScaleLossWeight) -or $BrushScaleLossWeight -lt 0.0) {
    throw "BrushScaleLossWeight must be a finite non-negative number"
}
if ($EvalSplitEvery -lt 2) { throw "EvalSplitEvery must be at least 2" }
if ($RequiresAerialDiagnostic -and $TrainingMaxResolution -ne $AerialDiagnosticMaxResolution) {
    throw "AerialExterior Brush training at 40000 or more steps requires TrainingMaxResolution $AerialDiagnosticMaxResolution for the fail-closed diagnostic."
}
if ($RequiresAerialDiagnostic -and $EvalSplitEvery -ne $AerialDiagnosticEvalSplitEvery) {
    throw "AerialExterior Brush training at 40000 or more steps requires EvalSplitEvery $AerialDiagnosticEvalSplitEvery for the fail-closed diagnostic."
}
if ($RequiresThreeDGrutAerialStages -and $TrainingSteps -lt $ThreeDGrutSmokeSteps) {
    throw "AerialExterior 3DGUT-MCMC requires at least $ThreeDGrutSmokeSteps requested steps for the exact 1K/250K visual-review smoke."
}
if ($RequiresThreeDGrutAerialStages -and $ThreeDGrutMcmcMaxSplats -lt $ThreeDGrutSmokeMaxSplats) {
    throw "AerialExterior 3DGUT-MCMC requires ThreeDGRUT.McmcMaxSplats of at least $ThreeDGrutSmokeMaxSplats for the capped smoke."
}

if (-not $RunName) {
    $RunName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath).ToLowerInvariant()
    $RunName = ($RunName -replace "[^a-z0-9]+", "-").Trim([char[]]"-")
}
if ($RunName -notmatch "^[a-zA-Z0-9][a-zA-Z0-9_-]*$") {
    throw "RunName may contain only letters, numbers, hyphens, and underscores"
}

$RunRoot = Join-Path $OutputRoot $RunName
$FramesRoot = Join-Path $RunRoot "frames"
$RawFrames = Join-Path $FramesRoot "raw"
$ReconRoot = Join-Path $RunRoot "recon"
$ImagesPath = Join-Path $ReconRoot "images"
$SparseRoot = Join-Path $ReconRoot "sparse"
$SparsePath = Join-Path $SparseRoot "0"
$ModelTextRoot = Join-Path $ReconRoot "sparse_txt"
$DatabasePath = Join-Path $ReconRoot "database.db"
$CalibratedDatabasePath = Join-Path $ReconRoot "database_global.db"
$ImageOrderPath = Join-Path $ReconRoot "image_order.txt"
$UndistortedRoot = Join-Path $ReconRoot "undistorted"
$BrushOutput = Join-Path $RunRoot "brush"
$SpirulaOutput = Join-Path $RunRoot "spirula"
$ThreeDGrutOutput = Join-Path $RunRoot "3dgrut"
$FinalRoot = Join-Path $RunRoot "final"
$FinalPly = Join-Path $FinalRoot "$RunName.ply"
$BlenderRoot = Join-Path $RunRoot "blender"
$BlenderFile = Join-Path $BlenderRoot "$RunName.blend"
$BlenderReportPath = Join-Path $BlenderRoot "blender_handoff_report.json"
$BlenderPreviewPath = Join-Path $BlenderRoot "blender_handoff_preview.png"
$BlenderOpenReportPath = Join-Path $BlenderRoot "blender_52_open_report.json"
$LogsRoot = Join-Path $RunRoot "logs"
$DiagnosticsRoot = Join-Path $RunRoot "diagnostics"
$RunManifestPath = Join-Path $RunRoot "run_manifest.json"
$RunAttemptPath = Join-Path $RunRoot "run_attempt.json"
$TrainingDecisionPath = Join-Path $RunRoot "training_decision.json"
$FrameQualityPath = Join-Path $RunRoot "frame_quality.json"

Assert-SplatDiskCapacity -Path $OutputRoot -RequiredGB $MinimumFreeSpaceGB | Out-Null
New-Item -ItemType Directory -Force -Path $RunRoot, $FramesRoot, $ReconRoot, $LogsRoot | Out-Null
$GlobalLock = $null
$RunLock = $null
try {
    $GlobalLock = Enter-SplatExclusiveLock -Path (Join-Path $OutputRoot ".splatitup.gpu.lock") -Metadata @{
        kind = "gpu"
        run_name = $RunName
        video_path = $VideoPath
    }
    $RunLock = Enter-SplatExclusiveLock -Path (Join-Path $RunRoot ".splatitup.run.lock") -Metadata @{
        kind = "run"
        run_name = $RunName
        video_path = $VideoPath
    }
} catch {
    Exit-SplatExclusiveLock $RunLock
    Exit-SplatExclusiveLock $GlobalLock
    throw
}

try {
$SourcePath = Join-Path $RunRoot "source.json"
$VideoFile = Get-Item -LiteralPath $VideoPath
$ResolvedVideo = $VideoFile.FullName
$SourceBytes = [int64]$VideoFile.Length
$SourceLastWriteUtc = $VideoFile.LastWriteTimeUtc.ToString("o")
$CreatedUtc = [DateTime]::UtcNow.ToString("o")
if (Test-Path -LiteralPath $SourcePath) {
    try {
        $existingSource = Get-Content -LiteralPath $SourcePath -Raw | ConvertFrom-Json
        if ($existingSource.video_path -ne $ResolvedVideo) {
            throw "Run '$RunName' already belongs to a different video: $($existingSource.video_path)"
        }
        if ($existingSource.PSObject.Properties.Name -contains "created_utc") {
            $CreatedUtc = [string]$existingSource.created_utc
        }
    } catch {
        if ($_.Exception.Message -like "Run '$RunName' already belongs*") { throw }
        $corruptSource = "$SourcePath.corrupt-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))"
        Move-Item -LiteralPath $SourcePath -Destination $corruptSource
        Write-Warning "Archived malformed source metadata to $corruptSource; all downstream provenance will be revalidated."
    }
}
Write-Host "[source] Hashing input video for reproducible resume state"
$SourceSha256 = (Get-FileHash -LiteralPath $ResolvedVideo -Algorithm SHA256).Hash.ToLowerInvariant()
[ordered]@{
    video_path = $ResolvedVideo
    run_name = $RunName
    scene_type = $SceneType
    source_bytes = $SourceBytes
    source_last_write_utc = $SourceLastWriteUtc
    source_sha256 = $SourceSha256
    created_utc = $CreatedUtc
} | ConvertTo-Json | Set-Content -LiteralPath "$SourcePath.tmp" -Encoding utf8
Move-Item -LiteralPath "$SourcePath.tmp" -Destination $SourcePath -Force

function Invoke-LoggedCommand {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$LogPath,
        [string]$WorkingDirectory
    )
    Write-Host "`n> $([System.IO.Path]::GetFileName($FilePath)) $($ArgumentList -join ' ')" -ForegroundColor Cyan
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($WorkingDirectory) { Push-Location -LiteralPath $WorkingDirectory }
        Set-Content -LiteralPath $LogPath -Value ([string]::Empty) -NoNewline
        & $FilePath @ArgumentList 2>&1 | Tee-Object -FilePath $LogPath
        $exitCode = $LASTEXITCODE
    } finally {
        if ($WorkingDirectory) { Pop-Location }
        $ErrorActionPreference = $previousErrorAction
    }
    if ($exitCode -ne 0) {
        throw "$([System.IO.Path]::GetFileName($FilePath)) failed with exit code $exitCode. See $LogPath"
    }
}

function Read-Marker([string]$Name) {
    $path = Join-Path $RunRoot ".$Name.complete.json"
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try {
        return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
    } catch {
        Write-Warning "Ignoring malformed resume marker: $path"
        return $null
    }
}

function Test-ObjectProperty($Object, [string]$Name) {
    return $null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name
}

function Write-Marker([string]$Name, [System.Collections.IDictionary]$Details) {
    $Details["completed_utc"] = [DateTime]::UtcNow.ToString("o")
    $path = Join-Path $RunRoot ".$Name.complete.json"
    $temp = "$path.tmp"
    $Details | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temp -Encoding utf8
    Move-Item -LiteralPath $temp -Destination $path -Force
}

function Convert-FractionToDouble([string]$Value) {
    if (-not $Value -or $Value -eq "0/0") { return 0.0 }
    if ($Value -match "^([0-9.]+)\/([0-9.]+)$") {
        $denominator = [double]::Parse($Matches[2], [Globalization.CultureInfo]::InvariantCulture)
        if ($denominator -eq 0) { return 0.0 }
        return [double]::Parse($Matches[1], [Globalization.CultureInfo]::InvariantCulture) / $denominator
    }
    return [double]::Parse($Value, [Globalization.CultureInfo]::InvariantCulture)
}

function Get-ImageSetHash([string]$Path) {
    $manifest = Get-ChildItem -LiteralPath $Path -Filter "*.jpg" | Sort-Object Name | ForEach-Object {
        $fileHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$($_.Name)|$fileHash"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($manifest -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-FileSetHash([string]$Path, [string]$Filter) {
    $manifest = Get-ChildItem -LiteralPath $Path -Filter $Filter | Sort-Object Name | ForEach-Object {
        $fileHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$($_.Name)|$fileHash"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($manifest -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-MarkerEvidenceFileState($Marker, [string]$PathProperty, [string]$HashProperty) {
    if (-not $Marker -or -not (Test-ObjectProperty $Marker $PathProperty) -or
        -not (Test-ObjectProperty $Marker $HashProperty)) {
        return [pscustomobject]@{ valid = $false; path = $null; sha256 = $null }
    }
    $pathValue = [string]$Marker.$PathProperty
    $expectedHash = [string]$Marker.$HashProperty
    if (-not $pathValue -or -not $expectedHash) {
        return [pscustomobject]@{ valid = $false; path = $pathValue; sha256 = $null }
    }
    $resolvedPath = [System.IO.Path]::GetFullPath($pathValue)
    $runPrefix = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd([char[]]"\/") + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedPath.StartsWith($runPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
        return [pscustomobject]@{ valid = $false; path = $resolvedPath; sha256 = $null }
    }
    $actualHash = (Get-FileHash -LiteralPath $resolvedPath -Algorithm SHA256).Hash.ToLowerInvariant()
    return [pscustomobject]@{
        valid = $actualHash -eq $expectedHash
        path = $resolvedPath
        sha256 = $actualHash
    }
}

function Assert-AerialGateInputEvidence {
    param(
        $Payload,
        [string]$QualityReport,
        [string]$PlyReport,
        [string]$Ply
    )
    if (-not $Payload -or -not (Test-ObjectProperty $Payload "input_evidence")) {
        throw "The aerial gate did not bind its exact input evidence."
    }
    $evidence = $Payload.input_evidence
    foreach ($name in @("quality_report", "ply_report", "ply")) {
        if (-not (Test-ObjectProperty $evidence $name) -or
            -not (Test-ObjectProperty $evidence.$name "path") -or
            -not (Test-ObjectProperty $evidence.$name "sha256")) {
            throw "The aerial gate input evidence is incomplete for $name."
        }
    }
    foreach ($path in @($QualityReport, $PlyReport, $Ply)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "An aerial gate input disappeared before authorization: $path"
        }
    }

    $qualityPath = [System.IO.Path]::GetFullPath($QualityReport)
    $plyReportPath = [System.IO.Path]::GetFullPath($PlyReport)
    $plyPath = [System.IO.Path]::GetFullPath($Ply)
    if (-not [string]::Equals($qualityPath, [System.IO.Path]::GetFullPath([string]$evidence.quality_report.path), [StringComparison]::OrdinalIgnoreCase) -or
        -not [string]::Equals($plyReportPath, [System.IO.Path]::GetFullPath([string]$evidence.ply_report.path), [StringComparison]::OrdinalIgnoreCase) -or
        -not [string]::Equals($plyPath, [System.IO.Path]::GetFullPath([string]$evidence.ply.path), [StringComparison]::OrdinalIgnoreCase)) {
        throw "The aerial gate input paths do not match the files being authorized."
    }

    $qualitySha = (Get-FileHash -LiteralPath $qualityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $plyReportSha = (Get-FileHash -LiteralPath $plyReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $plySha = (Get-FileHash -LiteralPath $plyPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($qualitySha -ne ([string]$evidence.quality_report.sha256).ToLowerInvariant() -or
        $plyReportSha -ne ([string]$evidence.ply_report.sha256).ToLowerInvariant() -or
        $plySha -ne ([string]$evidence.ply.sha256).ToLowerInvariant()) {
        throw "An aerial gate input changed after evaluation."
    }
    if (-not (Test-ObjectProperty $evidence.ply "bytes") -or
        [long]$evidence.ply.bytes -ne (Get-Item -LiteralPath $plyPath).Length) {
        throw "The gated aerial PLY byte count does not match the candidate."
    }
    return [pscustomobject]@{
        quality_report_sha256 = $qualitySha
        ply_report_sha256 = $plyReportSha
        ply_sha256 = $plySha
    }
}

function Get-AerialDiagnosticEvidenceState($Marker) {
    if (-not $RequiresAerialDiagnostic) {
        return [pscustomobject]@{ valid = $true; required = $false }
    }
    if (-not $Marker) {
        return [pscustomobject]@{ valid = $false; required = $true; reason = "training marker is missing" }
    }
    $settingsValid =
        (Test-ObjectProperty $Marker "aerial_diagnostic_required") -and [bool]$Marker.aerial_diagnostic_required -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_steps") -and [int]$Marker.aerial_diagnostic_steps -eq $AerialDiagnosticSteps -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_growth_stop_iter") -and [int]$Marker.aerial_diagnostic_growth_stop_iter -eq $AerialDiagnosticGrowthStopIter -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_max_splats") -and [int]$Marker.aerial_diagnostic_max_splats -eq $AerialDiagnosticMaxSplats -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_max_resolution") -and [int]$Marker.aerial_diagnostic_max_resolution -eq $AerialDiagnosticMaxResolution -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_eval_split_every") -and [int]$Marker.aerial_diagnostic_eval_split_every -eq $AerialDiagnosticEvalSplitEvery -and
        (Test-ObjectProperty $Marker "aerial_sentinel_selector_sha256") -and [string]$Marker.aerial_sentinel_selector_sha256 -eq $AerialSentinelSelectorSha256 -and
        (Test-ObjectProperty $Marker "aerial_gate_evaluator_sha256") -and [string]$Marker.aerial_gate_evaluator_sha256 -eq $AerialGateEvaluatorSha256

    $sentinel = Get-MarkerEvidenceFileState $Marker "aerial_sentinel_manifest_path" "aerial_sentinel_manifest_sha256"
    $diagnosticQuality = Get-MarkerEvidenceFileState $Marker "aerial_diagnostic_quality_report_path" "aerial_diagnostic_quality_report_sha256"
    $diagnosticPly = Get-MarkerEvidenceFileState $Marker "aerial_diagnostic_ply_report_path" "aerial_diagnostic_ply_report_sha256"
    $diagnosticGate = Get-MarkerEvidenceFileState $Marker "aerial_diagnostic_gate_report_path" "aerial_diagnostic_gate_report_sha256"
    $finalGate = Get-MarkerEvidenceFileState $Marker "aerial_final_gate_report_path" "aerial_final_gate_report_sha256"

    $sentinelPayload = if ($sentinel.valid) {
        try { Get-Content -LiteralPath $sentinel.path -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $diagnosticQualityPayload = if ($diagnosticQuality.valid) {
        try { Get-Content -LiteralPath $diagnosticQuality.path -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $diagnosticPlyPayload = if ($diagnosticPly.valid) {
        try { Get-Content -LiteralPath $diagnosticPly.path -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $diagnosticGatePayload = if ($diagnosticGate.valid) {
        try { Get-Content -LiteralPath $diagnosticGate.path -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $finalGatePayload = if ($finalGate.valid) {
        try { Get-Content -LiteralPath $finalGate.path -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }

    $diagnosticGateInputsValid = $false
    $finalGateInputsValid = $false
    $diagnosticGateSentinelsValid = $false
    $finalGateSentinelsValid = $false
    if ($sentinelPayload -and $diagnosticGatePayload -and $finalGatePayload) {
        try {
            $expectedCloseStems = @($sentinelPayload.close_images | ForEach-Object {
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_)
            })
            $expectedControlStems = @($sentinelPayload.control_images | ForEach-Object {
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_)
            })
            $diagnosticCloseStems = @($diagnosticGatePayload.close_images | ForEach-Object {
                if (-not (Test-ObjectProperty $_ "image")) { throw "Missing diagnostic close image" }
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_.image)
            })
            $diagnosticControlStems = @($diagnosticGatePayload.control_images | ForEach-Object {
                if (-not (Test-ObjectProperty $_ "image")) { throw "Missing diagnostic control image" }
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_.image)
            })
            $finalCloseStems = @($finalGatePayload.close_images | ForEach-Object {
                if (-not (Test-ObjectProperty $_ "image")) { throw "Missing final close image" }
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_.image)
            })
            $finalControlStems = @($finalGatePayload.control_images | ForEach-Object {
                if (-not (Test-ObjectProperty $_ "image")) { throw "Missing final control image" }
                [System.IO.Path]::GetFileNameWithoutExtension([string]$_.image)
            })
            $diagnosticGateSentinelsValid =
                ($expectedCloseStems -join "`n") -eq ($diagnosticCloseStems -join "`n") -and
                ($expectedControlStems -join "`n") -eq ($diagnosticControlStems -join "`n")
            $finalGateSentinelsValid =
                ($expectedCloseStems -join "`n") -eq ($finalCloseStems -join "`n") -and
                ($expectedControlStems -join "`n") -eq ($finalControlStems -join "`n")

            $diagnosticInput = $diagnosticGatePayload.input_evidence
            $diagnosticGateInputsValid =
                (Test-ObjectProperty $diagnosticGatePayload "input_evidence") -and
                (Test-ObjectProperty $diagnosticInput "quality_report") -and
                (Test-ObjectProperty $diagnosticInput "ply_report") -and
                (Test-ObjectProperty $diagnosticInput "ply") -and
                [string]$diagnosticInput.quality_report.sha256 -eq [string]$diagnosticQuality.sha256 -and
                [string]$diagnosticInput.ply_report.sha256 -eq [string]$diagnosticPly.sha256 -and
                [string]$diagnosticInput.ply.sha256 -eq [string]$diagnosticPlyPayload.sha256

            $finalInput = $finalGatePayload.input_evidence
            $finalPlyReportPath = [System.IO.Path]::GetFullPath([string]$finalInput.ply_report.path)
            $runPrefix = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd([char[]]"\/") + [System.IO.Path]::DirectorySeparatorChar
            $finalPlyReportSha = if ($finalPlyReportPath.StartsWith($runPrefix, [StringComparison]::OrdinalIgnoreCase) -and
                (Test-Path -LiteralPath $finalPlyReportPath -PathType Leaf)) {
                (Get-FileHash -LiteralPath $finalPlyReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
            } else { $null }
            $finalGateInputsValid =
                (Test-ObjectProperty $finalGatePayload "input_evidence") -and
                (Test-ObjectProperty $finalInput "quality_report") -and
                (Test-ObjectProperty $finalInput "ply_report") -and
                (Test-ObjectProperty $finalInput "ply") -and
                (Test-ObjectProperty $Marker "quality_report_sha256") -and
                (Test-ObjectProperty $Marker "final_ply_sha256") -and
                [string]$finalInput.quality_report.sha256 -eq [string]$Marker.quality_report_sha256 -and
                [string]$finalInput.ply.sha256 -eq [string]$Marker.final_ply_sha256 -and
                [string]$finalInput.ply_report.sha256 -eq $finalPlyReportSha
        } catch {
            $diagnosticGateInputsValid = $false
            $finalGateInputsValid = $false
            $diagnosticGateSentinelsValid = $false
            $finalGateSentinelsValid = $false
        }
    }

    $sentinelValid = $sentinelPayload -and
        (Test-ObjectProperty $sentinelPayload "close_images") -and @($sentinelPayload.close_images).Count -eq 8 -and
        (Test-ObjectProperty $sentinelPayload "control_images") -and @($sentinelPayload.control_images).Count -eq 4
    $diagnosticQualityValid = $diagnosticQualityPayload -and
        (Test-ObjectProperty $diagnosticQualityPayload "quality_status") -and
        [string]$diagnosticQualityPayload.quality_status -eq "measured_unrated" -and
        (Test-ObjectProperty $diagnosticQualityPayload "render_directory") -and
        (Test-ObjectProperty $Marker "aerial_diagnostic_holdout_render_set_sha256")
    $diagnosticHoldoutHash = if ($diagnosticQualityValid -and
        (Test-Path -LiteralPath ([string]$diagnosticQualityPayload.render_directory) -PathType Container)) {
        Get-FileSetHash ([string]$diagnosticQualityPayload.render_directory) "*.png"
    } else { $null }
    $diagnosticPlyValid = $diagnosticPlyPayload -and
        (Test-ObjectProperty $diagnosticPlyPayload "valid_gaussian_ply") -and [bool]$diagnosticPlyPayload.valid_gaussian_ply -and
        (Test-ObjectProperty $diagnosticPlyPayload "nonfinite_vertices") -and [int]$diagnosticPlyPayload.nonfinite_vertices -eq 0
    $diagnosticGateValid = $diagnosticGatePayload -and
        (Test-ObjectProperty $diagnosticGatePayload "quality_status") -and
        [string]$diagnosticGatePayload.quality_status -eq "MECHANICAL_PASS__AWAITING_VISUAL_QC" -and
        $diagnosticGateInputsValid -and $diagnosticGateSentinelsValid
    $finalGateValid = $finalGatePayload -and
        (Test-ObjectProperty $finalGatePayload "quality_status") -and
        [string]$finalGatePayload.quality_status -eq "MECHANICAL_PASS__AWAITING_VISUAL_QC" -and
        $finalGateInputsValid -and $finalGateSentinelsValid

    $valid = [bool]($settingsValid -and $sentinel.valid -and $sentinelValid -and
        $diagnosticQuality.valid -and $diagnosticQualityValid -and
        [string]$Marker.aerial_diagnostic_holdout_render_set_sha256 -eq $diagnosticHoldoutHash -and
        $diagnosticPly.valid -and $diagnosticPlyValid -and
        $diagnosticGate.valid -and $diagnosticGateValid -and
        $finalGate.valid -and $finalGateValid)
    return [pscustomobject]@{
        valid = $valid
        required = $true
        sentinel = $sentinel
        diagnostic_quality = $diagnosticQuality
        diagnostic_ply = $diagnosticPly
        diagnostic_gate = $diagnosticGate
        final_gate = $finalGate
        diagnostic_holdout_sha256 = $diagnosticHoldoutHash
        diagnostic_status = if ($diagnosticGatePayload) { [string]$diagnosticGatePayload.quality_status } else { $null }
        final_status = if ($finalGatePayload) { [string]$finalGatePayload.quality_status } else { $null }
    }
}

function Get-ThreeDGrutAerialEvidenceState($Marker) {
    if (-not $RequiresThreeDGrutAerialStages) {
        return [pscustomobject]@{ valid = $true; required = $false }
    }
    return [pscustomobject]@{
        valid = $false
        required = $true
        approval_code = "THREEDGRUT_VISUAL_APPROVAL_REQUIRED"
        status = "AWAITING_VISUAL_QC"
        reason = "Legacy 3DGRUT diagnostic/final markers are not approved evidence. Only the fresh 1K smoke may run, and it cannot authorize reuse or downstream handoff."
    }
}

function Get-SourceTreeHash([string]$Path) {
    $root = [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]"\/")
    $manifest = Get-ChildItem -LiteralPath $root -Recurse -File |
        Where-Object {
            $_.Extension -in @(".py", ".yaml", ".yml", ".toml", ".txt", ".lock", ".json", ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".pyd", ".dll", ".so") -and
            $_.FullName -notmatch "[\\/](\.git|\.venv|venv|build|dist|outputs?)[\\/]"
        } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($root.Length).TrimStart([char[]]"\/")
            $fileHash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$relative|$fileHash"
        }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($manifest -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-TextSha256([string[]]$Lines) {
    $bytes = [Text.Encoding]::UTF8.GetBytes(($Lines -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-ColmapModelHash([string]$Path) {
    $manifest = foreach ($name in @("cameras.bin", "images.bin", "points3D.bin")) {
        $modelFile = Join-Path $Path $name
        if (-not (Test-Path -LiteralPath $modelFile -PathType Leaf)) { return $null }
        $fileHash = (Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash.ToLowerInvariant()
        "$name|$fileHash"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($manifest -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Get-ToolVersion {
    param([string]$FilePath, [string[]]$Arguments)
    $versionLines = & $FilePath @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Could not read tool version from $FilePath" }
    return (($versionLines | ForEach-Object { "$_" }) -join "`n").Trim()
}

$ColmapVersionText = Get-ToolVersion $ColmapPath @("version")
$ColmapVersion = ([regex]::Match($ColmapVersionText, "(?im)^COLMAP\s+[^\r\n]+" )).Value.Trim()
if (-not $ColmapVersion) { $ColmapVersion = ($ColmapVersionText -split "`r?`n")[0].Trim() }
$ColmapSha256 = (Get-FileHash -LiteralPath $ColmapPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ffmpegVersionText = Get-ToolVersion $FfmpegPath @("-version")
$FfmpegVersion = (($ffmpegVersionText -split "`r?`n")[0]).Trim()
$ffprobeVersionText = Get-ToolVersion $FfprobePath @("-version")
$FfprobeVersion = (($ffprobeVersionText -split "`r?`n")[0]).Trim()
$FfmpegSha256 = (Get-FileHash -LiteralPath $FfmpegPath -Algorithm SHA256).Hash.ToLowerInvariant()
$FfprobeSha256 = (Get-FileHash -LiteralPath $FfprobePath -Algorithm SHA256).Hash.ToLowerInvariant()
$BrushVersion = if ($BrushPath) {
    try {
        $brushVersionText = Get-ToolVersion $BrushPath @("--version")
        (($brushVersionText -split "`r?`n")[0]).Trim()
    } catch { "version unavailable" }
} else { $null }
$BrushSha256 = if ($BrushPath) {
    (Get-FileHash -LiteralPath $BrushPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$SpirulaVersion = if ($SpirulaPath) {
    $spirulaVersionText = Get-ToolVersion $SpirulaPath @("--help")
    (($spirulaVersionText -split "`r?`n")[0]).Trim()
} else { $null }
$SpirulaSha256 = if ($SpirulaPath) {
    (Get-FileHash -LiteralPath $SpirulaPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$BlenderBuilderVersion = if ($BlenderBuilderPath) {
    $blenderBuilderVersionText = Get-ToolVersion $BlenderBuilderPath @("--version")
    (($blenderBuilderVersionText -split "`r?`n")[0]).Trim()
} else { $null }
$BlenderBuilderSha256 = if ($BlenderBuilderPath) { (Get-FileHash -LiteralPath $BlenderBuilderPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$BlenderOpenVersion = if ($BlenderOpenPath) {
    $blenderOpenVersionText = Get-ToolVersion $BlenderOpenPath @("--version")
    (($blenderOpenVersionText -split "`r?`n")[0]).Trim()
} else { $null }
$BlenderOpenSha256 = if ($BlenderOpenPath) { (Get-FileHash -LiteralPath $BlenderOpenPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$MatcherName = if ($UsesChronologicalMatching) { "chronological sequential/SIFT_BRUTEFORCE with vocabulary loop closure" } else { "exhaustive/SIFT_BRUTEFORCE" }
$MapperName = if ($UsesIncrementalMapper) { "incremental mapper" } else { "global_mapper with view_graph_calibrator" }
$EvaluateHoldoutPath = Join-Path $PSScriptRoot "evaluate_holdout.py"
$EvaluateSpirulaHoldoutPath = Join-Path $PSScriptRoot "evaluate_spirula_holdout.py"
$AerialSentinelSelectorPath = Join-Path $PSScriptRoot "select_aerial_diagnostic_sentinels.py"
$AerialGateEvaluatorPath = Join-Path $PSScriptRoot "evaluate_aerial_diagnostic_gate.py"
$ThreeDGrutStageEvaluatorPath = Join-Path $PSScriptRoot "evaluate_3dgrut_stage.py"
$ThreeDGrutSmokePreparerPath = Join-Path $PSScriptRoot "prepare_3dgrut_smoke_dataset.py"
$PlyVerifierPath = Join-Path $PSScriptRoot "verify_gaussian_ply.py"
$SelectorScriptPath = Join-Path $PSScriptRoot "select_frames.py"
$PoseValidatorPath = Join-Path $PSScriptRoot "validate_colmap_poses.py"
$ImageOrderValidatorPath = Join-Path $PSScriptRoot "validate_colmap_image_order.py"
$EvaluatorSha256 = if ($Trainer -eq "Spirula") {
    (Get-FileHash -LiteralPath $EvaluateSpirulaHoldoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else {
    (Get-FileHash -LiteralPath $EvaluateHoldoutPath -Algorithm SHA256).Hash.ToLowerInvariant()
}
$AerialSentinelSelectorSha256 = if ($RequiresAerialDiagnostic) {
    (Get-FileHash -LiteralPath $AerialSentinelSelectorPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$AerialGateEvaluatorSha256 = if ($RequiresAerialDiagnostic) {
    (Get-FileHash -LiteralPath $AerialGateEvaluatorPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$ThreeDGrutStageEvaluatorSha256 = if ($RequiresThreeDGrutAerialStages) {
    (Get-FileHash -LiteralPath $ThreeDGrutStageEvaluatorPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$ThreeDGrutSmokePreparerSha256 = if ($RequiresThreeDGrutAerialStages) {
    (Get-FileHash -LiteralPath $ThreeDGrutSmokePreparerPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$PlyVerifierSha256 = (Get-FileHash -LiteralPath $PlyVerifierPath -Algorithm SHA256).Hash.ToLowerInvariant()
$SelectorScriptSha256 = (Get-FileHash -LiteralPath $SelectorScriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$PoseValidatorSha256 = (Get-FileHash -LiteralPath $PoseValidatorPath -Algorithm SHA256).Hash.ToLowerInvariant()
$ImageOrderValidatorSha256 = (Get-FileHash -LiteralPath $ImageOrderValidatorPath -Algorithm SHA256).Hash.ToLowerInvariant()
$RunnerScriptSha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
$VocabTreeSha256 = if ($UsesChronologicalMatching -and $VocabTreePath) {
    (Get-FileHash -LiteralPath $VocabTreePath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$QualityReportPath = Join-Path $RunRoot "training_quality_report.json"

$ResolvedShotManifestPath = $null
$ShotManifestSha256 = $null
if ($ShotManifestPath) {
    $shotManifestCandidate = [System.IO.Path]::GetFullPath($ShotManifestPath)
    if (-not (Test-Path -LiteralPath $shotManifestCandidate -PathType Leaf)) {
        throw "Shot manifest not found: $shotManifestCandidate"
    }
    $ResolvedShotManifestPath = (Get-Item -LiteralPath $shotManifestCandidate).FullName
    $ShotManifestSha256 = (Get-FileHash -LiteralPath $ResolvedShotManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ShotManifestContractState($SelectState) {
    if (-not $ResolvedShotManifestPath) {
        return [pscustomobject]@{ valid = $true; reason = $null; manifest = $null }
    }
    try {
        if (-not (Test-Path -LiteralPath $ResolvedShotManifestPath -PathType Leaf)) {
            return [pscustomobject]@{ valid = $false; reason = "the configured file no longer exists"; manifest = $null }
        }
        $currentHash = (Get-FileHash -LiteralPath $ResolvedShotManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($currentHash -ne $ShotManifestSha256) {
            return [pscustomobject]@{ valid = $false; reason = "the configured file changed after it was signed"; manifest = $null }
        }
        $shotManifest = Get-Content -LiteralPath $ResolvedShotManifestPath -Raw | ConvertFrom-Json
        foreach ($requiredProperty in @("source_sha256", "selected_image_set_sha256", "candidate_frame_count")) {
            if (-not (Test-ObjectProperty $shotManifest $requiredProperty)) {
                return [pscustomobject]@{ valid = $false; reason = "required property '$requiredProperty' is missing"; manifest = $shotManifest }
            }
        }
        if ([string]$shotManifest.source_sha256 -ne $SourceSha256) {
            return [pscustomobject]@{ valid = $false; reason = "source_sha256 does not match the exact input video"; manifest = $shotManifest }
        }
        if ($null -ne $SelectState) {
            if ([int64]$shotManifest.candidate_frame_count -ne [int64]$SelectState.extract.count) {
                return [pscustomobject]@{ valid = $false; reason = "candidate_frame_count does not match the extracted candidate frame set"; manifest = $shotManifest }
            }
            if ([string]$shotManifest.selected_image_set_sha256 -ne [string]$SelectState.hash) {
                return [pscustomobject]@{ valid = $false; reason = "selected_image_set_sha256 does not match the exact selected images"; manifest = $shotManifest }
            }
        }
        return [pscustomobject]@{ valid = $true; reason = $null; manifest = $shotManifest }
    } catch {
        return [pscustomobject]@{ valid = $false; reason = "the JSON contract could not be read: $($_.Exception.Message)"; manifest = $null }
    }
}

function Assert-ShotManifestContract($SelectState) {
    $state = Get-ShotManifestContractState $SelectState
    if (-not $state.valid) {
        throw "Shot manifest contract failed for '$ResolvedShotManifestPath': $($state.reason)"
    }
    return $state
}

Assert-ShotManifestContract $null | Out-Null

$ThreeDGrutSourceSha256 = $null
$ThreeDGrutPythonSha256 = $null
$ThreeDGrutConfigSha256 = $null
$ThreeDGrutCommit = $null
if ($Trainer -in @("3DGUT", "3DGUT-MCMC")) {
    $ThreeDGrutSourceSha256 = Get-SourceTreeHash $ThreeDGrutRepo
    $ThreeDGrutPythonSha256 = (Get-FileHash -LiteralPath $ThreeDGrutPython -Algorithm SHA256).Hash.ToLowerInvariant()
    $ThreeDGrutConfigSha256 = (Get-FileHash -LiteralPath $ThreeDGrutModelConfig -Algorithm SHA256).Hash.ToLowerInvariant()
    $git = Get-Command git.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($git -and (Test-Path -LiteralPath (Join-Path $ThreeDGrutRepo ".git"))) {
        $commitLines = & $git.Source -C $ThreeDGrutRepo rev-parse HEAD 2>$null
        if ($LASTEXITCODE -eq 0) { $ThreeDGrutCommit = ($commitLines -join "").Trim() }
    }
}

function Get-ExtractChainState {
    $marker = Read-Marker "extract"
    $count = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    $hash = if ($count -gt 0) { Get-ImageSetHash $RawFrames } else { $null }
    $valid = [bool]($marker -and $count -gt 0 -and
        (Test-ObjectProperty $marker "source_bytes") -and [int64]$marker.source_bytes -eq $SourceBytes -and
        (Test-ObjectProperty $marker "source_last_write_utc") -and [string]$marker.source_last_write_utc -eq $SourceLastWriteUtc -and
        (Test-ObjectProperty $marker "source_sha256") -and [string]$marker.source_sha256 -eq $SourceSha256 -and
        (Test-ObjectProperty $marker "selected_frame_target") -and [int]$marker.selected_frame_target -eq $SelectedFrames -and
        (Test-ObjectProperty $marker "adaptive_extraction") -and [bool]$marker.adaptive_extraction -eq [bool]$AdaptiveExtraction -and
        (Test-ObjectProperty $marker "candidate_multiplier") -and [int]$marker.candidate_multiplier -eq $CandidateMultiplier -and
        (Test-ObjectProperty $marker "max_long_side") -and [int]$marker.max_long_side -eq $MaxLongSide -and
        (Test-ObjectProperty $marker "auto_rotate") -and [bool]$marker.auto_rotate -eq (-not [bool]$NoAutoRotate) -and
        (Test-ObjectProperty $marker "raw_frames") -and [int]$marker.raw_frames -eq $count -and
        (Test-ObjectProperty $marker "raw_frame_hash") -and [string]$marker.raw_frame_hash -eq $hash -and
        (Test-ObjectProperty $marker "ffmpeg_sha256") -and [string]$marker.ffmpeg_sha256 -eq $FfmpegSha256 -and
        (Test-ObjectProperty $marker "ffprobe_sha256") -and [string]$marker.ffprobe_sha256 -eq $FfprobeSha256)
    return [pscustomobject]@{ valid = $valid; marker = $marker; count = $count; hash = $hash }
}

function Get-SelectChainState($ExtractState) {
    if ($null -eq $ExtractState) { $ExtractState = Get-ExtractChainState }
    $marker = Read-Marker "select"
    $count = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    $hash = if ($count -gt 0) { Get-ImageSetHash $ImagesPath } else { $null }
    $minimumCount = [Math]::Min($SelectedFrames, [int]$ExtractState.count)
    # Adaptive flow sampling treats SelectedFrames as a minimum; the marker and
    # quality-report hashes still bind the exact output set for fail-closed resume.
    $countValid = if ($MaxCumulativeFlow -gt 0.0) {
        $count -ge $minimumCount -and $count -le [int]$ExtractState.count
    } else {
        $count -eq $minimumCount
    }
    $frameQualitySha = if (Test-Path -LiteralPath $FrameQualityPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $FrameQualityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $frameQuality = if ($frameQualitySha) {
        try { Get-Content -LiteralPath $FrameQualityPath -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $frameQualityValid = $frameQuality -and
        (Test-ObjectProperty $frameQuality "selected_frames") -and [int]$frameQuality.selected_frames -eq $count -and
        (Test-ObjectProperty $frameQuality "max_cumulative_flow") -and [double]$frameQuality.max_cumulative_flow -eq $MaxCumulativeFlow
    $valid = [bool]($ExtractState.valid -and $marker -and $countValid -and $frameQualityValid -and
        (Test-ObjectProperty $marker "source_frames") -and [int]$marker.source_frames -eq [int]$ExtractState.count -and
        (Test-ObjectProperty $marker "raw_frame_hash") -and [string]$marker.raw_frame_hash -eq [string]$ExtractState.hash -and
        (Test-ObjectProperty $marker "requested_frames") -and [int]$marker.requested_frames -eq $SelectedFrames -and
        (Test-ObjectProperty $marker "selected_frames") -and [int]$marker.selected_frames -eq $count -and
        (Test-ObjectProperty $marker "max_cumulative_flow") -and [double]$marker.max_cumulative_flow -eq $MaxCumulativeFlow -and
        (Test-ObjectProperty $marker "frame_quality_sha256") -and [string]$marker.frame_quality_sha256 -eq $frameQualitySha -and
        (Test-ObjectProperty $marker "selected_frame_hash") -and [string]$marker.selected_frame_hash -eq $hash -and
        (Test-ObjectProperty $marker "scene_type") -and [string]$marker.scene_type -eq $SceneType -and
        (Test-ObjectProperty $marker "selector_version") -and [string]$marker.selector_version -eq $SelectorVersion -and
        (Test-ObjectProperty $marker "selector_script_sha256") -and [string]$marker.selector_script_sha256 -eq $SelectorScriptSha256)
    return [pscustomobject]@{
        valid = $valid; marker = $marker; count = $count; hash = $hash
        frame_quality_sha256 = $frameQualitySha; extract = $ExtractState
    }
}

function Get-SolveChainState($SelectState) {
    if ($null -eq $SelectState) { $SelectState = Get-SelectChainState $null }
    $marker = Read-Marker "solve"
    $modelHash = Get-ColmapModelHash $SparsePath
    $database = if ($UsesIncrementalMapper) { $DatabasePath } else { $CalibratedDatabasePath }
    $databaseSha = if (Test-Path -LiteralPath $database -PathType Leaf) {
        (Get-FileHash -LiteralPath $database -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $reportPath = Join-Path $RunRoot "reconstruction_report.json"
    $reportSha = if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $camerasTextPath = Join-Path $ModelTextRoot "cameras.txt"
    $imagesTextPath = Join-Path $ModelTextRoot "images.txt"
    $pointsTextPath = Join-Path $ModelTextRoot "points3D.txt"
    $camerasTextSha = if (Test-Path -LiteralPath $camerasTextPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $camerasTextPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $imagesTextSha = if (Test-Path -LiteralPath $imagesTextPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $imagesTextPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $pointsTextSha = if (Test-Path -LiteralPath $pointsTextPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $pointsTextPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $modelTextValid = $marker -and $camerasTextSha -and $imagesTextSha -and $pointsTextSha -and
        (Test-ObjectProperty $marker "cameras_text_sha256") -and [string]$marker.cameras_text_sha256 -eq $camerasTextSha -and
        (Test-ObjectProperty $marker "images_text_sha256") -and [string]$marker.images_text_sha256 -eq $imagesTextSha -and
        (Test-ObjectProperty $marker "points3d_text_sha256") -and [string]$marker.points3d_text_sha256 -eq $pointsTextSha
    $shotManifestState = Get-ShotManifestContractState $SelectState
    $imageOrderSha = if ($UsesChronologicalMatching -and (Test-Path -LiteralPath $ImageOrderPath -PathType Leaf)) {
        (Get-FileHash -LiteralPath $ImageOrderPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $imageOrderValid = if ($UsesChronologicalMatching) {
        $marker -and
        (Test-ObjectProperty $marker "image_order_sha256") -and [string]$marker.image_order_sha256 -eq [string]$imageOrderSha -and
        (Test-ObjectProperty $marker "image_order_validator_sha256") -and [string]$marker.image_order_validator_sha256 -eq $ImageOrderValidatorSha256
    } else { $true }
    $valid = [bool]($SelectState.valid -and $shotManifestState.valid -and $imageOrderValid -and $modelTextValid -and $marker -and $modelHash -and $databaseSha -and $reportSha -and
        (Test-ObjectProperty $marker "quality_gates_pass") -and [bool]$marker.quality_gates_pass -and
        (Test-ObjectProperty $marker "image_set_hash") -and [string]$marker.image_set_hash -eq [string]$SelectState.hash -and
        (Test-ObjectProperty $marker "shot_manifest_path") -and [string]$marker.shot_manifest_path -eq [string]$ResolvedShotManifestPath -and
        (Test-ObjectProperty $marker "shot_manifest_sha256") -and [string]$marker.shot_manifest_sha256 -eq [string]$ShotManifestSha256 -and
        (Test-ObjectProperty $marker "model_hash") -and [string]$marker.model_hash -eq $modelHash -and
        (Test-ObjectProperty $marker "scene_type") -and [string]$marker.scene_type -eq $SceneType -and
        (Test-ObjectProperty $marker "matcher") -and [string]$marker.matcher -eq $MatcherName -and
        (Test-ObjectProperty $marker "mapper") -and [string]$marker.mapper -eq $MapperName -and
        (Test-ObjectProperty $marker "selector_version") -and [string]$marker.selector_version -eq $SelectorVersion -and
        (Test-ObjectProperty $marker "gate_version") -and [string]$marker.gate_version -eq $GateVersion -and
        (Test-ObjectProperty $marker "colmap_version") -and [string]$marker.colmap_version -eq $ColmapVersion -and
        (Test-ObjectProperty $marker "colmap_sha256") -and [string]$marker.colmap_sha256 -eq $ColmapSha256 -and
        (Test-ObjectProperty $marker "pose_validator_sha256") -and [string]$marker.pose_validator_sha256 -eq $PoseValidatorSha256 -and
        (Test-ObjectProperty $marker "runner_script_sha256") -and [string]$marker.runner_script_sha256 -eq $RunnerScriptSha256 -and
        (Test-ObjectProperty $marker "database_sha256") -and [string]$marker.database_sha256 -eq $databaseSha -and
        (Test-ObjectProperty $marker "reconstruction_report_sha256") -and [string]$marker.reconstruction_report_sha256 -eq $reportSha -and
        (Test-ObjectProperty $marker "vocab_tree_sha256") -and [string]$marker.vocab_tree_sha256 -eq [string]$VocabTreeSha256)
    return [pscustomobject]@{
        valid = $valid; marker = $marker; model_hash = $modelHash; database_sha256 = $databaseSha
        cameras_text_sha256 = $camerasTextSha; images_text_sha256 = $imagesTextSha; points3d_text_sha256 = $pointsTextSha
        report_sha256 = $reportSha; select = $SelectState; shot_manifest = $shotManifestState
    }
}

function Get-TrainChainState($SolveState) {
    if ($Trainer -notin @("Brush", "Spirula")) {
        return [pscustomobject]@{
            valid = $false
            marker = Read-Marker "train"
            ply_sha256 = $null
            ply_report = $null
            quality = $null
            quality_sha256 = $null
            holdout_sha256 = $null
            solve = $SolveState
            aerial_diagnostic = [pscustomobject]@{ valid = $false; required = $false }
            three_dgrut_aerial = Get-ThreeDGrutAerialEvidenceState $null
            approval_code = "THREEDGRUT_VISUAL_APPROVAL_REQUIRED"
        }
    }
    if ($null -eq $SolveState) { $SolveState = Get-SolveChainState $null }
    $marker = Read-Marker "train"
    $plyReportPath = Join-Path $RunRoot "gaussian_ply_report.json"
    $plyReport = if (Test-Path -LiteralPath $plyReportPath -PathType Leaf) {
        try { Get-Content -LiteralPath $plyReportPath -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $plySha = if (Test-Path -LiteralPath $FinalPly -PathType Leaf) {
        (Get-FileHash -LiteralPath $FinalPly -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $plyValid = [bool]($plyReport -and (Test-ObjectProperty $plyReport "valid_gaussian_ply") -and
        [bool]$plyReport.valid_gaussian_ply -and (Test-ObjectProperty $plyReport "sha256") -and
        [string]$plyReport.sha256 -eq $plySha)
    $qualitySha = if (Test-Path -LiteralPath $QualityReportPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $QualityReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $quality = if ($qualitySha) {
        try { Get-Content -LiteralPath $QualityReportPath -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $holdoutHash = $null
    $holdoutReferenceHash = $null
    $qualityValid = $false
    if ($quality -and (Test-ObjectProperty $quality "render_directory") -and
        (Test-ObjectProperty $quality "expected_holdout_renders") -and (Test-ObjectProperty $quality "saved_holdout_renders")) {
        $renderDirectory = [string]$quality.render_directory
        if (Test-Path -LiteralPath $renderDirectory -PathType Container) {
            $renderCount = @(Get-ChildItem -LiteralPath $renderDirectory -Filter "*.png").Count
            if ($renderCount -eq [int]$quality.saved_holdout_renders) {
                $holdoutHash = Get-FileSetHash $renderDirectory "*.png"
                $qualityValid = (Test-ObjectProperty $quality "quality_status") -and
                    [string]$quality.quality_status -eq "measured_unrated" -and
                    [int]$quality.expected_holdout_renders -eq [int]$quality.saved_holdout_renders
            }
        }
    }
    if ($Trainer -eq "Spirula" -and $quality -and (Test-ObjectProperty $quality "reference_directory") -and
        (Test-Path -LiteralPath ([string]$quality.reference_directory) -PathType Container)) {
        $referenceDirectory = [string]$quality.reference_directory
        $referenceCount = @(Get-ChildItem -LiteralPath $referenceDirectory -Filter "*.png").Count
        if ($referenceCount -eq [int]$quality.saved_holdout_renders) {
            $holdoutReferenceHash = Get-FileSetHash $referenceDirectory "*.png"
        } else {
            $qualityValid = $false
        }
    }
    $spirulaMetricsSha = $null
    if ($Trainer -eq "Spirula" -and $quality -and (Test-ObjectProperty $quality "native_metrics_path") -and
        (Test-Path -LiteralPath ([string]$quality.native_metrics_path) -PathType Leaf)) {
        $spirulaMetricsSha = (Get-FileHash -LiteralPath ([string]$quality.native_metrics_path) -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $spirulaConfigSha = $null
    if ($Trainer -eq "Spirula" -and $marker -and (Test-ObjectProperty $marker "spirula_config_path") -and
        (Test-Path -LiteralPath ([string]$marker.spirula_config_path) -PathType Leaf)) {
        $spirulaConfigSha = (Get-FileHash -LiteralPath ([string]$marker.spirula_config_path) -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $trainerValid = if ($Trainer -eq "Brush") {
        $marker -and (Test-ObjectProperty $marker "brush_version") -and [string]$marker.brush_version -eq $BrushVersion -and
            (Test-ObjectProperty $marker "brush_sha256") -and [string]$marker.brush_sha256 -eq $BrushSha256 -and
            (Test-ObjectProperty $marker "brush_max_splats") -and [int]$marker.brush_max_splats -eq $BrushMaxSplats -and
            (Test-ObjectProperty $marker "brush_scale_loss_weight") -and [double]$marker.brush_scale_loss_weight -eq $BrushScaleLossWeight -and
            (Test-ObjectProperty $marker "seed") -and [int]$marker.seed -eq $TrainingSeed -and
            (Test-ObjectProperty $marker "evaluator_sha256") -and [string]$marker.evaluator_sha256 -eq $EvaluatorSha256 -and
            (Test-ObjectProperty $marker "holdout_render_set_sha256") -and [string]$marker.holdout_render_set_sha256 -eq $holdoutHash
    } else {
        $marker -and $holdoutReferenceHash -and $spirulaMetricsSha -and $spirulaConfigSha -and
            (Test-ObjectProperty $marker "spirula_version") -and [string]$marker.spirula_version -eq $SpirulaVersion -and
            (Test-ObjectProperty $marker "spirula_sha256") -and [string]$marker.spirula_sha256 -eq $SpirulaSha256 -and
            (Test-ObjectProperty $marker "spirula_max_splats") -and [int]$marker.spirula_max_splats -eq $BrushMaxSplats -and
            (Test-ObjectProperty $marker "spirula_floater_suppression") -and [string]$marker.spirula_floater_suppression -eq $SpirulaFloaterSuppression -and
            (Test-ObjectProperty $marker "spirula_distraction_robustness") -and [string]$marker.spirula_distraction_robustness -eq $SpirulaDistractionRobustness -and
            (Test-ObjectProperty $marker "spirula_save_full_checkpoint") -and [bool]$marker.spirula_save_full_checkpoint -eq $SpirulaSaveFullCheckpoint -and
            (Test-ObjectProperty $marker "spirula_metrics_sha256") -and [string]$marker.spirula_metrics_sha256 -eq $spirulaMetricsSha -and
            (Test-ObjectProperty $marker "spirula_config_sha256") -and [string]$marker.spirula_config_sha256 -eq $spirulaConfigSha -and
            (Test-ObjectProperty $marker "evaluator_sha256") -and [string]$marker.evaluator_sha256 -eq $EvaluatorSha256 -and
            (Test-ObjectProperty $marker "holdout_render_set_sha256") -and [string]$marker.holdout_render_set_sha256 -eq $holdoutHash -and
            (Test-ObjectProperty $marker "holdout_reference_set_sha256") -and [string]$marker.holdout_reference_set_sha256 -eq $holdoutReferenceHash
    }
    $aerialDiagnostic = Get-AerialDiagnosticEvidenceState $marker
    $threeDGrutAerial = Get-ThreeDGrutAerialEvidenceState $marker
    $valid = [bool]($SolveState.valid -and $plyValid -and $qualityValid -and $marker -and $trainerValid -and $aerialDiagnostic.valid -and $threeDGrutAerial.valid -and
        (Test-ObjectProperty $marker "trainer") -and [string]$marker.trainer -eq $Trainer -and
        (Test-ObjectProperty $marker "training_steps") -and [int]$marker.training_steps -eq $TrainingSteps -and
        (Test-ObjectProperty $marker "training_max_resolution") -and [int]$marker.training_max_resolution -eq $TrainingMaxResolution -and
        (Test-ObjectProperty $marker "eval_split_every") -and [int]$marker.eval_split_every -eq $EvalSplitEvery -and
        (Test-ObjectProperty $marker "scene_type") -and [string]$marker.scene_type -eq $SceneType -and
        (Test-ObjectProperty $marker "image_set_hash") -and [string]$marker.image_set_hash -eq [string]$SolveState.select.hash -and
        (Test-ObjectProperty $marker "solve_model_hash") -and [string]$marker.solve_model_hash -eq [string]$SolveState.model_hash -and
        (Test-ObjectProperty $marker "final_ply_sha256") -and [string]$marker.final_ply_sha256 -eq $plySha -and
        (Test-ObjectProperty $marker "quality_report_sha256") -and [string]$marker.quality_report_sha256 -eq $qualitySha -and
        (Test-ObjectProperty $marker "ply_verifier_sha256") -and [string]$marker.ply_verifier_sha256 -eq $PlyVerifierSha256)
    return [pscustomobject]@{
        valid = $valid; marker = $marker; ply_sha256 = $plySha; ply_report = $plyReport
        quality = $quality; quality_sha256 = $qualitySha; holdout_sha256 = $holdoutHash
        holdout_reference_sha256 = $holdoutReferenceHash; solve = $SolveState
        aerial_diagnostic = $aerialDiagnostic
        three_dgrut_aerial = $threeDGrutAerial
    }
}

function Get-PublishBackupRoot([string]$TransactionAttemptId) {
    if ($TransactionAttemptId -notmatch "^[0-9a-fA-F-]{36}$") { throw "Malformed publish transaction attempt id" }
    $backupRoot = [System.IO.Path]::GetFullPath((Join-Path $RunRoot ".publish-backup-$TransactionAttemptId"))
    $runPrefix = [System.IO.Path]::GetFullPath($RunRoot).TrimEnd([char[]]"\/") + [System.IO.Path]::DirectorySeparatorChar
    if (-not $backupRoot.StartsWith($runPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Publish backup escaped the run directory"
    }
    return $backupRoot
}

function Restore-PublishTransaction {
    $transactionPath = Join-Path $RunRoot "publish_transaction.json"
    if (-not (Test-Path -LiteralPath $transactionPath -PathType Leaf)) { return }
    $state = Get-Content -LiteralPath $transactionPath -Raw | ConvertFrom-Json
    $backupRoot = Get-PublishBackupRoot ([string]$state.attempt_id)
    $targets = [ordered]@{
        final_ply = $FinalPly
        quality_report = $QualityReportPath
        gaussian_report = (Join-Path $RunRoot "gaussian_ply_report.json")
        train_marker = (Join-Path $RunRoot ".train.complete.json")
    }
    foreach ($name in $targets.Keys) {
        $target = [string]$targets[$name]
        $backup = Join-Path $backupRoot "$name.bak"
        $existedProperty = "${name}_existed"
        $previouslyExisted = [bool]$state.$existedProperty
        if ($previouslyExisted) {
            if (Test-Path -LiteralPath $backup -PathType Leaf) {
                if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
                Move-Item -LiteralPath $backup -Destination $target -Force
            } elseif (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                throw "Cannot restore interrupted publish transaction; missing $name backup"
            }
        } elseif (Test-Path -LiteralPath $target -PathType Leaf) {
            Remove-Item -LiteralPath $target -Force
        }
    }
    if (Test-Path -LiteralPath $backupRoot -PathType Container) { Remove-Item -LiteralPath $backupRoot -Recurse -Force }
    Remove-Item -LiteralPath $transactionPath -Force
    Write-Warning "Restored the last complete PLY/report bundle after an interrupted publish."
}

function Start-PublishTransaction {
    $transactionPath = Join-Path $RunRoot "publish_transaction.json"
    if (Test-Path -LiteralPath $transactionPath) { throw "A prior publish transaction must be restored before publishing" }
    $backupRoot = Get-PublishBackupRoot $AttemptId
    New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
    $targets = [ordered]@{
        final_ply = $FinalPly
        quality_report = $QualityReportPath
        gaussian_report = (Join-Path $RunRoot "gaussian_ply_report.json")
        train_marker = (Join-Path $RunRoot ".train.complete.json")
    }
    $state = [ordered]@{ schema_version = 1; attempt_id = $AttemptId; prepared_utc = [DateTime]::UtcNow.ToString("o") }
    foreach ($name in $targets.Keys) {
        $state["${name}_existed"] = Test-Path -LiteralPath ([string]$targets[$name]) -PathType Leaf
    }
    $state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$transactionPath.tmp" -Encoding utf8
    Move-Item -LiteralPath "$transactionPath.tmp" -Destination $transactionPath -Force
    foreach ($name in $targets.Keys) {
        $target = [string]$targets[$name]
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Move-Item -LiteralPath $target -Destination (Join-Path $backupRoot "$name.bak")
        }
    }
}

function Complete-PublishTransaction {
    $transactionPath = Join-Path $RunRoot "publish_transaction.json"
    if (-not (Test-Path -LiteralPath $transactionPath -PathType Leaf)) { throw "Publish transaction state is missing" }
    $state = Get-Content -LiteralPath $transactionPath -Raw | ConvertFrom-Json
    $backupRoot = Get-PublishBackupRoot ([string]$state.attempt_id)
    if (Test-Path -LiteralPath $backupRoot -PathType Container) { Remove-Item -LiteralPath $backupRoot -Recurse -Force }
    Remove-Item -LiteralPath $transactionPath -Force
}

function Publish-VerifiedPly {
    param(
        [string]$SourcePly,
        [string]$CandidateQualityReport,
        [string]$ExpectedPlySha256 = $null,
        [string]$ExpectedQualityReportSha256 = $null
    )
    if (-not (Test-Path -LiteralPath $SourcePly -PathType Leaf)) {
        throw "Training did not produce a candidate PLY: $SourcePly"
    }
    if (-not (Test-Path -LiteralPath $CandidateQualityReport -PathType Leaf)) {
        throw "Training did not produce a candidate quality report: $CandidateQualityReport"
    }
    try { Get-Content -LiteralPath $CandidateQualityReport -Raw | ConvertFrom-Json | Out-Null }
    catch { throw "Candidate quality report is malformed: $CandidateQualityReport" }
    $hasExpectedPly = -not [string]::IsNullOrWhiteSpace($ExpectedPlySha256)
    $hasExpectedQuality = -not [string]::IsNullOrWhiteSpace($ExpectedQualityReportSha256)
    if ($hasExpectedPly -ne $hasExpectedQuality) {
        throw "Expected PLY and quality-report hashes must be supplied together."
    }
    $sourcePlySha = (Get-FileHash -LiteralPath $SourcePly -Algorithm SHA256).Hash.ToLowerInvariant()
    $sourceQualitySha = (Get-FileHash -LiteralPath $CandidateQualityReport -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($hasExpectedPly -and
        ($sourcePlySha -ne $ExpectedPlySha256.ToLowerInvariant() -or
        $sourceQualitySha -ne $ExpectedQualityReportSha256.ToLowerInvariant())) {
        throw "The publish candidate does not match the files authorized by the aerial gate."
    }
    New-Item -ItemType Directory -Force -Path $FinalRoot | Out-Null
    $publishTemp = "$FinalPly.new-$AttemptId"
    $qualityTemp = "$QualityReportPath.new-$AttemptId"
    if (Test-Path -LiteralPath $publishTemp) { Remove-Item -LiteralPath $publishTemp -Force }
    if (Test-Path -LiteralPath $qualityTemp) { Remove-Item -LiteralPath $qualityTemp -Force }
    Copy-Item -LiteralPath $SourcePly -Destination $publishTemp
    Copy-Item -LiteralPath $CandidateQualityReport -Destination $qualityTemp
    if ($hasExpectedPly -and
        ((Get-FileHash -LiteralPath $publishTemp -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourcePlySha -or
        (Get-FileHash -LiteralPath $qualityTemp -Algorithm SHA256).Hash.ToLowerInvariant() -ne $sourceQualitySha)) {
        throw "The aerial publish staging copies do not match the gated inputs."
    }
    $candidateReportPath = Join-Path (Split-Path -Parent $SourcePly) "candidate_ply_report.json"
    Invoke-LoggedCommand $PythonPath @(
        $PlyVerifierPath, $publishTemp, "--json", $candidateReportPath
    ) (Join-Path $LogsRoot "verify_training_candidate.log")
    $candidateReport = Get-Content -LiteralPath $candidateReportPath -Raw | ConvertFrom-Json
    $newSha = [string]$candidateReport.sha256
    if ($hasExpectedPly -and $newSha -ne $ExpectedPlySha256.ToLowerInvariant()) {
        throw "The verified publish PLY does not match the PLY authorized by the aerial gate."
    }

    if (Test-Path -LiteralPath $FinalPly -PathType Leaf) {
        $oldSha = (Get-FileHash -LiteralPath $FinalPly -Algorithm SHA256).Hash.ToLowerInvariant()
        $historyRoot = Join-Path $FinalRoot "history"
        New-Item -ItemType Directory -Force -Path $historyRoot | Out-Null
        $historyPath = Join-Path $historyRoot "$RunName-$($oldSha.Substring(0, 12)).ply"
        if (-not (Test-Path -LiteralPath $historyPath -PathType Leaf)) {
            Copy-Item -LiteralPath $FinalPly -Destination $historyPath
        } elseif ((Get-FileHash -LiteralPath $historyPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $oldSha) {
            throw "PLY history collision at $historyPath"
        }
        $historyEvidenceBase = "$RunName-$($oldSha.Substring(0, 12))-$AttemptId"
        $oldPlyReportPath = Join-Path $RunRoot "gaussian_ply_report.json"
        if (Test-Path -LiteralPath $oldPlyReportPath -PathType Leaf) {
            Copy-Item -LiteralPath $oldPlyReportPath -Destination (Join-Path $historyRoot "$historyEvidenceBase-gaussian_ply_report.json")
        }
        if (Test-Path -LiteralPath $QualityReportPath -PathType Leaf) {
            Copy-Item -LiteralPath $QualityReportPath -Destination (Join-Path $historyRoot "$historyEvidenceBase-training_quality_report.json")
        }
        if (Test-Path -LiteralPath $RunManifestPath -PathType Leaf) {
            Copy-Item -LiteralPath $RunManifestPath -Destination (Join-Path $historyRoot "$historyEvidenceBase-run_manifest.json")
        }
    }
    Start-PublishTransaction
    Move-Item -LiteralPath $publishTemp -Destination $FinalPly -Force
    Move-Item -LiteralPath $qualityTemp -Destination $QualityReportPath -Force
    Remove-Item -LiteralPath $SourcePly -Force
}

function Invoke-ThreeDGrutTrainingStage {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("smoke", "diagnostic", "final")][string]$Stage,
        [Parameter(Mandatory = $true)][int]$Steps,
        [Parameter(Mandatory = $true)][int]$MaxSplats,
        [Parameter(Mandatory = $true)][string]$DatasetPath,
        [Parameter(Mandatory = $true)][string]$StageRoot,
        [Parameter(Mandatory = $true)][string]$DatasetImageSha256,
        [Parameter(Mandatory = $true)][string]$DatasetModelSha256,
        [string]$ResumeCheckpoint,
        [string]$BaselineReport
    )
    New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null
    $candidatePly = Join-Path $StageRoot "$RunName-$Stage-$Steps.ply"
    $arguments = @(
        "train.py", "--config-name", $ThreeDGrutConfigName, "path=$DatasetPath",
        "out_dir=$StageRoot", "experiment_name=$RunName-$Stage-$Steps", "n_iterations=$Steps",
        "seed_initialization=$TrainingSeed", "strategy.add.max_n_gaussians=$MaxSplats",
        "with_gui=false", "with_viser_gui=false", "test_last=true", "compute_extra_metrics=true",
        "log_frequency=500", "val_frequency=999999",
        "export_ply.enabled=true", "export_ply.path=$candidatePly", "checkpoint.iterations=[$Steps]",
        "hydra.run.dir=$(Join-Path $StageRoot 'hydra')"
    )
    if ($ResumeCheckpoint) { $arguments += "resume=$ResumeCheckpoint" }
    Invoke-LoggedCommand $ThreeDGrutPython $arguments (Join-Path $LogsRoot "3dgrut_$Stage.log") $ThreeDGrutRepo | Out-Host
    if (-not (Test-Path -LiteralPath $candidatePly -PathType Leaf)) {
        throw "3DGRUT $Stage stage completed without creating $candidatePly"
    }
    $checkpoints = @(Get-ChildItem -LiteralPath $StageRoot -Recurse -Filter "ckpt_last.pt" -File)
    if ($checkpoints.Count -ne 1) {
        throw "3DGRUT $Stage stage produced $($checkpoints.Count) ckpt_last.pt files; expected exactly one."
    }
    $checkpoint = $checkpoints[0].FullName
    $runDirectory = $checkpoints[0].Directory.FullName
    $metrics = Join-Path $runDirectory "metrics.json"
    $renders = Join-Path $runDirectory "ours_$Steps\renders"
    if (-not (Test-Path -LiteralPath $metrics -PathType Leaf) -or
        -not (Test-Path -LiteralPath $renders -PathType Container)) {
        throw "3DGRUT $Stage stage did not produce its required test metrics and renders."
    }
    $plyReport = Join-Path $StageRoot "$Stage-ply-report.json"
    Invoke-LoggedCommand $PythonPath @(
        $PlyVerifierPath, $candidatePly, "--json", $plyReport
    ) (Join-Path $LogsRoot "verify_3dgrut_$Stage.log") | Out-Host
    $stageReport = Join-Path $StageRoot "$Stage-stage-report.json"
    $gateArguments = @(
        $ThreeDGrutStageEvaluatorPath,
        "--stage", $Stage,
        "--trainer", $Trainer,
        "--expected-step", "$Steps",
        "--max-splats", "$MaxSplats",
        "--metrics", $metrics,
        "--renders", $renders,
        "--checkpoint", $checkpoint,
        "--ply-report", $plyReport,
        "--source-sha", $SourceSha256,
        "--dataset-image-sha", $DatasetImageSha256,
        "--dataset-model-sha", $DatasetModelSha256,
        "--config-sha", $ThreeDGrutConfigSha256,
        "--environment-sha", $ThreeDGrutEnvironmentSha256,
        "--json", $stageReport
    )
    if ($BaselineReport) { $gateArguments += @("--baseline", $BaselineReport) }
    Invoke-LoggedCommand $PythonPath $gateArguments (Join-Path $LogsRoot "evaluate_3dgrut_$Stage.log") | Out-Host
    $report = Get-Content -LiteralPath $stageReport -Raw | ConvertFrom-Json
    if ([string]$report.gate_status -ne "MECHANICAL_PASS__AWAITING_VISUAL_QC") {
        throw "3DGRUT $Stage stage did not authorize continuation: $($report.gate_status)"
    }
    return [pscustomobject]@{
        stage = $Stage
        steps = $Steps
        max_splats = $MaxSplats
        ply = $candidatePly
        checkpoint = $checkpoint
        report = $stageReport
        report_sha256 = (Get-FileHash -LiteralPath $stageReport -Algorithm SHA256).Hash.ToLowerInvariant()
        metrics = $metrics
        renders = $renders
    }
}

if (Test-Path -LiteralPath (Join-Path $RunRoot "publish_transaction.json") -PathType Leaf) {
    Restore-PublishTransaction
}
$AttemptId = [guid]::NewGuid().ToString()
if (Test-Path -LiteralPath $TrainingDecisionPath -PathType Leaf) {
    Remove-Item -LiteralPath $TrainingDecisionPath -Force
}
$attempt = [ordered]@{
    schema_version = 1
    attempt_id = $AttemptId
    status = "RUNNING"
    started_utc = [DateTime]::UtcNow.ToString("o")
    source_sha256 = $SourceSha256
    scene_type = $SceneType
    from_stage = $FromStage
    to_stage = $ToStage
    selected_frames = $SelectedFrames
    candidate_multiplier = $CandidateMultiplier
    max_cumulative_flow = $MaxCumulativeFlow
    training_steps = $TrainingSteps
    training_max_resolution = $TrainingMaxResolution
    brush_max_splats = $BrushMaxSplats
    brush_scale_loss_weight = $BrushScaleLossWeight
    eval_split_every = $EvalSplitEvery
    trainer = $Trainer
    spirula = [ordered]@{
        version = $SpirulaVersion
        executable_sha256 = $SpirulaSha256
        max_splats = if ($Trainer -eq "Spirula") { $BrushMaxSplats } else { $null }
        floater_suppression = if ($Trainer -eq "Spirula") { $SpirulaFloaterSuppression } else { $null }
        distraction_robustness = if ($Trainer -eq "Spirula") { $SpirulaDistractionRobustness } else { $null }
        save_full_checkpoint = if ($Trainer -eq "Spirula") { $SpirulaSaveFullCheckpoint } else { $null }
    }
    three_dgrut = [ordered]@{
        aerial_stages_required = $RequiresThreeDGrutAerialStages
        mcmc_max_splats = if ($Trainer -eq "3DGUT-MCMC") { $ThreeDGrutMcmcMaxSplats } else { $null }
        smoke_steps = if ($RequiresThreeDGrutAerialStages) { $ThreeDGrutSmokeSteps } else { $null }
        smoke_max_splats = if ($RequiresThreeDGrutAerialStages) { $ThreeDGrutSmokeMaxSplats } else { $null }
        diagnostic_steps = if ($RequiresThreeDGrutAerialStages) { $ThreeDGrutDiagnosticSteps } else { $null }
        diagnostic_max_splats = if ($RequiresThreeDGrutAerialStages) { $ThreeDGrutDiagnosticMaxSplats } else { $null }
        smoke_status = $null
        diagnostic_status = $null
        final_status = $null
    }
    aerial_diagnostic = [ordered]@{
        required = $RequiresAerialDiagnostic
        steps = if ($RequiresAerialDiagnostic) { $AerialDiagnosticSteps } else { $null }
        growth_stop_iter = if ($RequiresAerialDiagnostic) { $AerialDiagnosticGrowthStopIter } else { $null }
        max_splats = if ($RequiresAerialDiagnostic) { $AerialDiagnosticMaxSplats } else { $null }
        max_resolution = if ($RequiresAerialDiagnostic) { $AerialDiagnosticMaxResolution } else { $null }
        eval_split_every = if ($RequiresAerialDiagnostic) { $AerialDiagnosticEvalSplitEvery } else { $null }
        sentinel_selector_sha256 = $AerialSentinelSelectorSha256
        gate_evaluator_sha256 = $AerialGateEvaluatorSha256
        sentinel_manifest_sha256 = $null
        diagnostic_quality_report_sha256 = $null
        diagnostic_ply_report_sha256 = $null
        diagnostic_gate_report_sha256 = $null
        final_gate_report_sha256 = $null
    }
}
if (Test-Path -LiteralPath $RunAttemptPath -PathType Leaf) {
    try {
        $priorAttempt = Get-Content -LiteralPath $RunAttemptPath -Raw | ConvertFrom-Json
        $priorAttemptId = if ($priorAttempt.attempt_id) { [string]$priorAttempt.attempt_id } else { [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ") }
        Copy-Item -LiteralPath $RunAttemptPath -Destination (Join-Path $LogsRoot "run_attempt-$priorAttemptId.json") -Force
    } catch {
        Copy-Item -LiteralPath $RunAttemptPath -Destination (Join-Path $LogsRoot "run_attempt-unreadable-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')).json") -Force
    }
}
$attempt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
$CandidatePlyCleanupPath = $null

try {
if (Test-Stage "extract") {
    $probeLines = & $FfprobePath "-v" "error" "-select_streams" "v:0" "-show_streams" "-show_format" "-of" "json" $VideoPath 2>&1
    if ($LASTEXITCODE -ne 0) { throw "ffprobe failed: $($probeLines -join [Environment]::NewLine)" }
    $probeText = $probeLines -join [Environment]::NewLine
    $probeText | Set-Content -LiteralPath (Join-Path $RunRoot "video_probe.json") -Encoding utf8
    $probe = $probeText | ConvertFrom-Json
    $stream = @($probe.streams | Where-Object { $_.codec_type -eq "video" })[0]
    $duration = [double]::Parse([string]$probe.format.duration, [Globalization.CultureInfo]::InvariantCulture)
    $sourceFps = Convert-FractionToDouble ([string]$stream.avg_frame_rate)
    $rotation = 0
    if ($stream.PSObject.Properties.Name -contains "tags" -and $stream.tags -and $stream.tags.PSObject.Properties.Name -contains "rotate") {
        $rotation = [int]$stream.tags.rotate
    }
    if ($stream.PSObject.Properties.Name -contains "side_data_list") {
        foreach ($sideData in @($stream.side_data_list)) {
            if ($sideData.PSObject.Properties.Name -contains "rotation") { $rotation = [int]$sideData.rotation }
        }
    }

    $candidateFps = 0.0
    if ($AdaptiveExtraction -and $duration -gt 0) {
        $candidateFps = ($SelectedFrames * $CandidateMultiplier) / $duration
        if ($sourceFps -gt 0) { $candidateFps = [Math]::Min($candidateFps, $sourceFps) }
        $candidateFps = [Math]::Max(0.05, $candidateFps)
    }
    $extractMarker = Read-Marker "extract"
    $rawCount = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    $currentRawFrameHash = if ($rawCount -gt 0) { Get-ImageSetHash $RawFrames } else { $null }
    $canReuse = $extractMarker -and $rawCount -gt 0 -and
        (Test-ObjectProperty $extractMarker "source_bytes") -and
        (Test-ObjectProperty $extractMarker "source_last_write_utc") -and
        (Test-ObjectProperty $extractMarker "source_sha256") -and
        (Test-ObjectProperty $extractMarker "adaptive_extraction") -and
        (Test-ObjectProperty $extractMarker "candidate_multiplier") -and
        (Test-ObjectProperty $extractMarker "max_long_side") -and
        (Test-ObjectProperty $extractMarker "auto_rotate") -and
        (Test-ObjectProperty $extractMarker "selected_frame_target") -and
        (Test-ObjectProperty $extractMarker "raw_frames") -and
        (Test-ObjectProperty $extractMarker "raw_frame_hash") -and
        (Test-ObjectProperty $extractMarker "ffmpeg_sha256") -and
        (Test-ObjectProperty $extractMarker "ffprobe_sha256") -and
        [int64]$extractMarker.source_bytes -eq $SourceBytes -and
        [string]$extractMarker.source_last_write_utc -eq $SourceLastWriteUtc -and
        [string]$extractMarker.source_sha256 -eq $SourceSha256 -and
        [bool]$extractMarker.adaptive_extraction -eq [bool]$AdaptiveExtraction -and
        [int]$extractMarker.candidate_multiplier -eq $CandidateMultiplier -and
        [int]$extractMarker.max_long_side -eq $MaxLongSide -and
        [bool]$extractMarker.auto_rotate -eq (-not [bool]$NoAutoRotate) -and
        [int]$extractMarker.selected_frame_target -eq $SelectedFrames -and
        [int]$extractMarker.raw_frames -eq $rawCount -and
        [string]$extractMarker.raw_frame_hash -eq $currentRawFrameHash -and
        [string]$extractMarker.ffmpeg_sha256 -eq $FfmpegSha256 -and
        [string]$extractMarker.ffprobe_sha256 -eq $FfprobeSha256

    if ($canReuse) {
        Write-Host "[extract] Reusing $rawCount decoded frames"
    } else {
        New-Item -ItemType Directory -Force -Path $RawFrames | Out-Null
        Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg" -ErrorAction SilentlyContinue | Remove-Item -Force
        $filters = [System.Collections.Generic.List[string]]::new()
        if ($candidateFps -gt 0) {
            $filters.Add("fps=" + $candidateFps.ToString("0.######", [Globalization.CultureInfo]::InvariantCulture))
        }
        $sourceLongSide = [Math]::Max([int]$stream.width, [int]$stream.height)
        if ($MaxLongSide -gt 0 -and $sourceLongSide -gt $MaxLongSide) {
            $filters.Add("scale=${MaxLongSide}:${MaxLongSide}:force_original_aspect_ratio=decrease:force_divisible_by=2")
        }
        $arguments = [System.Collections.Generic.List[string]]::new()
        $arguments.AddRange([string[]]@("-hide_banner", "-loglevel", "error"))
        if ($NoAutoRotate) { $arguments.Add("-noautorotate") }
        $arguments.AddRange([string[]]@("-i", $VideoPath, "-map", "0:v:0"))
        if ($filters.Count -gt 0) { $arguments.AddRange([string[]]@("-vf", ($filters -join ","))) }
        $arguments.AddRange([string[]]@("-fps_mode", "passthrough", "-q:v", "1", "-an", (Join-Path $RawFrames "frame_%06d.jpg")))
        Invoke-LoggedCommand $FfmpegPath $arguments.ToArray() (Join-Path $LogsRoot "extract.log")
        $rawCount = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg").Count
        if ($rawCount -lt 2) { throw "FFmpeg extracted only $rawCount frames" }
        $currentRawFrameHash = Get-ImageSetHash $RawFrames
        Write-Marker "extract" ([ordered]@{
            source_bytes = $SourceBytes
            source_last_write_utc = $SourceLastWriteUtc
            source_sha256 = $SourceSha256
            raw_frames = $rawCount
            raw_frame_hash = $currentRawFrameHash
            selected_frame_target = $SelectedFrames
            adaptive_extraction = [bool]$AdaptiveExtraction
            candidate_multiplier = $CandidateMultiplier
            candidate_fps = [Math]::Round($candidateFps, 6)
            max_long_side = $MaxLongSide
            auto_rotate = -not [bool]$NoAutoRotate
            detected_rotation_degrees = $rotation
            source_width = [int]$stream.width
            source_height = [int]$stream.height
            source_fps = [Math]::Round($sourceFps, 4)
            duration_seconds = [Math]::Round($duration, 3)
            ffmpeg_version = $FfmpegVersion
            ffmpeg_sha256 = $FfmpegSha256
            ffprobe_version = $FfprobeVersion
            ffprobe_sha256 = $FfprobeSha256
        })
        Write-Host "[extract] Decoded $rawCount frames (rotation $rotation degrees, autorotate $(-not [bool]$NoAutoRotate))" -ForegroundColor Green
    }
}

if (Test-Stage "select") {
    $extractPrerequisite = Get-ExtractChainState
    if (-not $extractPrerequisite.valid) {
        throw "The extract stage does not match the current video, settings, or tools. Restart from extract before selecting frames."
    }
    if (-not (Test-Path -LiteralPath $RawFrames)) { throw "Raw frames are missing; start from extract" }
    $rawCount = [int]$extractPrerequisite.count
    $rawFrameHash = [string]$extractPrerequisite.hash
    $minimumCount = [Math]::Min($SelectedFrames, $rawCount)
    $selectState = Get-SelectChainState $extractPrerequisite
    $selectedCount = [int]$selectState.count
    if ($selectState.valid) {
        Write-Host "[select] Reusing $selectedCount selected frames"
    } else {
        New-Item -ItemType Directory -Force -Path $ImagesPath | Out-Null
        $maxCumulativeFlowText = $MaxCumulativeFlow.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
        Invoke-LoggedCommand $PythonPath @(
            $SelectorScriptPath,
            "--input", $RawFrames,
            "--output", $ImagesPath,
            "--target", "$SelectedFrames",
            "--profile", $SceneType,
            "--blur-percentile", "20",
            "--require-motion",
            "--records", $FrameQualityPath,
            "--contact-sheet", (Join-Path $RunRoot "selected_frames_contact_sheet.jpg"),
            "--max-cumulative-flow", $maxCumulativeFlowText
        ) (Join-Path $LogsRoot "select.log")
        $selectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg").Count
        $selectedCountValid = if ($MaxCumulativeFlow -gt 0.0) {
            $selectedCount -ge $minimumCount -and $selectedCount -le $rawCount
        } else {
            $selectedCount -eq $minimumCount
        }
        if (-not $selectedCountValid) {
            $countExpectation = if ($MaxCumulativeFlow -gt 0.0) {
                "at least $minimumCount and no more than $rawCount"
            } else { "exactly $minimumCount" }
            throw "Frame selector produced $selectedCount frames; expected $countExpectation"
        }
        if (-not (Test-Path -LiteralPath $FrameQualityPath -PathType Leaf)) {
            throw "Frame selector did not produce its quality report: $FrameQualityPath"
        }
        $frameQuality = Get-Content -LiteralPath $FrameQualityPath -Raw | ConvertFrom-Json
        if ([int]$frameQuality.source_frames -ne $rawCount -or
            [int]$frameQuality.requested_frames -ne $SelectedFrames -or
            [int]$frameQuality.selected_frames -ne $selectedCount -or
            [double]$frameQuality.max_cumulative_flow -ne $MaxCumulativeFlow) {
            throw "Frame selector report does not match the current input, settings, or selected image count"
        }
        $frameQualitySha = (Get-FileHash -LiteralPath $FrameQualityPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $selectedFrameHash = Get-ImageSetHash $ImagesPath
        Write-Marker "select" ([ordered]@{
            source_frames = $rawCount
            raw_frame_hash = $rawFrameHash
            requested_frames = $SelectedFrames
            selected_frames = $selectedCount
            max_cumulative_flow = $MaxCumulativeFlow
            frame_quality_sha256 = $frameQualitySha
            scene_type = $SceneType
            selector_version = $SelectorVersion
            selector_script_sha256 = $SelectorScriptSha256
            selected_frame_hash = $selectedFrameHash
        })
        Write-Host "[select] Kept $selectedCount sharp, exposed, motion-qualified frames for the $SceneType profile" -ForegroundColor Green
    }
}

if (Test-Stage "solve") {
    $selectPrerequisite = Get-SelectChainState $null
    if (-not $selectPrerequisite.valid) {
        throw "The selected frames do not have a current source-to-selection provenance chain. Restart from extract before solving cameras."
    }
    $selectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    if ($selectedCount -lt 2) { throw "Selected images are missing; start from select" }
    $imageSetHash = Get-ImageSetHash $ImagesPath
    Assert-ShotManifestContract $selectPrerequisite | Out-Null
    $SolveDatabasePath = if ($UsesIncrementalMapper) { $DatabasePath } else { $CalibratedDatabasePath }
    $modelFiles = @("cameras.bin", "images.bin", "points3D.bin") | ForEach-Object { Join-Path $SparsePath $_ }
    $hasModel = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0
    $currentModelHash = if ($hasModel) { Get-ColmapModelHash $SparsePath } else { $null }
    $currentDatabaseSha = if (Test-Path -LiteralPath $SolveDatabasePath -PathType Leaf) {
        (Get-FileHash -LiteralPath $SolveDatabasePath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $currentReconstructionReportPath = Join-Path $RunRoot "reconstruction_report.json"
    $currentReconstructionReportSha = if (Test-Path -LiteralPath $currentReconstructionReportPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $currentReconstructionReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $solveMarkerPath = Join-Path $RunRoot ".solve.complete.json"
    $previousSolveMarkerSha = if (Test-Path -LiteralPath $solveMarkerPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $solveMarkerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $solveMarker = Read-Marker "solve"
    $currentImageOrderSha = if ($UsesChronologicalMatching -and (Test-Path -LiteralPath $ImageOrderPath -PathType Leaf)) {
        (Get-FileHash -LiteralPath $ImageOrderPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $imageOrderArtifactsMatch = if ($UsesChronologicalMatching) {
        $solveMarker -and
        (Test-ObjectProperty $solveMarker "image_order_sha256") -and
        (Test-ObjectProperty $solveMarker "image_order_validator_sha256") -and
        [string]$solveMarker.image_order_sha256 -eq [string]$currentImageOrderSha -and
        [string]$solveMarker.image_order_validator_sha256 -eq $ImageOrderValidatorSha256
    } else { $true }
    $existingSolveArtifactsMatch = $hasModel -and (Test-Path -LiteralPath $SolveDatabasePath -PathType Leaf) -and $solveMarker -and
        $imageOrderArtifactsMatch -and
        (Test-ObjectProperty $solveMarker "image_set_hash") -and
        (Test-ObjectProperty $solveMarker "scene_type") -and
        (Test-ObjectProperty $solveMarker "matcher") -and
        (Test-ObjectProperty $solveMarker "mapper") -and
        (Test-ObjectProperty $solveMarker "selector_version") -and
        (Test-ObjectProperty $solveMarker "gate_version") -and
        (Test-ObjectProperty $solveMarker "colmap_version") -and
        (Test-ObjectProperty $solveMarker "colmap_sha256") -and
        (Test-ObjectProperty $solveMarker "model_hash") -and
        (Test-ObjectProperty $solveMarker "pose_validator_sha256") -and
        (Test-ObjectProperty $solveMarker "runner_script_sha256") -and
        (Test-ObjectProperty $solveMarker "database_sha256") -and
        (Test-ObjectProperty $solveMarker "reconstruction_report_sha256") -and
        (Test-ObjectProperty $solveMarker "vocab_tree_sha256") -and
        [string]$solveMarker.image_set_hash -eq $imageSetHash -and
        [string]$solveMarker.scene_type -eq $SceneType -and
        [string]$solveMarker.matcher -eq $MatcherName -and
        [string]$solveMarker.mapper -eq $MapperName -and
        [string]$solveMarker.selector_version -eq $SelectorVersion -and
        [string]$solveMarker.colmap_version -eq $ColmapVersion -and
        [string]$solveMarker.colmap_sha256 -eq $ColmapSha256 -and
        [string]$solveMarker.model_hash -eq $currentModelHash -and
        [string]$solveMarker.database_sha256 -eq $currentDatabaseSha -and
        [string]$solveMarker.reconstruction_report_sha256 -eq $currentReconstructionReportSha -and
        [string]$solveMarker.vocab_tree_sha256 -eq [string]$VocabTreeSha256
    $currentSolveValidationMatches = $existingSolveArtifactsMatch -and
        [string]$solveMarker.gate_version -eq $GateVersion -and
        [string]$solveMarker.pose_validator_sha256 -eq $PoseValidatorSha256 -and
        [string]$solveMarker.runner_script_sha256 -eq $RunnerScriptSha256 -and
        (Test-ObjectProperty $solveMarker "shot_manifest_path") -and
        [string]$solveMarker.shot_manifest_path -eq [string]$ResolvedShotManifestPath -and
        (Test-ObjectProperty $solveMarker "shot_manifest_sha256") -and
        [string]$solveMarker.shot_manifest_sha256 -eq [string]$ShotManifestSha256
    $canRevalidateModel = [bool]($RevalidateSolve -and $existingSolveArtifactsMatch)
    if ($RevalidateSolve -and -not $canRevalidateModel) {
        throw "RevalidateSolve refused: the selected images, COLMAP model, database, or prior solve report no longer match the recorded solve hashes. Run a normal solve to rebuild them."
    }
    $reusedExistingModel = [bool]($currentSolveValidationMatches -or $canRevalidateModel)
    if ($reusedExistingModel) {
        $reuseReason = if ($canRevalidateModel) { "for validation-only policy refresh" } else { "with current validation policy" }
        Write-Host "[solve] Reusing the exact signed $SceneType COLMAP model $reuseReason; COLMAP matching and mapping will not run"
    } else {
        if (Test-Path -LiteralPath $DatabasePath) { Remove-Item -LiteralPath $DatabasePath -Force }
        if (Test-Path -LiteralPath $CalibratedDatabasePath) { Remove-Item -LiteralPath $CalibratedDatabasePath -Force }
        if (Test-Path -LiteralPath $SparseRoot) { Remove-Item -LiteralPath $SparseRoot -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $SparseRoot | Out-Null
        $orderedImageNames = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" | Sort-Object Name | ForEach-Object { $_.Name })
        if ($orderedImageNames.Count -ne $selectedCount) {
            throw "Chronological image list has $($orderedImageNames.Count) names; expected $selectedCount"
        }
        [System.IO.File]::WriteAllLines($ImageOrderPath, $orderedImageNames, [System.Text.UTF8Encoding]::new($false))
        $currentImageOrderSha = (Get-FileHash -LiteralPath $ImageOrderPath -Algorithm SHA256).Hash.ToLowerInvariant()
        Invoke-LoggedCommand $ColmapPath @(
            "feature_extractor", "--database_path", $DatabasePath, "--image_path", $ImagesPath,
            "--image_list_path", $ImageOrderPath,
            "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
            "--FeatureExtraction.type", "SIFT", "--FeatureExtraction.num_threads", "1",
            "--FeatureExtraction.gpu_index", "0",
            "--SiftExtraction.max_num_features", "8192",
            "--log_path", (Join-Path $LogsRoot "colmap_feature")
        ) (Join-Path $LogsRoot "colmap_feature_console.log")
        if ($UsesChronologicalMatching) {
            Invoke-LoggedCommand $PythonPath @(
                $ImageOrderValidatorPath, "--database", $DatabasePath, "--image-list", $ImageOrderPath
            ) (Join-Path $LogsRoot "validate_colmap_image_order.log")
            Invoke-LoggedCommand $ColmapPath @(
                "sequential_matcher", "--database_path", $DatabasePath,
                "--FeatureMatching.type", "SIFT_BRUTEFORCE",
                "--SequentialMatching.overlap", "15",
                "--SequentialMatching.quadratic_overlap", "1",
                "--SequentialMatching.loop_detection", "1",
                "--SequentialMatching.loop_detection_period", "10",
                "--SequentialMatching.loop_detection_num_images", "50",
                "--SequentialMatching.loop_detection_num_nearest_neighbors", "5",
                "--SequentialMatching.vocab_tree_path", $VocabTreePath,
                "--log_path", (Join-Path $LogsRoot "colmap_match")
            ) (Join-Path $LogsRoot "colmap_match_console.log")
        } else {
            Invoke-LoggedCommand $ColmapPath @(
                "exhaustive_matcher", "--database_path", $DatabasePath,
                "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--log_path", (Join-Path $LogsRoot "colmap_match")
            ) (Join-Path $LogsRoot "colmap_match_console.log")
        }
        if ($UsesIncrementalMapper) {
            Invoke-LoggedCommand $ColmapPath @(
                "mapper", "--database_path", $DatabasePath, "--image_path", $ImagesPath,
                "--output_path", $SparseRoot, "--Mapper.multiple_models", "0", "--Mapper.ba_use_gpu", "1",
                "--log_path", (Join-Path $LogsRoot "colmap_incremental")
            ) (Join-Path $LogsRoot "colmap_incremental_console.log")
        } else {
            Copy-Item -LiteralPath $DatabasePath -Destination $CalibratedDatabasePath -Force
            Invoke-LoggedCommand $ColmapPath @(
                "view_graph_calibrator", "--database_path", $CalibratedDatabasePath,
                "--log_path", (Join-Path $LogsRoot "colmap_calibrate")
            ) (Join-Path $LogsRoot "colmap_calibrate_console.log")
            Invoke-LoggedCommand $ColmapPath @(
                "global_mapper", "--database_path", $CalibratedDatabasePath, "--image_path", $ImagesPath,
                "--output_path", $SparseRoot, "--GlobalMapper.ba_ceres_use_gpu", "1",
                "--log_path", (Join-Path $LogsRoot "colmap_global")
            ) (Join-Path $LogsRoot "colmap_global_console.log")
        }
        $hasModel = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0
        if (-not $hasModel) { throw "COLMAP $MapperName did not produce a complete sparse model" }
    }

    function Measure-And-ValidateModel([string]$ModelPath) {
        if (Test-Path -LiteralPath $ModelTextRoot) { Remove-Item -LiteralPath $ModelTextRoot -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $ModelTextRoot | Out-Null
        Invoke-LoggedCommand $ColmapPath @(
            "model_converter", "--input_path", $ModelPath, "--output_path", $ModelTextRoot, "--output_type", "TXT"
        ) (Join-Path $LogsRoot "colmap_model_converter.log")

        $analysisLines = & $ColmapPath "model_analyzer" "--path" $ModelPath "--log_target" "stdout" 2>&1
        if ($LASTEXITCODE -ne 0) { throw "COLMAP model_analyzer failed" }
        $analysisText = $analysisLines -join [Environment]::NewLine
        $analysisText | Set-Content -LiteralPath (Join-Path $RunRoot "reconstruction_analysis.txt") -Encoding utf8
        function Read-AnalysisMetric([string]$Label) {
            $match = [regex]::Match($analysisText, "(?i)" + [regex]::Escape($Label) + "\s*:\s*([0-9.eE+-]+)")
            if ($match.Success) { return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) }
            return $null
        }
        $measuredRegistered = Read-AnalysisMetric "Registered images"
        if ($null -eq $measuredRegistered) { $measuredRegistered = Read-AnalysisMetric "Images" }
        $measuredPoints = Read-AnalysisMetric "Points"
        $measuredReprojection = Read-AnalysisMetric "Mean reprojection error"
        $measuredTrackLength = Read-AnalysisMetric "Mean track length"
        if ($null -eq $measuredRegistered -or $null -eq $measuredPoints -or $null -eq $measuredTrackLength -or $null -eq $measuredReprojection) {
            throw "Could not parse all required reconstruction metrics from COLMAP model_analyzer"
        }
        $poseReportPath = Join-Path $RunRoot "reconstruction_report.json"
        Assert-ShotManifestContract $selectPrerequisite | Out-Null
        $poseArguments = @(
            $PoseValidatorPath,
            "--model-text", $ModelTextRoot,
            "--database", $SolveDatabasePath,
            "--selected-images", $ImagesPath,
            "--profile", $SceneType,
            "--registered", "$([int]$measuredRegistered)",
            "--points", "$([int]$measuredPoints)",
            "--mean-track-length", "$measuredTrackLength",
            "--reprojection-error", "$measuredReprojection",
            "--json", $poseReportPath
        )
        if ($ResolvedShotManifestPath) {
            $poseArguments += @("--shot-manifest", $ResolvedShotManifestPath)
        }
        Invoke-LoggedCommand $PythonPath $poseArguments (Join-Path $LogsRoot "validate_poses.log")
        Assert-ShotManifestContract $selectPrerequisite | Out-Null
        return [pscustomobject]@{
            registered = $measuredRegistered
            points = $measuredPoints
            reprojection = $measuredReprojection
            track_length = $measuredTrackLength
            report = (Get-Content -LiteralPath $poseReportPath -Raw | ConvertFrom-Json)
        }
    }

    $modelValidation = @(Measure-And-ValidateModel $SparsePath)[-1]
    $excludedPoseNames = @($modelValidation.report.quality_gates.trajectory_continuity.excluded_isolated_poses)
    $maximumExcludedPoses = [int]$modelValidation.report.quality_gates.trajectory_continuity.maximum_excluded_isolated_poses
    $removedPoseNames = @()
    if ($RevalidateSolve -and $excludedPoseNames.Count -gt 0) {
        throw "RevalidateSolve found isolated camera poses that would require changing the COLMAP model. Run a normal solve instead."
    } elseif ($excludedPoseNames.Count -gt $maximumExcludedPoses) {
        Write-Warning "Reconstruction has $($excludedPoseNames.Count) isolated poses; the $SceneType limit is $maximumExcludedPoses. Automatic cleanup was refused and training will remain locked."
    } elseif ($excludedPoseNames.Count -gt 0) {
        Write-Warning "Removing $($excludedPoseNames.Count) isolated camera pose(s) before Gaussian training"
        $exclusionListPath = Join-Path $ReconRoot "isolated_camera_poses.txt"
        $excludedPoseNames | Set-Content -LiteralPath $exclusionListPath -Encoding ascii
        $cleaningRoot = Join-Path $ReconRoot "sparse_cleaning"
        $cleaningPath = Join-Path $cleaningRoot "0"
        if (Test-Path -LiteralPath $cleaningRoot) { Remove-Item -LiteralPath $cleaningRoot -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $cleaningPath | Out-Null
        Invoke-LoggedCommand $ColmapPath @(
            "image_deleter", "--input_path", $SparsePath, "--output_path", $cleaningPath,
            "--image_names_path", $exclusionListPath
        ) (Join-Path $LogsRoot "colmap_remove_isolated_poses.log")
        foreach ($required in @("cameras.bin", "images.bin", "points3D.bin")) {
            if (-not (Test-Path -LiteralPath (Join-Path $cleaningPath $required) -PathType Leaf)) {
                throw "COLMAP image_deleter did not produce a complete cleaned model"
            }
        }
        $archivePath = Join-Path $DiagnosticsRoot "sparse_with_isolated_poses\0"
        if (-not (Test-Path -LiteralPath $archivePath -PathType Container)) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $archivePath) | Out-Null
            Copy-Item -LiteralPath $SparsePath -Destination $archivePath -Recurse
        }
        Remove-Item -LiteralPath $SparsePath -Recurse -Force
        Move-Item -LiteralPath $cleaningPath -Destination $SparsePath
        Remove-Item -LiteralPath $cleaningRoot -Recurse -Force
        $modelValidation = @(Measure-And-ValidateModel $SparsePath)[-1]
        if (@($modelValidation.report.quality_gates.trajectory_continuity.excluded_isolated_poses).Count -gt 0) {
            throw "Isolated camera poses remained after model cleanup"
        }
        $removedPoseNames = $excludedPoseNames
    }

    $registered = $modelValidation.registered
    $points = $modelValidation.points
    $reprojectionError = $modelValidation.reprojection
    $meanTrackLength = $modelValidation.track_length
    $reconReport = $modelValidation.report
    $poseReportPath = Join-Path $RunRoot "reconstruction_report.json"
    $validatedUtc = [DateTime]::UtcNow.ToString("o")
    $currentModelHash = Get-ColmapModelHash $SparsePath
    $currentDatabaseSha = (Get-FileHash -LiteralPath $SolveDatabasePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $currentCamerasTextSha = (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "cameras.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
    $currentImagesTextSha = (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "images.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
    $currentPointsTextSha = (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "points3D.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
    $modelCreatedRunnerScriptSha = if ($reusedExistingModel -and (Test-ObjectProperty $solveMarker "model_created_runner_script_sha256")) {
        [string]$solveMarker.model_created_runner_script_sha256
    } elseif ($reusedExistingModel) {
        [string]$solveMarker.runner_script_sha256
    } else {
        $RunnerScriptSha256
    }
    $modelCreatedUtc = if ($reusedExistingModel -and (Test-ObjectProperty $solveMarker "model_created_utc")) {
        [string]$solveMarker.model_created_utc
    } elseif ($reusedExistingModel -and (Test-ObjectProperty $solveMarker "completed_utc")) {
        [string]$solveMarker.completed_utc
    } else {
        $validatedUtc
    }
    $validationMode = if ($reusedExistingModel) { "revalidated_existing_model" } else { "reconstructed_and_validated" }
    $reconReport | Add-Member -NotePropertyName "image_set_hash" -NotePropertyValue $imageSetHash -Force
    $reconReport | Add-Member -NotePropertyName "shot_manifest_path" -NotePropertyValue $ResolvedShotManifestPath -Force
    $reconReport | Add-Member -NotePropertyName "shot_manifest_sha256" -NotePropertyValue $ShotManifestSha256 -Force
    $reconReport | Add-Member -NotePropertyName "matcher" -NotePropertyValue $MatcherName -Force
    $reconReport | Add-Member -NotePropertyName "mapper" -NotePropertyValue $MapperName -Force
    $reconReport | Add-Member -NotePropertyName "colmap_version" -NotePropertyValue $ColmapVersion -Force
    $reconReport | Add-Member -NotePropertyName "model_path" -NotePropertyValue $SparsePath -Force
    $reconReport | Add-Member -NotePropertyName "model_text_path" -NotePropertyValue $ModelTextRoot -Force
    $reconReport | Add-Member -NotePropertyName "database_path" -NotePropertyValue $SolveDatabasePath -Force
    $reconReport | Add-Member -NotePropertyName "removed_isolated_camera_poses" -NotePropertyValue $removedPoseNames -Force
    $reconReport | Add-Member -NotePropertyName "validation_mode" -NotePropertyValue $validationMode -Force
    $reconReport | Add-Member -NotePropertyName "validated_utc" -NotePropertyValue $validatedUtc -Force
    $reconReport | Add-Member -NotePropertyName "model_hash" -NotePropertyValue $currentModelHash -Force
    $reconReport | Add-Member -NotePropertyName "cameras_text_sha256" -NotePropertyValue $currentCamerasTextSha -Force
    $reconReport | Add-Member -NotePropertyName "images_text_sha256" -NotePropertyValue $currentImagesTextSha -Force
    $reconReport | Add-Member -NotePropertyName "points3d_text_sha256" -NotePropertyValue $currentPointsTextSha -Force
    $reconReport | Add-Member -NotePropertyName "database_sha256" -NotePropertyValue $currentDatabaseSha -Force
    $reconReport | Add-Member -NotePropertyName "model_created_utc" -NotePropertyValue $modelCreatedUtc -Force
    $reconReport | Add-Member -NotePropertyName "model_created_runner_script_sha256" -NotePropertyValue $modelCreatedRunnerScriptSha -Force
    $reconReport | Add-Member -NotePropertyName "validation_runner_script_sha256" -NotePropertyValue $RunnerScriptSha256 -Force
    $reconReport | Add-Member -NotePropertyName "pose_validator_sha256" -NotePropertyValue $PoseValidatorSha256 -Force
    $reconReport | Add-Member -NotePropertyName "previous_solve_marker_sha256" -NotePropertyValue $(if ($reusedExistingModel) { $previousSolveMarkerSha } else { $null }) -Force
    $reconReport | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath "$poseReportPath.tmp" -Encoding utf8
    Move-Item -LiteralPath "$poseReportPath.tmp" -Destination $poseReportPath -Force
    $qualityPass = [bool]$reconReport.quality_gates.overall_pass
    $registrationRate = [double]$reconReport.registration_percent
    $currentReconstructionReportSha = (Get-FileHash -LiteralPath $poseReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Marker "solve" ([ordered]@{
        image_set_hash = $imageSetHash
        shot_manifest_path = $ResolvedShotManifestPath
        shot_manifest_sha256 = $ShotManifestSha256
        registered_images = $registered
        selected_images = $selectedCount
        registration_percent = $registrationRate
        points = $points
        mean_track_length = $meanTrackLength
        reprojection_error_pixels = $reprojectionError
        quality_gates_pass = $qualityPass
        scene_type = $SceneType
        matcher = $MatcherName
        mapper = $MapperName
        selector_version = $SelectorVersion
        gate_version = $GateVersion
        colmap_version = $ColmapVersion
        colmap_sha256 = $ColmapSha256
        model_hash = $currentModelHash
        cameras_text_sha256 = $currentCamerasTextSha
        images_text_sha256 = $currentImagesTextSha
        points3d_text_sha256 = $currentPointsTextSha
        validation_mode = $validationMode
        model_created_utc = $modelCreatedUtc
        model_created_runner_script_sha256 = $modelCreatedRunnerScriptSha
        validation_runner_script_sha256 = $RunnerScriptSha256
        previous_solve_marker_sha256 = if ($reusedExistingModel) { $previousSolveMarkerSha } else { $null }
        pose_validator_sha256 = $PoseValidatorSha256
        runner_script_sha256 = $RunnerScriptSha256
        image_order_sha256 = if ($UsesChronologicalMatching) { $currentImageOrderSha } else { $null }
        image_order_validator_sha256 = $ImageOrderValidatorSha256
        database_sha256 = $currentDatabaseSha
        reconstruction_report_sha256 = $currentReconstructionReportSha
        vocab_tree_sha256 = $VocabTreeSha256
    })
    $qualityText = "registered $registered/$selectedCount ($([Math]::Round($registrationRate, 1))%), points $points, track $meanTrackLength, reprojection $reprojectionError px"
    if ($qualityPass) {
        Write-Host "[solve] Quality gates passed: $qualityText" -ForegroundColor Green
    } else {
        Write-Warning "Reconstruction gates failed: $qualityText. Training will remain locked."
    }
}

if (Test-Stage "train") {
    $solvePrerequisite = Get-SolveChainState $null
    if (-not $solvePrerequisite.valid) {
        throw "The reconstruction is not current or did not pass the signed camera-quality gates. Restart from solve before training."
    }
    foreach ($required in @("cameras.bin", "images.bin", "points3D.bin")) {
        if (-not (Test-Path -LiteralPath (Join-Path $SparsePath $required))) { throw "COLMAP model is incomplete; start from solve" }
    }
    $solveMarker = Read-Marker "solve"
    if (-not $solveMarker -or -not (Test-ObjectProperty $solveMarker "image_set_hash") -or
        -not (Test-ObjectProperty $solveMarker "quality_gates_pass")) {
        throw "Reconstruction quality state is missing; rerun from solve before training"
    }
    if (-not [bool]$solveMarker.quality_gates_pass) {
        throw "Reconstruction quality gates failed; training was stopped to avoid producing a bad splat. Review reconstruction_report.json and recapture or adjust frame selection."
    }
    $solvedSceneType = if (Test-ObjectProperty $solveMarker "scene_type") { [string]$solveMarker.scene_type } else { "missing profile" }
    if ($solvedSceneType -ne $SceneType) {
        throw "The completed reconstruction belongs to '$solvedSceneType', not '$SceneType'. Rerun from solve with the requested capture profile."
    }
    $imageSetHash = Get-ImageSetHash $ImagesPath
    if ($imageSetHash -ne [string]$solveMarker.image_set_hash) {
        throw "Selected image content changed after the solve. Rerun from solve before training."
    }
    $legacySparseArchive = Join-Path $ReconRoot "sparse_with_isolated_poses"
    if (Test-Path -LiteralPath $legacySparseArchive -PathType Container) {
        New-Item -ItemType Directory -Force -Path $DiagnosticsRoot | Out-Null
        $legacyArchiveDestination = Join-Path $DiagnosticsRoot "sparse_with_isolated_poses"
        if (Test-Path -LiteralPath $legacyArchiveDestination) {
            $legacyArchiveDestination = Join-Path $DiagnosticsRoot "sparse_with_isolated_poses-legacy-$AttemptId"
        }
        Move-Item -LiteralPath $legacySparseArchive -Destination $legacyArchiveDestination
        Write-Warning "Moved archived COLMAP diagnostics outside Brush's dataset root"
    }
    $currentModelHash = Get-ColmapModelHash $SparsePath
    if (-not (Test-ObjectProperty $solveMarker "model_hash") -or $currentModelHash -ne [string]$solveMarker.model_hash) {
        throw "The COLMAP model changed after validation. Rerun from solve before training."
    }
    $trainMarker = Read-Marker "train"
    $existingAerialDiagnosticEvidence = Get-AerialDiagnosticEvidenceState $trainMarker
    $existingThreeDGrutAerialEvidence = Get-ThreeDGrutAerialEvidenceState $trainMarker
    $aerialSentinelManifestForMarker = $null
    $aerialSentinelManifestShaForMarker = $null
    $aerialDiagnosticQualityForMarker = $null
    $aerialDiagnosticQualityShaForMarker = $null
    $aerialDiagnosticPlyReportForMarker = $null
    $aerialDiagnosticPlyReportShaForMarker = $null
    $aerialDiagnosticGateForMarker = $null
    $aerialDiagnosticGateShaForMarker = $null
    $aerialDiagnosticHoldoutShaForMarker = $null
    $aerialFinalGateForMarker = $null
    $aerialFinalGateShaForMarker = $null
    $aerialExpectedPublishPlySha = $null
    $aerialExpectedPublishQualitySha = $null
    $threeDGrutSmokeDatasetForMarker = $null
    $threeDGrutSmokeDatasetShaForMarker = $null
    $threeDGrutSmokeStageForMarker = $null
    $threeDGrutSmokeStageShaForMarker = $null
    $threeDGrutDiagnosticStageForMarker = $null
    $threeDGrutDiagnosticStageShaForMarker = $null
    $threeDGrutFinalStageForMarker = $null
    $threeDGrutFinalStageShaForMarker = $null
    $spirulaMetricsPathForMarker = $null
    $spirulaConfigPathForMarker = $null
    if ($Trainer -eq "Spirula" -and $trainMarker) {
        if (Test-ObjectProperty $trainMarker "spirula_metrics_path") {
            $spirulaMetricsPathForMarker = [string]$trainMarker.spirula_metrics_path
        }
        if (Test-ObjectProperty $trainMarker "spirula_config_path") {
            $spirulaConfigPathForMarker = [string]$trainMarker.spirula_config_path
        }
    }
    if ($RequiresAerialDiagnostic -and $existingAerialDiagnosticEvidence.valid) {
        $aerialSentinelManifestForMarker = [string]$existingAerialDiagnosticEvidence.sentinel.path
        $aerialSentinelManifestShaForMarker = [string]$existingAerialDiagnosticEvidence.sentinel.sha256
        $aerialDiagnosticQualityForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_quality.path
        $aerialDiagnosticQualityShaForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_quality.sha256
        $aerialDiagnosticPlyReportForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_ply.path
        $aerialDiagnosticPlyReportShaForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_ply.sha256
        $aerialDiagnosticGateForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_gate.path
        $aerialDiagnosticGateShaForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_gate.sha256
        $aerialDiagnosticHoldoutShaForMarker = [string]$existingAerialDiagnosticEvidence.diagnostic_holdout_sha256
        $aerialFinalGateForMarker = [string]$existingAerialDiagnosticEvidence.final_gate.path
        $aerialFinalGateShaForMarker = [string]$existingAerialDiagnosticEvidence.final_gate.sha256
    }
    if ($RequiresThreeDGrutAerialStages -and $existingThreeDGrutAerialEvidence.valid) {
        $threeDGrutSmokeDatasetForMarker = [string]$existingThreeDGrutAerialEvidence.smoke_dataset.path
        $threeDGrutSmokeDatasetShaForMarker = [string]$existingThreeDGrutAerialEvidence.smoke_dataset.sha256
        $threeDGrutSmokeStageForMarker = [string]$existingThreeDGrutAerialEvidence.smoke.path
        $threeDGrutSmokeStageShaForMarker = [string]$existingThreeDGrutAerialEvidence.smoke.sha256
        $threeDGrutDiagnosticStageForMarker = [string]$existingThreeDGrutAerialEvidence.diagnostic.path
        $threeDGrutDiagnosticStageShaForMarker = [string]$existingThreeDGrutAerialEvidence.diagnostic.sha256
        $threeDGrutFinalStageForMarker = [string]$existingThreeDGrutAerialEvidence.final.path
        $threeDGrutFinalStageShaForMarker = [string]$existingThreeDGrutAerialEvidence.final.sha256
    }
    $existingPlySha = if (Test-Path -LiteralPath $FinalPly -PathType Leaf) {
        (Get-FileHash -LiteralPath $FinalPly -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $existingQualitySha = if (Test-Path -LiteralPath $QualityReportPath -PathType Leaf) {
        (Get-FileHash -LiteralPath $QualityReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $existingQualityReport = if ($existingQualitySha) {
        try { Get-Content -LiteralPath $QualityReportPath -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    $existingRenderCountValid = $false
    $existingHoldoutRenderSetSha = $null
    if ($existingQualityReport -and (Test-ObjectProperty $existingQualityReport "render_directory") -and
        (Test-ObjectProperty $existingQualityReport "saved_holdout_renders")) {
        $existingRenderDirectory = [string]$existingQualityReport.render_directory
        if (Test-Path -LiteralPath $existingRenderDirectory -PathType Container) {
            $existingRenderCount = @(Get-ChildItem -LiteralPath $existingRenderDirectory -Filter "*.png").Count
            $existingRenderCountValid = $existingRenderCount -eq [int]$existingQualityReport.saved_holdout_renders
            if ($existingRenderCountValid) {
                $existingHoldoutRenderSetSha = Get-FileSetHash $existingRenderDirectory "*.png"
            }
        }
    }
    $qualityReportValid = $null -ne $existingQualityReport -and
        $existingRenderCountValid -and
        (Test-ObjectProperty $existingQualityReport "quality_status") -and
        [string]$existingQualityReport.quality_status -eq "measured_unrated" -and
        (Test-ObjectProperty $existingQualityReport "expected_holdout_renders") -and
        (Test-ObjectProperty $existingQualityReport "saved_holdout_renders") -and
        [int]$existingQualityReport.expected_holdout_renders -eq [int]$existingQualityReport.saved_holdout_renders
    $canReuseTraining = (Test-Path -LiteralPath $FinalPly -PathType Leaf) -and
        $Trainer -eq "Brush" -and $qualityReportValid -and $trainMarker -and $existingAerialDiagnosticEvidence.valid -and
        (Test-ObjectProperty $trainMarker "trainer") -and (Test-ObjectProperty $trainMarker "training_steps") -and
        (Test-ObjectProperty $trainMarker "image_set_hash") -and (Test-ObjectProperty $trainMarker "scene_type") -and
        (Test-ObjectProperty $trainMarker "training_max_resolution") -and
        (Test-ObjectProperty $trainMarker "brush_max_splats") -and
        (Test-ObjectProperty $trainMarker "brush_scale_loss_weight") -and
        (Test-ObjectProperty $trainMarker "seed") -and (Test-ObjectProperty $trainMarker "eval_split_every") -and
        (Test-ObjectProperty $trainMarker "final_ply_sha256") -and
        (Test-ObjectProperty $trainMarker "solve_model_hash") -and
        (Test-ObjectProperty $trainMarker "brush_version") -and
        (Test-ObjectProperty $trainMarker "brush_sha256") -and
        (Test-ObjectProperty $trainMarker "evaluator_sha256") -and
        (Test-ObjectProperty $trainMarker "ply_verifier_sha256") -and
        (Test-ObjectProperty $trainMarker "quality_report_sha256") -and
        (Test-ObjectProperty $trainMarker "holdout_render_set_sha256") -and
        [string]$trainMarker.trainer -eq $Trainer -and [int]$trainMarker.training_steps -eq $TrainingSteps -and
        [string]$trainMarker.image_set_hash -eq $imageSetHash -and [string]$trainMarker.scene_type -eq $SceneType -and
        [int]$trainMarker.training_max_resolution -eq $TrainingMaxResolution -and
        [int]$trainMarker.brush_max_splats -eq $BrushMaxSplats -and
        [double]$trainMarker.brush_scale_loss_weight -eq $BrushScaleLossWeight -and
        [int]$trainMarker.seed -eq $TrainingSeed -and [int]$trainMarker.eval_split_every -eq $EvalSplitEvery -and
        [string]$trainMarker.final_ply_sha256 -eq $existingPlySha -and
        [string]$trainMarker.solve_model_hash -eq $currentModelHash -and
        [string]$trainMarker.brush_version -eq $BrushVersion -and
        [string]$trainMarker.brush_sha256 -eq $BrushSha256 -and
        [string]$trainMarker.evaluator_sha256 -eq $EvaluatorSha256 -and
        [string]$trainMarker.ply_verifier_sha256 -eq $PlyVerifierSha256 -and
        [string]$trainMarker.quality_report_sha256 -eq $existingQualitySha -and
        [string]$trainMarker.holdout_render_set_sha256 -eq $existingHoldoutRenderSetSha
    if ($Trainer -eq "Spirula") {
        $canReuseTraining = [bool](Get-TrainChainState $solvePrerequisite).valid
    } elseif ($Trainer -ne "Brush") {
        $canReuseTraining = $false
    }
    if ($canReuseTraining) {
        Write-Host "[train] Reusing the hash-verified $Trainer result at $TrainingSteps steps"
    } elseif ($Trainer -eq "Brush") {
        $registeredImages = [int]$solveMarker.registered_images
        $brushScaleLossWeightText = $BrushScaleLossWeight.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
        $aerialCloseImages = @()
        $aerialControlImages = @()

        if ($RequiresAerialDiagnostic) {
            $aerialDiagnosticOutput = Join-Path $BrushOutput "aerial-diagnostic-attempt-$AttemptId"
            New-Item -ItemType Directory -Force -Path $aerialDiagnosticOutput | Out-Null
            $aerialSentinelManifest = Join-Path $aerialDiagnosticOutput "aerial_diagnostic_sentinels.json"
            Invoke-LoggedCommand $PythonPath @(
                $AerialSentinelSelectorPath,
                "--model-text", $ModelTextRoot,
                "--eval-split-every", "$AerialDiagnosticEvalSplitEvery",
                "--output", $aerialSentinelManifest
            ) (Join-Path $LogsRoot "select_aerial_diagnostic_sentinels.log")
            $aerialSentinelPayload = Get-Content -LiteralPath $aerialSentinelManifest -Raw | ConvertFrom-Json
            if (-not (Test-ObjectProperty $aerialSentinelPayload "close_images") -or
                -not (Test-ObjectProperty $aerialSentinelPayload "control_images") -or
                @($aerialSentinelPayload.close_images).Count -ne 8 -or
                @($aerialSentinelPayload.control_images).Count -ne 4) {
                throw "The aerial sentinel selector did not produce exactly 8 close and 4 control holdouts."
            }
            $aerialCloseImages = @($aerialSentinelPayload.close_images | ForEach-Object { [string]$_ })
            $aerialControlImages = @($aerialSentinelPayload.control_images | ForEach-Object { [string]$_ })
            $aerialRequestedNames = @($aerialCloseImages + $aerialControlImages)
            if (@($aerialRequestedNames | Sort-Object -Unique).Count -ne 12) {
                throw "The aerial sentinel selector produced duplicate or overlapping holdouts."
            }
            $expectedDiagnosticRenders = [Math]::Ceiling($registeredImages / [double]$AerialDiagnosticEvalSplitEvery)
            if (-not (Test-ObjectProperty $aerialSentinelPayload "counts") -or
                -not (Test-ObjectProperty $aerialSentinelPayload.counts "prospective_holdouts") -or
                [int]$aerialSentinelPayload.counts.prospective_holdouts -ne $expectedDiagnosticRenders) {
                throw "The aerial sentinel manifest does not match the expected Brush evaluation split."
            }

            $aerialDiagnosticStarted = Get-Date
            Invoke-LoggedCommand $BrushPath @(
                "--total-steps", "$AerialDiagnosticSteps", "--export-every", "$AerialDiagnosticSteps",
                "--seed", "$TrainingSeed",
                "--max-resolution", "$AerialDiagnosticMaxResolution",
                "--max-splats", "$AerialDiagnosticMaxSplats",
                "--scale-loss-weight", $brushScaleLossWeightText,
                "--growth-stop-iter", "$AerialDiagnosticGrowthStopIter",
                "--eval-split-every", "$AerialDiagnosticEvalSplitEvery",
                "--eval-every", "$AerialDiagnosticSteps",
                "--eval-save-to-disk",
                "--export-path", $aerialDiagnosticOutput,
                "--export-name", "${RunName}_aerial_diagnostic_{iter}.ply",
                $ReconRoot
            ) (Join-Path $LogsRoot "brush_aerial_diagnostic.log")
            $aerialDiagnosticPly = Get-ChildItem -LiteralPath $aerialDiagnosticOutput -Filter "*.ply" |
                Where-Object { $_.LastWriteTime -ge $aerialDiagnosticStarted.AddSeconds(-5) } |
                Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if (-not $aerialDiagnosticPly) { throw "Brush did not export a new 18K aerial diagnostic PLY." }

            $aerialDiagnosticPlyReport = Join-Path $aerialDiagnosticOutput "aerial_diagnostic_ply_report.json"
            Invoke-LoggedCommand $PythonPath @(
                $PlyVerifierPath, $aerialDiagnosticPly.FullName, "--json", $aerialDiagnosticPlyReport
            ) (Join-Path $LogsRoot "verify_aerial_diagnostic_ply.log")

            $aerialDiagnosticEvalRoot = Join-Path $aerialDiagnosticOutput "eval_$AerialDiagnosticSteps"
            $aerialDiagnosticRenders = @(Get-ChildItem -LiteralPath $aerialDiagnosticEvalRoot -Filter "*.png" -ErrorAction SilentlyContinue)
            if ($aerialDiagnosticRenders.Count -ne $expectedDiagnosticRenders) {
                throw "Brush saved $($aerialDiagnosticRenders.Count) aerial diagnostic renders; expected exactly $expectedDiagnosticRenders."
            }
            if (@($aerialDiagnosticRenders | Where-Object { $_.LastWriteTime -lt $aerialDiagnosticStarted.AddSeconds(-5) }).Count -gt 0) {
                throw "The aerial diagnostic contains stale holdout renders."
            }
            $aerialDiagnosticQualityReport = Join-Path $aerialDiagnosticOutput "aerial_diagnostic_quality_report.json"
            Invoke-LoggedCommand $PythonPath @(
                $EvaluateHoldoutPath,
                "--renders", $aerialDiagnosticEvalRoot,
                "--reference-images", $ImagesPath,
                "--registered-images", (Join-Path $ModelTextRoot "images.txt"),
                "--expected", "$expectedDiagnosticRenders",
                "--json", $aerialDiagnosticQualityReport
            ) (Join-Path $LogsRoot "evaluate_aerial_diagnostic_holdout.log")

            $aerialDiagnosticGateReport = Join-Path $aerialDiagnosticOutput "aerial_diagnostic_gate_report.json"
            $aerialDiagnosticGateArguments = @(
                $AerialGateEvaluatorPath,
                "--report", $aerialDiagnosticQualityReport,
                "--close-images"
            ) + $aerialCloseImages + @(
                "--control-images"
            ) + $aerialControlImages + @(
                "--training-steps", "$AerialDiagnosticSteps",
                "--growth-stop-iter", "$AerialDiagnosticGrowthStopIter",
                "--max-splats", "$AerialDiagnosticMaxSplats",
                "--ply-report", $aerialDiagnosticPlyReport,
                "--json", $aerialDiagnosticGateReport
            )
            Invoke-LoggedCommand $PythonPath $aerialDiagnosticGateArguments (Join-Path $LogsRoot "evaluate_aerial_diagnostic_gate.log")
            $aerialDiagnosticGatePayload = Get-Content -LiteralPath $aerialDiagnosticGateReport -Raw | ConvertFrom-Json
            $null = Assert-AerialGateInputEvidence `
                $aerialDiagnosticGatePayload `
                $aerialDiagnosticQualityReport `
                $aerialDiagnosticPlyReport `
                $aerialDiagnosticPly.FullName

            $aerialSentinelManifestForMarker = $aerialSentinelManifest
            $aerialSentinelManifestShaForMarker = (Get-FileHash -LiteralPath $aerialSentinelManifest -Algorithm SHA256).Hash.ToLowerInvariant()
            $aerialDiagnosticQualityForMarker = $aerialDiagnosticQualityReport
            $aerialDiagnosticQualityShaForMarker = (Get-FileHash -LiteralPath $aerialDiagnosticQualityReport -Algorithm SHA256).Hash.ToLowerInvariant()
            $aerialDiagnosticPlyReportForMarker = $aerialDiagnosticPlyReport
            $aerialDiagnosticPlyReportShaForMarker = (Get-FileHash -LiteralPath $aerialDiagnosticPlyReport -Algorithm SHA256).Hash.ToLowerInvariant()
            $aerialDiagnosticGateForMarker = $aerialDiagnosticGateReport
            $aerialDiagnosticGateShaForMarker = (Get-FileHash -LiteralPath $aerialDiagnosticGateReport -Algorithm SHA256).Hash.ToLowerInvariant()
            $aerialDiagnosticHoldoutShaForMarker = Get-FileSetHash $aerialDiagnosticEvalRoot "*.png"
            $attempt["aerial_diagnostic"]["sentinel_manifest_path"] = $aerialSentinelManifestForMarker
            $attempt["aerial_diagnostic"]["sentinel_manifest_sha256"] = $aerialSentinelManifestShaForMarker
            $attempt["aerial_diagnostic"]["diagnostic_quality_report_path"] = $aerialDiagnosticQualityForMarker
            $attempt["aerial_diagnostic"]["diagnostic_quality_report_sha256"] = $aerialDiagnosticQualityShaForMarker
            $attempt["aerial_diagnostic"]["diagnostic_ply_report_path"] = $aerialDiagnosticPlyReportForMarker
            $attempt["aerial_diagnostic"]["diagnostic_ply_report_sha256"] = $aerialDiagnosticPlyReportShaForMarker
            $attempt["aerial_diagnostic"]["diagnostic_gate_report_path"] = $aerialDiagnosticGateForMarker
            $attempt["aerial_diagnostic"]["diagnostic_gate_report_sha256"] = $aerialDiagnosticGateShaForMarker
            $attempt["aerial_diagnostic"]["diagnostic_holdout_render_set_sha256"] = $aerialDiagnosticHoldoutShaForMarker
            $attempt["aerial_diagnostic"]["diagnostic_status"] = [string]$aerialDiagnosticGatePayload.quality_status
            $attempt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
            Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
            if ([string]$aerialDiagnosticGatePayload.quality_status -ne "MECHANICAL_PASS__AWAITING_VISUAL_QC") {
                throw "The mature aerial diagnostic did not authorize final training: $($aerialDiagnosticGatePayload.quality_status)."
            }
            Write-Host "[train] Mature aerial diagnostic passed; authorizing the $TrainingSteps-step final Brush run" -ForegroundColor Green
        }

        $brushAttemptOutput = Join-Path $BrushOutput "attempt-$AttemptId"
        New-Item -ItemType Directory -Force -Path $brushAttemptOutput, $FinalRoot | Out-Null
        $trainingStarted = Get-Date
        $brushArguments = @(
            "--total-steps", "$TrainingSteps", "--export-every", "$TrainingSteps",
            "--seed", "$TrainingSeed",
            "--max-resolution", "$TrainingMaxResolution",
            "--max-splats", "$BrushMaxSplats",
            "--scale-loss-weight", $brushScaleLossWeightText,
            "--eval-split-every", "$EvalSplitEvery",
            "--eval-every", "$TrainingSteps",
            "--eval-save-to-disk",
            "--export-path", $brushAttemptOutput, "--export-name", "${RunName}_brush_{iter}.ply"
        )
        if ($RequiresAerialDiagnostic) {
            $brushArguments += @("--growth-stop-iter", "$AerialDiagnosticGrowthStopIter")
        }
        $brushArguments += $ReconRoot
        Invoke-LoggedCommand $BrushPath $brushArguments (Join-Path $LogsRoot "brush_train.log")
        $latestPly = Get-ChildItem -LiteralPath $brushAttemptOutput -Filter "*.ply" |
            Where-Object { $_.LastWriteTime -ge $trainingStarted.AddSeconds(-5) } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latestPly) { throw "Brush did not export a new PLY" }
        $CandidatePlyCleanupPath = $latestPly.FullName

        $expectedHoldoutRenders = [Math]::Ceiling($registeredImages / [double]$EvalSplitEvery)
        $evalRenderRoot = Join-Path $brushAttemptOutput "eval_$TrainingSteps"
        $holdoutRenders = @(Get-ChildItem -LiteralPath $evalRenderRoot -Filter "*.png" -ErrorAction SilentlyContinue)
        if ($holdoutRenders.Count -ne $expectedHoldoutRenders) {
            throw "Brush saved $($holdoutRenders.Count) holdout renders; expected exactly $expectedHoldoutRenders. The result was not accepted."
        }
        $staleHoldout = @($holdoutRenders | Where-Object { $_.LastWriteTime -lt $trainingStarted.AddSeconds(-5) })
        if ($staleHoldout.Count -gt 0) {
            throw "Brush holdout directory contains stale renders. The result was not accepted."
        }
        $candidateQualityReport = Join-Path $brushAttemptOutput "training_quality_report.json"
        Invoke-LoggedCommand $PythonPath @(
            $EvaluateHoldoutPath,
            "--renders", $evalRenderRoot,
            "--reference-images", $ImagesPath,
            "--registered-images", (Join-Path $ModelTextRoot "images.txt"),
            "--expected", "$expectedHoldoutRenders",
            "--json", $candidateQualityReport
        ) (Join-Path $LogsRoot "evaluate_holdout.log")

        if ($RequiresAerialDiagnostic) {
            $aerialFinalPlyReport = Join-Path $brushAttemptOutput "aerial_final_candidate_ply_report.json"
            Invoke-LoggedCommand $PythonPath @(
                $PlyVerifierPath, $latestPly.FullName, "--json", $aerialFinalPlyReport
            ) (Join-Path $LogsRoot "verify_aerial_final_candidate.log")
            $aerialFinalGateReport = Join-Path $brushAttemptOutput "aerial_final_gate_report.json"
            $aerialFinalGateArguments = @(
                $AerialGateEvaluatorPath,
                "--report", $candidateQualityReport,
                "--close-images"
            ) + $aerialCloseImages + @(
                "--control-images"
            ) + $aerialControlImages + @(
                "--training-steps", "$TrainingSteps",
                "--growth-stop-iter", "$AerialDiagnosticGrowthStopIter",
                "--max-splats", "$BrushMaxSplats",
                "--ply-report", $aerialFinalPlyReport,
                "--json", $aerialFinalGateReport
            )
            Invoke-LoggedCommand $PythonPath $aerialFinalGateArguments (Join-Path $LogsRoot "evaluate_aerial_final_gate.log")
            $aerialFinalGatePayload = Get-Content -LiteralPath $aerialFinalGateReport -Raw | ConvertFrom-Json
            $aerialFinalInputEvidence = Assert-AerialGateInputEvidence `
                $aerialFinalGatePayload `
                $candidateQualityReport `
                $aerialFinalPlyReport `
                $latestPly.FullName
            $aerialExpectedPublishPlySha = [string]$aerialFinalInputEvidence.ply_sha256
            $aerialExpectedPublishQualitySha = [string]$aerialFinalInputEvidence.quality_report_sha256
            $aerialFinalGateForMarker = $aerialFinalGateReport
            $aerialFinalGateShaForMarker = (Get-FileHash -LiteralPath $aerialFinalGateReport -Algorithm SHA256).Hash.ToLowerInvariant()
            $attempt["aerial_diagnostic"]["final_gate_report_path"] = $aerialFinalGateForMarker
            $attempt["aerial_diagnostic"]["final_gate_report_sha256"] = $aerialFinalGateShaForMarker
            $attempt["aerial_diagnostic"]["final_status"] = [string]$aerialFinalGatePayload.quality_status
            $attempt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
            Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
            if ([string]$aerialFinalGatePayload.quality_status -ne "MECHANICAL_PASS__AWAITING_VISUAL_QC") {
                throw "The final aerial candidate failed the close-view gate and was not published: $($aerialFinalGatePayload.quality_status)."
            }
        }
        if ($RequiresAerialDiagnostic) {
            Publish-VerifiedPly `
                $latestPly.FullName `
                $candidateQualityReport `
                $aerialExpectedPublishPlySha `
                $aerialExpectedPublishQualitySha
        } else {
            Publish-VerifiedPly $latestPly.FullName $candidateQualityReport
        }
        $CandidatePlyCleanupPath = $null
    } elseif ($Trainer -eq "Spirula") {
        $registeredImages = [int]$solveMarker.registered_images
        $expectedHoldoutRenders = [Math]::Ceiling($registeredImages / [double]$EvalSplitEvery)
        $spirulaAttemptOutput = Join-Path $SpirulaOutput "attempt-$AttemptId"
        $spirulaRunRoot = Join-Path $spirulaAttemptOutput "run"
        New-Item -ItemType Directory -Force -Path $spirulaAttemptOutput, $FinalRoot | Out-Null
        $checkpointFlag = if ($SpirulaSaveFullCheckpoint) { "1" } else { "0" }
        $stepsPerSave = [Math]::Min(5000, $TrainingSteps)
        Invoke-LoggedCommand $SpirulaPath @(
            "train", "3dgs",
            "--data", $ReconRoot,
            "--data-format", "colmap",
            "--colmap-recon-dir", "sparse/0",
            "--image-dir", "images",
            "--output-dir-prefix", $spirulaAttemptOutput,
            "--output-dir-name", "run",
            "--num-iterations", "$TrainingSteps",
            "--cap-max", "$BrushMaxSplats",
            "--sh-degree", "3",
            "--eval-mode", "interval",
            "--eval-interval", "$EvalSplitEvery",
            "--save-eval-images", "1",
            "--disable-viewer", "1",
            "--keep-viewer-alive", "0",
            "--steps-per-save", "$stepsPerSave",
            "--save-only-latest-checkpoint", "1",
            "--save-full-checkpoint", $checkpointFlag,
            "--floater-suppression", $SpirulaFloaterSuppression,
            "--distraction-robustness", $SpirulaDistractionRobustness
        ) (Join-Path $LogsRoot "spirula_train.log")
        $spirulaMetricsPathForMarker = Join-Path $spirulaRunRoot "metrics.json"
        $spirulaConfigPathForMarker = Join-Path $spirulaRunRoot "config.json"
        if (-not (Test-Path -LiteralPath $spirulaMetricsPathForMarker -PathType Leaf)) {
            throw "Spirula did not produce native holdout metrics. The result was not accepted."
        }
        if (-not (Test-Path -LiteralPath $spirulaConfigPathForMarker -PathType Leaf)) {
            throw "Spirula did not produce a signed run config. The result was not accepted."
        }
        $checkpointPly = Get-ChildItem -LiteralPath $spirulaRunRoot -Filter "splat.ply" -File -Recurse |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $checkpointPly) { throw "Spirula did not produce a checkpoint splat.ply." }
        $candidateSpirulaPlyPath = Join-Path $spirulaAttemptOutput "spirula_candidate.ply"
        Copy-Item -LiteralPath $checkpointPly.FullName -Destination $candidateSpirulaPlyPath
        $latestSpirulaPly = Get-Item -LiteralPath $candidateSpirulaPlyPath
        $CandidatePlyCleanupPath = $latestSpirulaPly.FullName
        $candidateQualityReport = Join-Path $spirulaRunRoot "training_quality_report.json"
        Invoke-LoggedCommand $PythonPath @(
            $EvaluateSpirulaHoldoutPath,
            "--run-dir", $spirulaRunRoot,
            "--expected", "$expectedHoldoutRenders",
            "--json", $candidateQualityReport
        ) (Join-Path $LogsRoot "evaluate_spirula_holdout.log")
        Publish-VerifiedPly $latestSpirulaPly.FullName $candidateQualityReport
        if (Test-Path -LiteralPath $checkpointPly.FullName -PathType Leaf) {
            Remove-Item -LiteralPath $checkpointPly.FullName -Force
        }
        $CandidatePlyCleanupPath = $null
    } else {
        if (-not $ThreeDGrutRepo -or -not $ThreeDGrutPython -or -not $ThreeDGrutModelConfig -or
            -not $ThreeDGrutSourceSha256 -or -not $ThreeDGrutPythonSha256 -or -not $ThreeDGrutConfigSha256 -or
            -not $ThreeDGrutEnvironmentSha256) {
            throw "3DGRUT is not configured. Add ThreeDGRUT.Repo and ThreeDGRUT.Python to splatitup.local.psd1."
        }
        $undistortedMarkerPath = Join-Path $UndistortedRoot ".complete.json"
        $undistortedMarker = if (Test-Path -LiteralPath $undistortedMarkerPath -PathType Leaf) {
            try { Get-Content -LiteralPath $undistortedMarkerPath -Raw | ConvertFrom-Json } catch { $null }
        } else { $null }
        $undistortedImages = Join-Path $UndistortedRoot "images"
        $undistortedSparseZero = Join-Path $UndistortedRoot "sparse\0"
        $undistortedCount = @(Get-ChildItem -LiteralPath $undistortedImages -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
        $undistortedImageHash = if ($undistortedCount -gt 0) { Get-ImageSetHash $undistortedImages } else { $null }
        $undistortedModelHash = Get-ColmapModelHash $undistortedSparseZero
        $canReuseUndistorted = $undistortedMarker -and $undistortedCount -gt 0 -and $undistortedModelHash -and
            (Test-ObjectProperty $undistortedMarker "input_image_set_hash") -and [string]$undistortedMarker.input_image_set_hash -eq $imageSetHash -and
            (Test-ObjectProperty $undistortedMarker "input_model_hash") -and [string]$undistortedMarker.input_model_hash -eq $currentModelHash -and
            (Test-ObjectProperty $undistortedMarker "max_image_size") -and [int]$undistortedMarker.max_image_size -eq $TrainingMaxResolution -and
            (Test-ObjectProperty $undistortedMarker "output_image_set_hash") -and [string]$undistortedMarker.output_image_set_hash -eq $undistortedImageHash -and
            (Test-ObjectProperty $undistortedMarker "output_model_hash") -and [string]$undistortedMarker.output_model_hash -eq $undistortedModelHash -and
            (Test-ObjectProperty $undistortedMarker "output_images") -and [int]$undistortedMarker.output_images -eq $undistortedCount -and
            (Test-ObjectProperty $undistortedMarker "colmap_sha256") -and [string]$undistortedMarker.colmap_sha256 -eq $ColmapSha256
        if (-not $canReuseUndistorted) {
            if (Test-Path -LiteralPath $UndistortedRoot) { Remove-Item -LiteralPath $UndistortedRoot -Recurse -Force }
            New-Item -ItemType Directory -Force -Path $UndistortedRoot | Out-Null
            Invoke-LoggedCommand $ColmapPath @(
                "image_undistorter", "--image_path", $ImagesPath, "--input_path", $SparsePath,
                "--output_path", $UndistortedRoot, "--output_type", "COLMAP",
                "--max_image_size", "$TrainingMaxResolution"
            ) (Join-Path $LogsRoot "colmap_undistort.log")
            $undistortedSparse = Join-Path $UndistortedRoot "sparse"
            $undistortedSparseZero = Join-Path $undistortedSparse "0"
            if ((Test-Path -LiteralPath (Join-Path $undistortedSparse "cameras.bin")) -and -not (Test-Path -LiteralPath $undistortedSparseZero)) {
                New-Item -ItemType Directory -Force -Path $undistortedSparseZero | Out-Null
                Copy-Item -LiteralPath (Join-Path $undistortedSparse "cameras.bin"),(Join-Path $undistortedSparse "images.bin"),(Join-Path $undistortedSparse "points3D.bin") -Destination $undistortedSparseZero
            }
            $undistortedCount = @(Get-ChildItem -LiteralPath $undistortedImages -Filter "*.jpg").Count
            if ($undistortedCount -lt 2) { throw "COLMAP undistortion produced only $undistortedCount images" }
            $undistortedImageHash = Get-ImageSetHash $undistortedImages
            $undistortedModelHash = Get-ColmapModelHash $undistortedSparseZero
            if (-not $undistortedModelHash) { throw "COLMAP undistortion did not produce a complete sparse model" }
            [ordered]@{
                input_image_set_hash = $imageSetHash
                input_model_hash = $currentModelHash
                max_image_size = $TrainingMaxResolution
                output_images = $undistortedCount
                output_image_set_hash = $undistortedImageHash
                output_model_hash = $undistortedModelHash
                colmap_sha256 = $ColmapSha256
            } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath "$undistortedMarkerPath.tmp" -Encoding utf8
            Move-Item -LiteralPath "$undistortedMarkerPath.tmp" -Destination $undistortedMarkerPath -Force
        }
        $threeDGrutAttemptOutput = Join-Path $ThreeDGrutOutput "attempt-$AttemptId"
        New-Item -ItemType Directory -Force -Path $threeDGrutAttemptOutput, $FinalRoot | Out-Null
        if ($RequiresThreeDGrutAerialStages) {
            $smokeDataset = Join-Path $threeDGrutAttemptOutput "smoke-dataset"
            $smokeDatasetReport = Join-Path $smokeDataset "smoke-dataset-report.json"
            Invoke-LoggedCommand $PythonPath @(
                $ThreeDGrutSmokePreparerPath,
                "--source-dataset", $UndistortedRoot,
                "--source-points", (Join-Path $ModelTextRoot "points3D.txt"),
                "--output", $smokeDataset,
                "--max-points", "$ThreeDGrutSmokeMaxSplats",
                "--json", $smokeDatasetReport
            ) (Join-Path $LogsRoot "prepare_3dgrut_smoke_dataset.log") | Out-Host
            $smokeImages = Join-Path $smokeDataset "images"
            New-Item -ItemType Junction -Path $smokeImages -Target $undistortedImages | Out-Null
            if (-not (Test-Path -LiteralPath $smokeImages -PathType Container)) {
                throw "3DGRUT smoke dataset image junction was not created."
            }

            $smokeStage = Invoke-ThreeDGrutTrainingStage `
                -Stage "smoke" `
                -Steps $ThreeDGrutSmokeSteps `
                -MaxSplats $ThreeDGrutSmokeMaxSplats `
                -DatasetPath $smokeDataset `
                -StageRoot (Join-Path $threeDGrutAttemptOutput "smoke-$ThreeDGrutSmokeSteps") `
                -DatasetImageSha256 $undistortedImageHash `
                -DatasetModelSha256 $undistortedModelHash
            $attempt["three_dgrut"]["smoke_status"] = "MECHANICAL_PASS__AWAITING_VISUAL_QC"
            $smokeDecision = [ordered]@{
                schema_version = 1
                status = "AWAITING_VISUAL_QC"
                code = "THREEDGRUT_SMOKE_AWAITING_VISUAL_QC"
                reason = "The exact 1K/250K 3DGRUT-MCMC smoke passed mechanical checks, but this backend is not approved for automatic continuation. No 7K, final, or publish stage ran."
                attempt_id = $AttemptId
                created_utc = [DateTime]::UtcNow.ToString("o")
                scene_type = $SceneType
                trainer = $Trainer
                training_stage = "smoke"
                training_steps = $ThreeDGrutSmokeSteps
                max_splats = $ThreeDGrutSmokeMaxSplats
                evidence = [ordered]@{
                    stage_report_path = $smokeStage.report
                    stage_report_sha256 = [string]$smokeStage.report_sha256
                    ply_path = $smokeStage.ply
                    ply_sha256 = (Get-FileHash -LiteralPath $smokeStage.ply -Algorithm SHA256).Hash.ToLowerInvariant()
                    checkpoint_path = $smokeStage.checkpoint
                    checkpoint_sha256 = (Get-FileHash -LiteralPath $smokeStage.checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
                    metrics_path = $smokeStage.metrics
                    metrics_sha256 = (Get-FileHash -LiteralPath $smokeStage.metrics -Algorithm SHA256).Hash.ToLowerInvariant()
                    renders_path = $smokeStage.renders
                    render_count = @(Get-ChildItem -LiteralPath $smokeStage.renders -Filter "*.png" -File).Count
                    smoke_dataset_report_path = $smokeDatasetReport
                    smoke_dataset_report_sha256 = (Get-FileHash -LiteralPath $smokeDatasetReport -Algorithm SHA256).Hash.ToLowerInvariant()
                }
            }
            $smokeDecision | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$TrainingDecisionPath.tmp" -Encoding utf8
            Move-Item -LiteralPath "$TrainingDecisionPath.tmp" -Destination $TrainingDecisionPath -Force
            $attempt["status"] = "AWAITING_VISUAL_QC"
            $attempt["completed_utc"] = [DateTime]::UtcNow.ToString("o")
            $attempt["terminal_code"] = [string]$smokeDecision.code
            $attempt["training_decision_path"] = $TrainingDecisionPath
            $attempt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
            Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
            Write-Warning "$($smokeDecision.code): $($smokeDecision.reason)"
            exit 2
        } else {
            throw "EXPERIMENTAL_TRAINER_UNSUPPORTED: No unrated 3DGRUT path may train, publish, or be reused."
        }
    }

    Invoke-LoggedCommand $PythonPath @(
        $PlyVerifierPath, $FinalPly,
        "--json", (Join-Path $RunRoot "gaussian_ply_report.json")
    ) (Join-Path $LogsRoot "verify_ply.log")
    $plyReport = Get-Content -LiteralPath (Join-Path $RunRoot "gaussian_ply_report.json") -Raw | ConvertFrom-Json
    $completedQualityReport = Get-Content -LiteralPath $QualityReportPath -Raw | ConvertFrom-Json
    $holdoutRenderSetSha = if ($completedQualityReport -and (Test-ObjectProperty $completedQualityReport "render_directory")) {
        Get-FileSetHash ([string]$completedQualityReport.render_directory) "*.png"
    } else { $null }
    $holdoutReferenceSetSha = if ($Trainer -eq "Spirula" -and $completedQualityReport -and
        (Test-ObjectProperty $completedQualityReport "reference_directory")) {
        Get-FileSetHash ([string]$completedQualityReport.reference_directory) "*.png"
    } else { $null }
    $spirulaMetricsShaForMarker = if ($spirulaMetricsPathForMarker -and
        (Test-Path -LiteralPath $spirulaMetricsPathForMarker -PathType Leaf)) {
        (Get-FileHash -LiteralPath $spirulaMetricsPathForMarker -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    $spirulaConfigShaForMarker = if ($spirulaConfigPathForMarker -and
        (Test-Path -LiteralPath $spirulaConfigPathForMarker -PathType Leaf)) {
        (Get-FileHash -LiteralPath $spirulaConfigPathForMarker -Algorithm SHA256).Hash.ToLowerInvariant()
    } else { $null }
    Write-Marker "train" ([ordered]@{
        final_ply = $FinalPly
        final_ply_sha256 = [string]$plyReport.sha256
        solve_model_hash = $currentModelHash
        image_set_hash = $imageSetHash
        training_steps = $TrainingSteps
        trainer = $Trainer
        scene_type = $SceneType
        training_max_resolution = $TrainingMaxResolution
        brush_max_splats = $BrushMaxSplats
        three_dgrut_mcmc_max_splats = if ($Trainer -eq "3DGUT-MCMC") { $ThreeDGrutMcmcMaxSplats } else { $null }
        brush_scale_loss_weight = $BrushScaleLossWeight
        seed = $TrainingSeed
        eval_split_every = $EvalSplitEvery
        brush_version = $BrushVersion
        brush_sha256 = $BrushSha256
        spirula_version = $SpirulaVersion
        spirula_sha256 = $SpirulaSha256
        spirula_max_splats = if ($Trainer -eq "Spirula") { $BrushMaxSplats } else { $null }
        spirula_floater_suppression = if ($Trainer -eq "Spirula") { $SpirulaFloaterSuppression } else { $null }
        spirula_distraction_robustness = if ($Trainer -eq "Spirula") { $SpirulaDistractionRobustness } else { $null }
        spirula_save_full_checkpoint = if ($Trainer -eq "Spirula") { $SpirulaSaveFullCheckpoint } else { $null }
        spirula_metrics_path = $spirulaMetricsPathForMarker
        spirula_metrics_sha256 = $spirulaMetricsShaForMarker
        spirula_config_path = $spirulaConfigPathForMarker
        spirula_config_sha256 = $spirulaConfigShaForMarker
        evaluator_sha256 = $EvaluatorSha256
        ply_verifier_sha256 = $PlyVerifierSha256
        quality_report_sha256 = (Get-FileHash -LiteralPath $QualityReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        holdout_render_set_sha256 = $holdoutRenderSetSha
        holdout_reference_set_sha256 = $holdoutReferenceSetSha
        aerial_diagnostic_required = $RequiresAerialDiagnostic
        aerial_diagnostic_steps = if ($RequiresAerialDiagnostic) { $AerialDiagnosticSteps } else { $null }
        aerial_diagnostic_growth_stop_iter = if ($RequiresAerialDiagnostic) { $AerialDiagnosticGrowthStopIter } else { $null }
        aerial_diagnostic_max_splats = if ($RequiresAerialDiagnostic) { $AerialDiagnosticMaxSplats } else { $null }
        aerial_diagnostic_max_resolution = if ($RequiresAerialDiagnostic) { $AerialDiagnosticMaxResolution } else { $null }
        aerial_diagnostic_eval_split_every = if ($RequiresAerialDiagnostic) { $AerialDiagnosticEvalSplitEvery } else { $null }
        aerial_sentinel_selector_sha256 = $AerialSentinelSelectorSha256
        aerial_gate_evaluator_sha256 = $AerialGateEvaluatorSha256
        aerial_sentinel_manifest_path = $aerialSentinelManifestForMarker
        aerial_sentinel_manifest_sha256 = $aerialSentinelManifestShaForMarker
        aerial_diagnostic_quality_report_path = $aerialDiagnosticQualityForMarker
        aerial_diagnostic_quality_report_sha256 = $aerialDiagnosticQualityShaForMarker
        aerial_diagnostic_ply_report_path = $aerialDiagnosticPlyReportForMarker
        aerial_diagnostic_ply_report_sha256 = $aerialDiagnosticPlyReportShaForMarker
        aerial_diagnostic_gate_report_path = $aerialDiagnosticGateForMarker
        aerial_diagnostic_gate_report_sha256 = $aerialDiagnosticGateShaForMarker
        aerial_diagnostic_holdout_render_set_sha256 = $aerialDiagnosticHoldoutShaForMarker
        aerial_final_gate_report_path = $aerialFinalGateForMarker
        aerial_final_gate_report_sha256 = $aerialFinalGateShaForMarker
        three_dgrut_repo = $ThreeDGrutRepo
        three_dgrut_commit = $ThreeDGrutCommit
        three_dgrut_source_sha256 = $ThreeDGrutSourceSha256
        three_dgrut_python_sha256 = $ThreeDGrutPythonSha256
        three_dgrut_config_sha256 = $ThreeDGrutConfigSha256
        three_dgrut_environment_sha256 = $ThreeDGrutEnvironmentSha256
        three_dgrut_environment_fingerprint_version = if ($Trainer -in @("3DGUT", "3DGUT-MCMC")) { $ThreeDGrutEnvironmentFingerprintVersion } else { $null }
        three_dgrut_stage_evaluator_sha256 = $ThreeDGrutStageEvaluatorSha256
        three_dgrut_smoke_preparer_sha256 = $ThreeDGrutSmokePreparerSha256
        three_dgrut_smoke_dataset_report_path = $threeDGrutSmokeDatasetForMarker
        three_dgrut_smoke_dataset_report_sha256 = $threeDGrutSmokeDatasetShaForMarker
        three_dgrut_smoke_stage_report_path = $threeDGrutSmokeStageForMarker
        three_dgrut_smoke_stage_report_sha256 = $threeDGrutSmokeStageShaForMarker
        three_dgrut_diagnostic_stage_report_path = $threeDGrutDiagnosticStageForMarker
        three_dgrut_diagnostic_stage_report_sha256 = $threeDGrutDiagnosticStageShaForMarker
        three_dgrut_final_stage_report_path = $threeDGrutFinalStageForMarker
        three_dgrut_final_stage_report_sha256 = $threeDGrutFinalStageShaForMarker
    })
    if (Test-Path -LiteralPath (Join-Path $RunRoot "publish_transaction.json") -PathType Leaf) {
        Complete-PublishTransaction
    }
    Write-Host "[train] $Trainer Gaussian PLY ready: $FinalPly" -ForegroundColor Green
}

if (Test-Stage "blender") {
    Assert-ApprovedTrainerForDownstream "Blender handoff"
    if ($NoBlender) {
        Write-Warning "Blender handoff was explicitly disabled with -NoBlender"
    } else {
        $trainPrerequisite = Get-TrainChainState $null
        if (-not $trainPrerequisite.valid) {
            throw "The final PLY does not have a current source-to-training provenance chain. Restart from train before building Blender."
        }
        if (-not (Test-Path -LiteralPath $FinalPly -PathType Leaf)) { throw "Final PLY is missing; start from train" }
        foreach ($required in @("cameras.txt", "images.txt")) {
            if (-not (Test-Path -LiteralPath (Join-Path $ModelTextRoot $required) -PathType Leaf)) {
                throw "COLMAP text model is incomplete; rerun from solve before building the Blender handoff"
            }
        }
        $plyReportPath = Join-Path $RunRoot "gaussian_ply_report.json"
        Invoke-LoggedCommand $PythonPath @(
            (Join-Path $PSScriptRoot "verify_gaussian_ply.py"), $FinalPly, "--json", $plyReportPath
        ) (Join-Path $LogsRoot "verify_ply_for_blender.log")
        $plyReport = Get-Content -LiteralPath $plyReportPath -Raw | ConvertFrom-Json
        $handoffScriptPath = Join-Path $PSScriptRoot "blender_handoff.py"
        $syncScriptPath = Join-Path $PSScriptRoot "kiri_camera_sync.py"
        $handoffScriptSha = (Get-FileHash -LiteralPath $handoffScriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $syncScriptSha = (Get-FileHash -LiteralPath $syncScriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $camerasTextSha = (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "cameras.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
        $imagesTextSha = (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "images.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
        $poseReportSha = (Get-FileHash -LiteralPath (Join-Path $RunRoot "reconstruction_report.json") -Algorithm SHA256).Hash.ToLowerInvariant()
        $plyReportSha = (Get-FileHash -LiteralPath $plyReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $blenderMarker = Read-Marker "blender"
        $currentBlendSha = if (Test-Path -LiteralPath $BlenderFile -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderFile -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
        $currentHandoffReportSha = if (Test-Path -LiteralPath $BlenderReportPath -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderReportPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
        $currentPreviewSha = if (Test-Path -LiteralPath $BlenderPreviewPath -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderPreviewPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
        $canReuseBlender = (Test-Path -LiteralPath $BlenderFile -PathType Leaf) -and
            (Test-Path -LiteralPath $BlenderReportPath -PathType Leaf) -and $blenderMarker -and
            (Test-Path -LiteralPath $BlenderPreviewPath -PathType Leaf) -and
            (Test-ObjectProperty $blenderMarker "source_ply_sha256") -and
            (Test-ObjectProperty $blenderMarker "scene_type") -and
            (Test-ObjectProperty $blenderMarker "blender_builder_version") -and
            (Test-ObjectProperty $blenderMarker "blender_builder_sha256") -and
            (Test-ObjectProperty $blenderMarker "blender_open_sha256") -and
            (Test-ObjectProperty $blenderMarker "handoff_script_sha256") -and
            (Test-ObjectProperty $blenderMarker "sync_script_sha256") -and
            (Test-ObjectProperty $blenderMarker "cameras_text_sha256") -and
            (Test-ObjectProperty $blenderMarker "images_text_sha256") -and
            (Test-ObjectProperty $blenderMarker "pose_report_sha256") -and
            (Test-ObjectProperty $blenderMarker "ply_report_sha256") -and
            (Test-ObjectProperty $blenderMarker "blend_file_sha256") -and
            (Test-ObjectProperty $blenderMarker "handoff_report_sha256") -and
            (Test-ObjectProperty $blenderMarker "preview_sha256") -and
            [string]$blenderMarker.source_ply_sha256 -eq [string]$plyReport.sha256 -and
            [string]$blenderMarker.scene_type -eq $SceneType -and
            [string]$blenderMarker.blender_builder_version -eq $BlenderBuilderVersion -and
            [string]$blenderMarker.blender_builder_sha256 -eq $BlenderBuilderSha256 -and
            [string]$blenderMarker.blender_open_sha256 -eq $BlenderOpenSha256 -and
            [string]$blenderMarker.handoff_script_sha256 -eq $handoffScriptSha -and
            [string]$blenderMarker.sync_script_sha256 -eq $syncScriptSha -and
            [string]$blenderMarker.cameras_text_sha256 -eq $camerasTextSha -and
            [string]$blenderMarker.images_text_sha256 -eq $imagesTextSha -and
            [string]$blenderMarker.pose_report_sha256 -eq $poseReportSha -and
            [string]$blenderMarker.ply_report_sha256 -eq $plyReportSha -and
            [string]$blenderMarker.blend_file_sha256 -eq $currentBlendSha -and
            [string]$blenderMarker.handoff_report_sha256 -eq $currentHandoffReportSha -and
            [string]$blenderMarker.preview_sha256 -eq $currentPreviewSha
        if ($canReuseBlender) {
            Write-Host "[blender] Reusing the hash-matched Blender handoff"
        } else {
            New-Item -ItemType Directory -Force -Path $BlenderRoot | Out-Null
            if (Test-Path -LiteralPath $BlenderFile -PathType Leaf) {
                $blenderHistoryRoot = Join-Path $BlenderRoot "history"
                New-Item -ItemType Directory -Force -Path $blenderHistoryRoot | Out-Null
                $existingBlendSha = (Get-FileHash -LiteralPath $BlenderFile -Algorithm SHA256).Hash.ToLowerInvariant()
                $archiveTag = "$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))-$($existingBlendSha.Substring(0, 12))-$AttemptId"
                Move-Item -LiteralPath $BlenderFile -Destination (Join-Path $blenderHistoryRoot "$RunName-$archiveTag.blend")
                foreach ($proof in @($BlenderReportPath, $BlenderOpenReportPath, $BlenderPreviewPath)) {
                    if (Test-Path -LiteralPath $proof -PathType Leaf) {
                        Move-Item -LiteralPath $proof -Destination (Join-Path $blenderHistoryRoot "$archiveTag-$([System.IO.Path]::GetFileName($proof))")
                    }
                }
                Write-Warning "Preserved the previous or user-edited Blender handoff under $blenderHistoryRoot before rebuilding."
            } else {
                foreach ($staleProof in @($BlenderReportPath, $BlenderOpenReportPath, $BlenderPreviewPath)) {
                    if (Test-Path -LiteralPath $staleProof) { Remove-Item -LiteralPath $staleProof -Force }
                }
            }
            Invoke-LoggedCommand $BlenderBuilderPath @(
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--addons", "bl_ext.user_default.dgs_render_by_kiri_engine",
                "--python", $handoffScriptPath,
                "--",
                "--ply", $FinalPly,
                "--model-text", $ModelTextRoot,
                "--pose-report", (Join-Path $RunRoot "reconstruction_report.json"),
                "--ply-report", $plyReportPath,
                "--output", $BlenderFile,
                "--report", $BlenderReportPath,
                "--preview", $BlenderPreviewPath,
                "--sync-script", $syncScriptPath,
                "--profile", $SceneType
            ) (Join-Path $LogsRoot "blender_handoff.log")
        }
        $blenderReport = Get-Content -LiteralPath $BlenderReportPath -Raw | ConvertFrom-Json
        if ([string]$blenderReport.status -ne "MECHANICAL PASS" -or
            [string]$blenderReport.visual_approval -ne "AWAITING USER APPROVAL" -or
            -not [bool]$blenderReport.render_proof.nonblank) {
            throw "Blender handoff report did not contain the required fail-closed status"
        }
        $blenderValidationScriptPath = Join-Path $PSScriptRoot "blender_validate.py"
        $blenderValidationScriptSha = (Get-FileHash -LiteralPath $blenderValidationScriptPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if (Test-Path -LiteralPath $BlenderOpenReportPath) { Remove-Item -LiteralPath $BlenderOpenReportPath -Force }
        Invoke-LoggedCommand $BlenderOpenPath @(
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            $BlenderFile,
            "--python", $blenderValidationScriptPath,
            "--",
            "--handoff-report", $BlenderReportPath,
            "--sync-script", $syncScriptPath,
            "--report", $BlenderOpenReportPath
        ) (Join-Path $LogsRoot "blender_52_open.log")
        if (-not (Test-Path -LiteralPath $BlenderOpenReportPath -PathType Leaf)) {
            throw "Blender 5.2 did not produce its destination validation report"
        }
        $blenderOpenReport = Get-Content -LiteralPath $BlenderOpenReportPath -Raw | ConvertFrom-Json
        if ([string]$blenderOpenReport.status -ne "MECHANICAL PASS" -or
            [string]$blenderOpenReport.visual_approval -ne "AWAITING USER APPROVAL") {
            throw "Blender 5.2 destination validation failed"
        }
        $blendFileSha = (Get-FileHash -LiteralPath $BlenderFile -Algorithm SHA256).Hash.ToLowerInvariant()
        $handoffReportSha = (Get-FileHash -LiteralPath $BlenderReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $openReportSha = (Get-FileHash -LiteralPath $BlenderOpenReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $previewSha = (Get-FileHash -LiteralPath $BlenderPreviewPath -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Marker "blender" ([ordered]@{
            blend_file = $BlenderFile
            source_ply_sha256 = [string]$plyReport.sha256
            scene_type = $SceneType
            blender_builder_version = $BlenderBuilderVersion
            blender_open_version = $BlenderOpenVersion
            blender_builder_sha256 = $BlenderBuilderSha256
            blender_open_sha256 = $BlenderOpenSha256
            kiri_addon_version = "4.1.5"
            handoff_script_sha256 = $handoffScriptSha
            sync_script_sha256 = $syncScriptSha
            cameras_text_sha256 = $camerasTextSha
            images_text_sha256 = $imagesTextSha
            pose_report_sha256 = $poseReportSha
            ply_report_sha256 = $plyReportSha
            validation_script_sha256 = $blenderValidationScriptSha
            blend_file_sha256 = $blendFileSha
            blend_file_bytes = (Get-Item -LiteralPath $BlenderFile).Length
            handoff_report_sha256 = $handoffReportSha
            blender_open_report_sha256 = $openReportSha
            preview_sha256 = $previewSha
            status = [string]$blenderReport.status
            blender_52_status = [string]$blenderOpenReport.status
            visual_approval = [string]$blenderReport.visual_approval
            render_proof = $BlenderPreviewPath
        })
        Write-Host "[blender] Self-contained scene ready: $BlenderFile" -ForegroundColor Green
        if ($OpenBlender) {
            Start-Process -FilePath $BlenderOpenPath -ArgumentList "--enable-autoexec `"$BlenderFile`"" | Out-Null
        }
    }
}

if (Test-Stage "view") {
    Assert-ApprovedTrainerForDownstream "viewer handoff"
    $viewPrerequisite = Get-TrainChainState $null
    if (-not $viewPrerequisite.valid) {
        throw "The final PLY does not have a current source-to-training provenance chain. Restart from train before viewing it."
    }
    if (-not (Test-Path -LiteralPath $FinalPly -PathType Leaf)) { throw "Final PLY is missing; start from train" }
    if ($SuperSplatDist) {
        $port = $ViewerPort
        while (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $port++ }
        $viewerScript = Join-Path $PSScriptRoot "serve_supersplat.py"
        $viewerArguments = "`"$viewerScript`" `"$SuperSplatDist`" `"$FinalPly`" $port"
        Start-Process -FilePath $PythonPath -ArgumentList $viewerArguments -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 1
        $encodedPlyUrl = [uri]::EscapeDataString("http://127.0.0.1:$port/$([System.IO.Path]::GetFileName($FinalPly))")
        $viewerUrl = "http://127.0.0.1:$port/?load=$encodedPlyUrl"
        $viewerUrl | Set-Content -LiteralPath (Join-Path $RunRoot "viewer_url.txt") -Encoding ascii
        Write-Host "[view] $viewerUrl" -ForegroundColor Green
        if (-not $NoBrowser) { Start-Process $viewerUrl | Out-Null }
    } else {
        Write-Warning "Local SuperSplat build is not configured. Opening the web editor and PLY folder instead."
        if (-not $NoBrowser) {
            Start-Process "https://superspl.at/editor" | Out-Null
            Start-Process -FilePath "explorer.exe" -ArgumentList "/select,`"$FinalPly`"" | Out-Null
        }
    }
}

$currentPlyReport = if (Test-Path -LiteralPath (Join-Path $RunRoot "gaussian_ply_report.json") -PathType Leaf) {
    try { Get-Content -LiteralPath (Join-Path $RunRoot "gaussian_ply_report.json") -Raw | ConvertFrom-Json } catch { $null }
} else { $null }
$currentPlySha = if (Test-Path -LiteralPath $FinalPly -PathType Leaf) {
    (Get-FileHash -LiteralPath $FinalPly -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$currentPlyPayloadValid = $null -ne $currentPlyReport -and
    (Test-ObjectProperty $currentPlyReport "valid_gaussian_ply") -and [bool]$currentPlyReport.valid_gaussian_ply -and
    (Test-ObjectProperty $currentPlyReport "sha256") -and [string]$currentPlyReport.sha256 -eq $currentPlySha

$currentExtractMarker = Read-Marker "extract"
$currentRawCount = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
$currentRawHash = if ($currentRawCount -gt 0) { Get-ImageSetHash $RawFrames } else { $null }
$currentExtractValid = $currentExtractMarker -and $currentRawCount -gt 0 -and
    (Test-ObjectProperty $currentExtractMarker "source_sha256") -and [string]$currentExtractMarker.source_sha256 -eq $SourceSha256 -and
    (Test-ObjectProperty $currentExtractMarker "selected_frame_target") -and [int]$currentExtractMarker.selected_frame_target -eq $SelectedFrames -and
    (Test-ObjectProperty $currentExtractMarker "adaptive_extraction") -and [bool]$currentExtractMarker.adaptive_extraction -eq [bool]$AdaptiveExtraction -and
    (Test-ObjectProperty $currentExtractMarker "candidate_multiplier") -and [int]$currentExtractMarker.candidate_multiplier -eq $CandidateMultiplier -and
    (Test-ObjectProperty $currentExtractMarker "max_long_side") -and [int]$currentExtractMarker.max_long_side -eq $MaxLongSide -and
    (Test-ObjectProperty $currentExtractMarker "auto_rotate") -and [bool]$currentExtractMarker.auto_rotate -eq (-not [bool]$NoAutoRotate) -and
    (Test-ObjectProperty $currentExtractMarker "raw_frames") -and [int]$currentExtractMarker.raw_frames -eq $currentRawCount -and
    (Test-ObjectProperty $currentExtractMarker "raw_frame_hash") -and [string]$currentExtractMarker.raw_frame_hash -eq $currentRawHash -and
    (Test-ObjectProperty $currentExtractMarker "ffmpeg_sha256") -and [string]$currentExtractMarker.ffmpeg_sha256 -eq $FfmpegSha256 -and
    (Test-ObjectProperty $currentExtractMarker "ffprobe_sha256") -and [string]$currentExtractMarker.ffprobe_sha256 -eq $FfprobeSha256

$currentSelectMarker = Read-Marker "select"
$currentSelectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
$currentSelectedHash = if ($currentSelectedCount -gt 0) { Get-ImageSetHash $ImagesPath } else { $null }
$currentSelectMinimumCount = [Math]::Min($SelectedFrames, $currentRawCount)
$currentSelectCountValid = if ($MaxCumulativeFlow -gt 0.0) {
    $currentSelectedCount -ge $currentSelectMinimumCount -and $currentSelectedCount -le $currentRawCount
} else {
    $currentSelectedCount -eq $currentSelectMinimumCount
}
$currentFrameQualitySha = if (Test-Path -LiteralPath $FrameQualityPath -PathType Leaf) {
    (Get-FileHash -LiteralPath $FrameQualityPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$currentFrameQuality = if ($currentFrameQualitySha) {
    try { Get-Content -LiteralPath $FrameQualityPath -Raw | ConvertFrom-Json } catch { $null }
} else { $null }
$currentFrameQualityValid = $currentFrameQuality -and
    (Test-ObjectProperty $currentFrameQuality "selected_frames") -and [int]$currentFrameQuality.selected_frames -eq $currentSelectedCount -and
    (Test-ObjectProperty $currentFrameQuality "max_cumulative_flow") -and [double]$currentFrameQuality.max_cumulative_flow -eq $MaxCumulativeFlow
$currentSelectValid = $currentExtractValid -and $currentSelectMarker -and $currentSelectCountValid -and $currentFrameQualityValid -and
    (Test-ObjectProperty $currentSelectMarker "raw_frame_hash") -and [string]$currentSelectMarker.raw_frame_hash -eq $currentRawHash -and
    (Test-ObjectProperty $currentSelectMarker "requested_frames") -and [int]$currentSelectMarker.requested_frames -eq $SelectedFrames -and
    (Test-ObjectProperty $currentSelectMarker "selected_frames") -and [int]$currentSelectMarker.selected_frames -eq $currentSelectedCount -and
    (Test-ObjectProperty $currentSelectMarker "max_cumulative_flow") -and [double]$currentSelectMarker.max_cumulative_flow -eq $MaxCumulativeFlow -and
    (Test-ObjectProperty $currentSelectMarker "frame_quality_sha256") -and [string]$currentSelectMarker.frame_quality_sha256 -eq $currentFrameQualitySha -and
    (Test-ObjectProperty $currentSelectMarker "selected_frame_hash") -and [string]$currentSelectMarker.selected_frame_hash -eq $currentSelectedHash -and
    (Test-ObjectProperty $currentSelectMarker "scene_type") -and [string]$currentSelectMarker.scene_type -eq $SceneType -and
    (Test-ObjectProperty $currentSelectMarker "selector_version") -and [string]$currentSelectMarker.selector_version -eq $SelectorVersion -and
    (Test-ObjectProperty $currentSelectMarker "selector_script_sha256") -and [string]$currentSelectMarker.selector_script_sha256 -eq $SelectorScriptSha256

$currentSolveMarker = Read-Marker "solve"
$currentModelHashForManifest = Get-ColmapModelHash $SparsePath
$currentSolveValid = $currentSelectValid -and $currentSolveMarker -and
    (Test-ObjectProperty $currentSolveMarker "quality_gates_pass") -and [bool]$currentSolveMarker.quality_gates_pass -and
    (Test-ObjectProperty $currentSolveMarker "image_set_hash") -and [string]$currentSolveMarker.image_set_hash -eq $currentSelectedHash -and
    (Test-ObjectProperty $currentSolveMarker "shot_manifest_path") -and [string]$currentSolveMarker.shot_manifest_path -eq [string]$ResolvedShotManifestPath -and
    (Test-ObjectProperty $currentSolveMarker "shot_manifest_sha256") -and [string]$currentSolveMarker.shot_manifest_sha256 -eq [string]$ShotManifestSha256 -and
    (Test-ObjectProperty $currentSolveMarker "model_hash") -and [string]$currentSolveMarker.model_hash -eq $currentModelHashForManifest -and
    (Test-ObjectProperty $currentSolveMarker "scene_type") -and [string]$currentSolveMarker.scene_type -eq $SceneType -and
    (Test-ObjectProperty $currentSolveMarker "selector_version") -and [string]$currentSolveMarker.selector_version -eq $SelectorVersion -and
    (Test-ObjectProperty $currentSolveMarker "gate_version") -and [string]$currentSolveMarker.gate_version -eq $GateVersion -and
    (Test-ObjectProperty $currentSolveMarker "colmap_version") -and [string]$currentSolveMarker.colmap_version -eq $ColmapVersion -and
    (Test-ObjectProperty $currentSolveMarker "colmap_sha256") -and [string]$currentSolveMarker.colmap_sha256 -eq $ColmapSha256

$currentTrainingQualityPath = Join-Path $RunRoot "training_quality_report.json"
$currentTrainingQualitySha = if (Test-Path -LiteralPath $currentTrainingQualityPath -PathType Leaf) {
    (Get-FileHash -LiteralPath $currentTrainingQualityPath -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$currentTrainingQuality = if ($currentTrainingQualitySha) {
    try { Get-Content -LiteralPath $currentTrainingQualityPath -Raw | ConvertFrom-Json } catch { $null }
} else { $null }
$currentHoldoutRenderSetSha = $null
if ($Trainer -eq "Brush" -and $currentTrainingQuality -and (Test-ObjectProperty $currentTrainingQuality "render_directory") -and
    (Test-Path -LiteralPath ([string]$currentTrainingQuality.render_directory) -PathType Container)) {
    $currentHoldoutRenderSetSha = Get-FileSetHash ([string]$currentTrainingQuality.render_directory) "*.png"
}
$currentTrainMarker = Read-Marker "train"
$currentAerialDiagnosticEvidence = Get-AerialDiagnosticEvidenceState $currentTrainMarker
$currentTrainerSignatureValid = if ($Trainer -eq "Brush") {
    $currentTrainMarker -and
        (Test-ObjectProperty $currentTrainMarker "brush_version") -and [string]$currentTrainMarker.brush_version -eq $BrushVersion -and
        (Test-ObjectProperty $currentTrainMarker "brush_sha256") -and [string]$currentTrainMarker.brush_sha256 -eq $BrushSha256 -and
        (Test-ObjectProperty $currentTrainMarker "evaluator_sha256") -and [string]$currentTrainMarker.evaluator_sha256 -eq $EvaluatorSha256 -and
        (Test-ObjectProperty $currentTrainMarker "ply_verifier_sha256") -and [string]$currentTrainMarker.ply_verifier_sha256 -eq $PlyVerifierSha256 -and
        (Test-ObjectProperty $currentTrainMarker "holdout_render_set_sha256") -and [string]$currentTrainMarker.holdout_render_set_sha256 -eq $currentHoldoutRenderSetSha
} else { $false }
$currentTrainValid = $currentPlyPayloadValid -and $currentSolveValid -and $currentTrainMarker -and $currentTrainerSignatureValid -and $currentAerialDiagnosticEvidence.valid -and
    (Test-ObjectProperty $currentTrainMarker "trainer") -and [string]$currentTrainMarker.trainer -eq $Trainer -and
    (Test-ObjectProperty $currentTrainMarker "training_steps") -and [int]$currentTrainMarker.training_steps -eq $TrainingSteps -and
    (Test-ObjectProperty $currentTrainMarker "training_max_resolution") -and [int]$currentTrainMarker.training_max_resolution -eq $TrainingMaxResolution -and
    (Test-ObjectProperty $currentTrainMarker "brush_max_splats") -and [int]$currentTrainMarker.brush_max_splats -eq $BrushMaxSplats -and
    (Test-ObjectProperty $currentTrainMarker "brush_scale_loss_weight") -and [double]$currentTrainMarker.brush_scale_loss_weight -eq $BrushScaleLossWeight -and
    (Test-ObjectProperty $currentTrainMarker "seed") -and [int]$currentTrainMarker.seed -eq $TrainingSeed -and
    (Test-ObjectProperty $currentTrainMarker "eval_split_every") -and [int]$currentTrainMarker.eval_split_every -eq $EvalSplitEvery -and
    (Test-ObjectProperty $currentTrainMarker "image_set_hash") -and [string]$currentTrainMarker.image_set_hash -eq $currentSelectedHash -and
    (Test-ObjectProperty $currentTrainMarker "solve_model_hash") -and [string]$currentTrainMarker.solve_model_hash -eq $currentModelHashForManifest -and
    (Test-ObjectProperty $currentTrainMarker "final_ply_sha256") -and [string]$currentTrainMarker.final_ply_sha256 -eq $currentPlySha -and
    (Test-ObjectProperty $currentTrainMarker "quality_report_sha256") -and [string]$currentTrainMarker.quality_report_sha256 -eq $currentTrainingQualitySha
$currentPlyValid = $currentTrainValid

# Re-evaluate the final manifest through the same fail-closed chain used at each
# stage boundary. These signed states supersede the compatibility calculations
# above for markers created by older app versions.
$manifestExtractState = Get-ExtractChainState
$manifestSelectState = Get-SelectChainState $manifestExtractState
$manifestSolveState = Get-SolveChainState $manifestSelectState
$manifestTrainState = Get-TrainChainState $manifestSolveState
$currentExtractValid = [bool]$manifestExtractState.valid
$currentSelectValid = [bool]$manifestSelectState.valid
$currentSolveValid = [bool]$manifestSolveState.valid
$currentTrainValid = [bool]$manifestTrainState.valid
$currentPlyValid = $currentTrainValid
$currentSelectedCount = [int]$manifestSelectState.count
$currentSelectedHash = [string]$manifestSelectState.hash
$currentModelHashForManifest = [string]$manifestSolveState.model_hash
$currentPlySha = [string]$manifestTrainState.ply_sha256
$currentPlyReport = $manifestTrainState.ply_report
$currentTrainingQualityPath = $QualityReportPath
$currentTrainingQualitySha = [string]$manifestTrainState.quality_sha256
$currentTrainingQuality = $manifestTrainState.quality
$currentHoldoutRenderSetSha = [string]$manifestTrainState.holdout_sha256
$currentAerialDiagnosticEvidence = $manifestTrainState.aerial_diagnostic

$currentBlenderMarker = Read-Marker "blender"
$manifestHandoffScriptSha = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot "blender_handoff.py") -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestSyncScriptSha = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot "kiri_camera_sync.py") -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestValidationScriptSha = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot "blender_validate.py") -Algorithm SHA256).Hash.ToLowerInvariant()
$manifestCamerasTextSha = if (Test-Path -LiteralPath (Join-Path $ModelTextRoot "cameras.txt") -PathType Leaf) { (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "cameras.txt") -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestImagesTextSha = if (Test-Path -LiteralPath (Join-Path $ModelTextRoot "images.txt") -PathType Leaf) { (Get-FileHash -LiteralPath (Join-Path $ModelTextRoot "images.txt") -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestPoseReportSha = if (Test-Path -LiteralPath (Join-Path $RunRoot "reconstruction_report.json") -PathType Leaf) { (Get-FileHash -LiteralPath (Join-Path $RunRoot "reconstruction_report.json") -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestPlyReportSha = if (Test-Path -LiteralPath (Join-Path $RunRoot "gaussian_ply_report.json") -PathType Leaf) { (Get-FileHash -LiteralPath (Join-Path $RunRoot "gaussian_ply_report.json") -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestBlendSha = if (Test-Path -LiteralPath $BlenderFile -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderFile -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestHandoffReportSha = if (Test-Path -LiteralPath $BlenderReportPath -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderReportPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestOpenReportSha = if (Test-Path -LiteralPath $BlenderOpenReportPath -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderOpenReportPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestPreviewSha = if (Test-Path -LiteralPath $BlenderPreviewPath -PathType Leaf) { (Get-FileHash -LiteralPath $BlenderPreviewPath -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
$manifestBlenderReport = if ($manifestHandoffReportSha) { try { Get-Content -LiteralPath $BlenderReportPath -Raw | ConvertFrom-Json } catch { $null } } else { $null }
$manifestOpenReport = if ($manifestOpenReportSha) { try { Get-Content -LiteralPath $BlenderOpenReportPath -Raw | ConvertFrom-Json } catch { $null } } else { $null }
$manifestBlenderReportsValid = $manifestBlenderReport -and $manifestOpenReport -and
    [string]$manifestBlenderReport.status -eq "MECHANICAL PASS" -and
    [string]$manifestBlenderReport.visual_approval -eq "AWAITING USER APPROVAL" -and
    [bool]$manifestBlenderReport.render_proof.nonblank -and
    [string]$manifestOpenReport.status -eq "MECHANICAL PASS" -and
    [string]$manifestOpenReport.visual_approval -eq "AWAITING USER APPROVAL"
$currentBlenderValid = $currentTrainValid -and $currentBlenderMarker -and
    (Test-Path -LiteralPath $BlenderFile -PathType Leaf) -and
    (Test-Path -LiteralPath $BlenderReportPath -PathType Leaf) -and
    (Test-Path -LiteralPath $BlenderOpenReportPath -PathType Leaf) -and
    (Test-Path -LiteralPath $BlenderPreviewPath -PathType Leaf) -and $manifestBlenderReportsValid -and
    (Test-ObjectProperty $currentBlenderMarker "source_ply_sha256") -and
    [string]$currentBlenderMarker.source_ply_sha256 -eq $currentPlySha -and
    (Test-ObjectProperty $currentBlenderMarker "scene_type") -and [string]$currentBlenderMarker.scene_type -eq $SceneType -and
    (Test-ObjectProperty $currentBlenderMarker "blender_builder_version") -and [string]$currentBlenderMarker.blender_builder_version -eq $BlenderBuilderVersion -and
    (Test-ObjectProperty $currentBlenderMarker "blender_open_version") -and [string]$currentBlenderMarker.blender_open_version -eq $BlenderOpenVersion -and
    (Test-ObjectProperty $currentBlenderMarker "blender_builder_sha256") -and [string]$currentBlenderMarker.blender_builder_sha256 -eq $BlenderBuilderSha256 -and
    (Test-ObjectProperty $currentBlenderMarker "blender_open_sha256") -and [string]$currentBlenderMarker.blender_open_sha256 -eq $BlenderOpenSha256 -and
    (Test-ObjectProperty $currentBlenderMarker "handoff_script_sha256") -and [string]$currentBlenderMarker.handoff_script_sha256 -eq $manifestHandoffScriptSha -and
    (Test-ObjectProperty $currentBlenderMarker "sync_script_sha256") -and [string]$currentBlenderMarker.sync_script_sha256 -eq $manifestSyncScriptSha -and
    (Test-ObjectProperty $currentBlenderMarker "validation_script_sha256") -and [string]$currentBlenderMarker.validation_script_sha256 -eq $manifestValidationScriptSha -and
    (Test-ObjectProperty $currentBlenderMarker "cameras_text_sha256") -and [string]$currentBlenderMarker.cameras_text_sha256 -eq $manifestCamerasTextSha -and
    (Test-ObjectProperty $currentBlenderMarker "images_text_sha256") -and [string]$currentBlenderMarker.images_text_sha256 -eq $manifestImagesTextSha -and
    (Test-ObjectProperty $currentBlenderMarker "pose_report_sha256") -and [string]$currentBlenderMarker.pose_report_sha256 -eq $manifestPoseReportSha -and
    (Test-ObjectProperty $currentBlenderMarker "ply_report_sha256") -and [string]$currentBlenderMarker.ply_report_sha256 -eq $manifestPlyReportSha -and
    (Test-ObjectProperty $currentBlenderMarker "blend_file_sha256") -and [string]$currentBlenderMarker.blend_file_sha256 -eq $manifestBlendSha -and
    (Test-ObjectProperty $currentBlenderMarker "blend_file_bytes") -and [int64]$currentBlenderMarker.blend_file_bytes -eq (Get-Item -LiteralPath $BlenderFile).Length -and
    (Test-ObjectProperty $currentBlenderMarker "handoff_report_sha256") -and [string]$currentBlenderMarker.handoff_report_sha256 -eq $manifestHandoffReportSha -and
    (Test-ObjectProperty $currentBlenderMarker "blender_open_report_sha256") -and [string]$currentBlenderMarker.blender_open_report_sha256 -eq $manifestOpenReportSha -and
    (Test-ObjectProperty $currentBlenderMarker "preview_sha256") -and [string]$currentBlenderMarker.preview_sha256 -eq $manifestPreviewSha -and
    (Test-ObjectProperty $currentBlenderMarker "status") -and [string]$currentBlenderMarker.status -eq "MECHANICAL PASS" -and
    (Test-ObjectProperty $currentBlenderMarker "blender_52_status") -and [string]$currentBlenderMarker.blender_52_status -eq "MECHANICAL PASS"
$manifest = [ordered]@{
    schema_version = 3
    attempt_id = $AttemptId
    generated_utc = [DateTime]::UtcNow.ToString("o")
    run_name = $RunName
    scene_type = $SceneType
    status = if ($currentBlenderValid) { "MECHANICAL PASS; VISUAL APPROVAL REQUIRED" } elseif ($currentPlyValid) { "SPLAT READY; BLENDER HANDOFF NOT BUILT" } else { "INCOMPLETE" }
    source = [ordered]@{
        path = $VideoPath
        bytes = $SourceBytes
        sha256 = $SourceSha256
    }
    shot_manifest = if ($ResolvedShotManifestPath) {
        [ordered]@{ path = $ResolvedShotManifestPath; sha256 = $ShotManifestSha256 }
    } else { $null }
    settings = [ordered]@{
        selected_frames = $SelectedFrames
        actual_selected_images = $currentSelectedCount
        candidate_multiplier = $CandidateMultiplier
        max_cumulative_flow = $MaxCumulativeFlow
        training_steps = $TrainingSteps
        training_max_resolution = $TrainingMaxResolution
        brush_max_splats = $BrushMaxSplats
        spirula_max_splats = if ($Trainer -eq "Spirula") { $BrushMaxSplats } else { $null }
        spirula_floater_suppression = if ($Trainer -eq "Spirula") { $SpirulaFloaterSuppression } else { $null }
        spirula_distraction_robustness = if ($Trainer -eq "Spirula") { $SpirulaDistractionRobustness } else { $null }
        spirula_save_full_checkpoint = if ($Trainer -eq "Spirula") { $SpirulaSaveFullCheckpoint } else { $null }
        three_dgrut_mcmc_max_splats = if ($Trainer -eq "3DGUT-MCMC") { $ThreeDGrutMcmcMaxSplats } else { $null }
        brush_scale_loss_weight = $BrushScaleLossWeight
        trainer = $Trainer
        seed = if ($Trainer -eq "Brush") { $TrainingSeed } else { $null }
        evaluation_split_every = $EvalSplitEvery
        aerial_diagnostic = if ($RequiresAerialDiagnostic) {
            [ordered]@{
                steps = $AerialDiagnosticSteps
                growth_stop_iter = $AerialDiagnosticGrowthStopIter
                max_splats = $AerialDiagnosticMaxSplats
                max_resolution = $AerialDiagnosticMaxResolution
                evaluation_split_every = $AerialDiagnosticEvalSplitEvery
            }
        } else { $null }
        three_dgrut_aerial_stages = if ($RequiresThreeDGrutAerialStages) {
            [ordered]@{
                smoke_steps = $ThreeDGrutSmokeSteps
                smoke_max_splats = $ThreeDGrutSmokeMaxSplats
                diagnostic_steps = $ThreeDGrutDiagnosticSteps
                diagnostic_max_splats = $ThreeDGrutDiagnosticMaxSplats
            }
        } else { $null }
    }
    tools = [ordered]@{
        ffmpeg = [ordered]@{ path = $FfmpegPath; version = $FfmpegVersion; sha256 = $FfmpegSha256 }
        ffprobe = [ordered]@{ path = $FfprobePath; version = $FfprobeVersion; sha256 = $FfprobeSha256 }
        colmap = [ordered]@{ path = $ColmapPath; version = $ColmapVersion; sha256 = $ColmapSha256 }
        brush = if ($BrushPath) { [ordered]@{ path = $BrushPath; version = $BrushVersion; sha256 = $BrushSha256 } } else { $null }
        spirula = if ($SpirulaPath) { [ordered]@{ path = $SpirulaPath; version = $SpirulaVersion; sha256 = $SpirulaSha256 } } else { $null }
        aerial_sentinel_selector = if ($RequiresAerialDiagnostic) { [ordered]@{ path = $AerialSentinelSelectorPath; sha256 = $AerialSentinelSelectorSha256 } } else { $null }
        aerial_gate_evaluator = if ($RequiresAerialDiagnostic) { [ordered]@{ path = $AerialGateEvaluatorPath; sha256 = $AerialGateEvaluatorSha256 } } else { $null }
        blender_builder = if ($BlenderBuilderPath) { [ordered]@{ path = $BlenderBuilderPath; version = $BlenderBuilderVersion; sha256 = $BlenderBuilderSha256 } } else { $null }
        blender_open = if ($BlenderOpenPath) { [ordered]@{ path = $BlenderOpenPath; version = $BlenderOpenVersion; sha256 = $BlenderOpenSha256 } } else { $null }
        kiri_addon = if ($currentBlenderValid) { "3DGS Render 4.1.5 (installed add-on; not redistributed)" } else { $null }
    }
    reports = [ordered]@{
        capture = if ($currentSelectValid -and (Test-Path -LiteralPath (Join-Path $RunRoot "frame_quality.json"))) { Join-Path $RunRoot "frame_quality.json" } else { $null }
        reconstruction = if ($currentSolveValid -and (Test-Path -LiteralPath (Join-Path $RunRoot "reconstruction_report.json"))) { Join-Path $RunRoot "reconstruction_report.json" } else { $null }
        training = if ($currentTrainValid) { $currentTrainingQualityPath } else { $null }
        aerial_sentinels = if ($currentTrainValid -and $RequiresAerialDiagnostic) { [ordered]@{ path = $currentAerialDiagnosticEvidence.sentinel.path; sha256 = $currentAerialDiagnosticEvidence.sentinel.sha256 } } else { $null }
        aerial_diagnostic_holdout = if ($currentTrainValid -and $RequiresAerialDiagnostic) { [ordered]@{ path = $currentAerialDiagnosticEvidence.diagnostic_quality.path; sha256 = $currentAerialDiagnosticEvidence.diagnostic_quality.sha256 } } else { $null }
        aerial_diagnostic_ply = if ($currentTrainValid -and $RequiresAerialDiagnostic) { [ordered]@{ path = $currentAerialDiagnosticEvidence.diagnostic_ply.path; sha256 = $currentAerialDiagnosticEvidence.diagnostic_ply.sha256 } } else { $null }
        aerial_diagnostic_gate = if ($currentTrainValid -and $RequiresAerialDiagnostic) { [ordered]@{ path = $currentAerialDiagnosticEvidence.diagnostic_gate.path; sha256 = $currentAerialDiagnosticEvidence.diagnostic_gate.sha256; status = $currentAerialDiagnosticEvidence.diagnostic_status } } else { $null }
        aerial_final_gate = if ($currentTrainValid -and $RequiresAerialDiagnostic) { [ordered]@{ path = $currentAerialDiagnosticEvidence.final_gate.path; sha256 = $currentAerialDiagnosticEvidence.final_gate.sha256; status = $currentAerialDiagnosticEvidence.final_status } } else { $null }
        gaussian_ply = if ($currentPlyValid) { Join-Path $RunRoot "gaussian_ply_report.json" } else { $null }
        blender = if ($currentBlenderValid) { $BlenderReportPath } else { $null }
        blender_52_open = if ($currentBlenderValid) { $BlenderOpenReportPath } else { $null }
    }
    artifacts = [ordered]@{
        gaussian_ply = if ($currentPlyValid) { [ordered]@{ path = $FinalPly; sha256 = $currentPlySha; bytes = (Get-Item -LiteralPath $FinalPly).Length } } else { $null }
        blender_scene = if ($currentBlenderValid) { [ordered]@{ path = $BlenderFile; sha256 = (Get-FileHash -LiteralPath $BlenderFile -Algorithm SHA256).Hash.ToLowerInvariant(); bytes = (Get-Item -LiteralPath $BlenderFile).Length } } else { $null }
    }
}
$manifestTemp = "$RunManifestPath.tmp"
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestTemp -Encoding utf8
Move-Item -LiteralPath $manifestTemp -Destination $RunManifestPath -Force

Write-Host "`nRun complete: $RunRoot" -ForegroundColor Green
if ($currentPlyValid) { Write-Host "PLY: $FinalPly" -ForegroundColor Green }
if ($currentBlenderValid) { Write-Host "BLEND: $BlenderFile" -ForegroundColor Green }
$attempt["status"] = "COMPLETE"
$attempt["completed_utc"] = [DateTime]::UtcNow.ToString("o")
$attempt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
Copy-Item -LiteralPath $RunAttemptPath -Destination (Join-Path $LogsRoot "run_attempt-$AttemptId.json") -Force
} catch {
    if (Test-Path -LiteralPath (Join-Path $RunRoot "publish_transaction.json") -PathType Leaf) {
        Restore-PublishTransaction
    }
    if ($CandidatePlyCleanupPath -and (Test-Path -LiteralPath $CandidatePlyCleanupPath -PathType Leaf)) {
        Remove-Item -LiteralPath $CandidatePlyCleanupPath -Force -ErrorAction SilentlyContinue
    }
    foreach ($publishTemp in @("$FinalPly.new-$AttemptId", "$QualityReportPath.new-$AttemptId")) {
        if (Test-Path -LiteralPath $publishTemp -PathType Leaf) { Remove-Item -LiteralPath $publishTemp -Force -ErrorAction SilentlyContinue }
    }
    $attempt["status"] = "FAILED"
    $attempt["failed_utc"] = [DateTime]::UtcNow.ToString("o")
    $attempt["error"] = $_.Exception.Message
    $attempt | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$RunAttemptPath.tmp" -Encoding utf8
    Move-Item -LiteralPath "$RunAttemptPath.tmp" -Destination $RunAttemptPath -Force
    throw
}
} finally {
    Exit-SplatExclusiveLock $RunLock
    Exit-SplatExclusiveLock $GlobalLock
}
