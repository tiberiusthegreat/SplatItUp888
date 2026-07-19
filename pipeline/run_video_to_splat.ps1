[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$VideoPath,

    [string]$RunName,
    [int]$SelectedFrames = 180,
    [int]$TrainingSteps = 30000,
    [ValidateSet("Brush", "3DGUT", "3DGUT-MCMC")]
    [string]$Trainer = "Brush",
    [switch]$AdaptiveExtraction,
    [int]$CandidateMultiplier = 2,
    [int]$MaxLongSide = 0,
    [switch]$NoAutoRotate,
    [ValidateSet("extract", "select", "solve", "train", "view")]
    [string]$FromStage = "extract",
    [ValidateSet("extract", "select", "solve", "train", "view")]
    [string]$ToStage = "view",
    [string]$OutputRoot,
    [string]$ConfigPath,
    [int]$ViewerPort = 3010,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Stages = @("extract", "select", "solve", "train", "view")
$FromIndex = [array]::IndexOf($Stages, $FromStage)
$ToIndex = [array]::IndexOf($Stages, $ToStage)
if ($FromIndex -gt $ToIndex) { throw "FromStage must come before ToStage" }

function Test-Stage([string]$Name) {
    $index = [array]::IndexOf($Stages, $Name)
    return $index -ge $FromIndex -and $index -le $ToIndex
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
$SuperSplatDist = Resolve-DirectoryPath -Name "SuperSplatDist" -Candidates @(
    (Join-Path $ProjectRoot "tools\supersplat\dist")
)

if (-not $OutputRoot) {
    if ($Config.ContainsKey("OutputRoot") -and $Config.OutputRoot) {
        $OutputRoot = [string]$Config.OutputRoot
    } else {
        $OutputRoot = Join-Path $ProjectRoot "runs"
    }
}

$VideoPath = [System.IO.Path]::GetFullPath($VideoPath)
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) { throw "Video not found: $VideoPath" }
if ($SelectedFrames -lt 2) { throw "SelectedFrames must be at least 2" }
if ($TrainingSteps -lt 1) { throw "TrainingSteps must be positive" }
if ($CandidateMultiplier -lt 1) { throw "CandidateMultiplier must be at least 1" }
if ($MaxLongSide -lt 0) { throw "MaxLongSide cannot be negative" }

if (-not $RunName) {
    $RunName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath).ToLowerInvariant()
    $RunName = ($RunName -replace "[^a-z0-9]+", "-").Trim([char[]]"-")
}
if ($RunName -notmatch "^[a-zA-Z0-9][a-zA-Z0-9_-]*$") {
    throw "RunName may contain only letters, numbers, hyphens, and underscores"
}

$RunRoot = Join-Path ([System.IO.Path]::GetFullPath($OutputRoot)) $RunName
$FramesRoot = Join-Path $RunRoot "frames"
$RawFrames = Join-Path $FramesRoot "raw"
$ReconRoot = Join-Path $RunRoot "recon"
$ImagesPath = Join-Path $ReconRoot "images"
$SparseRoot = Join-Path $ReconRoot "sparse"
$SparsePath = Join-Path $SparseRoot "0"
$DatabasePath = Join-Path $ReconRoot "database.db"
$CalibratedDatabasePath = Join-Path $ReconRoot "database_global.db"
$UndistortedRoot = Join-Path $ReconRoot "undistorted"
$BrushOutput = Join-Path $RunRoot "brush"
$ThreeDGrutOutput = Join-Path $RunRoot "3dgrut"
$FinalRoot = Join-Path $RunRoot "final"
$FinalPly = Join-Path $FinalRoot "$RunName.ply"
$LogsRoot = Join-Path $RunRoot "logs"

New-Item -ItemType Directory -Force -Path $RunRoot, $FramesRoot, $ReconRoot, $LogsRoot | Out-Null
$SourcePath = Join-Path $RunRoot "source.json"
$ResolvedVideo = (Get-Item -LiteralPath $VideoPath).FullName
if (Test-Path -LiteralPath $SourcePath) {
    $existingSource = Get-Content -LiteralPath $SourcePath -Raw | ConvertFrom-Json
    if ($existingSource.video_path -ne $ResolvedVideo) {
        throw "Run '$RunName' already belongs to a different video: $($existingSource.video_path)"
    }
} else {
    [ordered]@{
        video_path = $ResolvedVideo
        run_name = $RunName
        created_utc = [DateTime]::UtcNow.ToString("o")
    } | ConvertTo-Json | Set-Content -LiteralPath $SourcePath -Encoding utf8
}

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
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Test-ObjectProperty($Object, [string]$Name) {
    return $null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name
}

function Write-Marker([string]$Name, [System.Collections.IDictionary]$Details) {
    $Details["completed_utc"] = [DateTime]::UtcNow.ToString("o")
    $Details | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunRoot ".$Name.complete.json") -Encoding utf8
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
        "$($_.Name)|$($_.Length)"
    }
    $bytes = [Text.Encoding]::UTF8.GetBytes(($manifest -join "`n"))
    $sha = [Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant() }
    finally { $sha.Dispose() }
}

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
    $canReuse = $extractMarker -and $rawCount -gt 0 -and
        (Test-ObjectProperty $extractMarker "adaptive_extraction") -and
        (Test-ObjectProperty $extractMarker "candidate_multiplier") -and
        (Test-ObjectProperty $extractMarker "max_long_side") -and
        (Test-ObjectProperty $extractMarker "auto_rotate") -and
        (Test-ObjectProperty $extractMarker "selected_frame_target") -and
        [bool]$extractMarker.adaptive_extraction -eq [bool]$AdaptiveExtraction -and
        [int]$extractMarker.candidate_multiplier -eq $CandidateMultiplier -and
        [int]$extractMarker.max_long_side -eq $MaxLongSide -and
        [bool]$extractMarker.auto_rotate -eq (-not [bool]$NoAutoRotate) -and
        [int]$extractMarker.selected_frame_target -eq $SelectedFrames

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
        $arguments.AddRange([string[]]@("-hide_banner", "-loglevel", "warning"))
        if ($NoAutoRotate) { $arguments.Add("-noautorotate") }
        $arguments.AddRange([string[]]@("-i", $VideoPath, "-map", "0:v:0"))
        if ($filters.Count -gt 0) { $arguments.AddRange([string[]]@("-vf", ($filters -join ","))) }
        $arguments.AddRange([string[]]@("-fps_mode", "passthrough", "-q:v", "1", "-an", (Join-Path $RawFrames "frame_%06d.jpg")))
        Invoke-LoggedCommand $FfmpegPath $arguments.ToArray() (Join-Path $LogsRoot "extract.log")
        $rawCount = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg").Count
        if ($rawCount -lt 2) { throw "FFmpeg extracted only $rawCount frames" }
        Write-Marker "extract" ([ordered]@{
            raw_frames = $rawCount
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
        })
        Write-Host "[extract] Decoded $rawCount frames (rotation $rotation degrees, autorotate $(-not [bool]$NoAutoRotate))" -ForegroundColor Green
    }
}

if (Test-Stage "select") {
    if (-not (Test-Path -LiteralPath $RawFrames)) { throw "Raw frames are missing; start from extract" }
    $rawCount = @(Get-ChildItem -LiteralPath $RawFrames -Filter "*.jpg").Count
    $expectedCount = [Math]::Min($SelectedFrames, $rawCount)
    $selectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    $selectMarker = Read-Marker "select"
    if ($selectMarker -and (Test-ObjectProperty $selectMarker "requested_frames") -and
        $selectedCount -eq $expectedCount -and [int]$selectMarker.requested_frames -eq $SelectedFrames) {
        Write-Host "[select] Reusing $selectedCount selected frames"
    } else {
        New-Item -ItemType Directory -Force -Path $ImagesPath | Out-Null
        Invoke-LoggedCommand $PythonPath @(
            (Join-Path $PSScriptRoot "select_frames.py"),
            "--input", $RawFrames,
            "--output", $ImagesPath,
            "--target", "$SelectedFrames",
            "--blur-percentile", "20",
            "--records", (Join-Path $RunRoot "frame_quality.json"),
            "--contact-sheet", (Join-Path $RunRoot "selected_frames_contact_sheet.jpg")
        ) (Join-Path $LogsRoot "select.log")
        $selectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg").Count
        if ($selectedCount -ne $expectedCount) { throw "Frame selector produced $selectedCount frames; expected $expectedCount" }
        Write-Marker "select" ([ordered]@{ source_frames = $rawCount; requested_frames = $SelectedFrames; selected_frames = $selectedCount })
        Write-Host "[select] Kept $selectedCount sharp, exposed, time-distributed frames" -ForegroundColor Green
    }
}

if (Test-Stage "solve") {
    $selectedCount = @(Get-ChildItem -LiteralPath $ImagesPath -Filter "*.jpg" -ErrorAction SilentlyContinue).Count
    if ($selectedCount -lt 2) { throw "Selected images are missing; start from select" }
    $imageSetHash = Get-ImageSetHash $ImagesPath
    $modelFiles = @("cameras.bin", "images.bin", "points3D.bin") | ForEach-Object { Join-Path $SparsePath $_ }
    $hasModel = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0
    $solveMarker = Read-Marker "solve"
    $canReuseModel = $hasModel -and $solveMarker -and (Test-ObjectProperty $solveMarker "image_set_hash") -and
        $solveMarker.image_set_hash -eq $imageSetHash
    if ($canReuseModel) {
        Write-Host "[solve] Reusing COLMAP global model in $SparsePath"
    } else {
        if (Test-Path -LiteralPath $DatabasePath) { Remove-Item -LiteralPath $DatabasePath -Force }
        if (Test-Path -LiteralPath $CalibratedDatabasePath) { Remove-Item -LiteralPath $CalibratedDatabasePath -Force }
        if (Test-Path -LiteralPath $SparseRoot) { Remove-Item -LiteralPath $SparseRoot -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $SparseRoot | Out-Null
        Invoke-LoggedCommand $ColmapPath @(
            "feature_extractor", "--database_path", $DatabasePath, "--image_path", $ImagesPath,
            "--ImageReader.camera_model", "SIMPLE_RADIAL", "--ImageReader.single_camera", "1",
            "--FeatureExtraction.type", "SIFT", "--SiftExtraction.max_num_features", "8192",
            "--log_path", (Join-Path $LogsRoot "colmap_feature")
        ) (Join-Path $LogsRoot "colmap_feature_console.log")
        Invoke-LoggedCommand $ColmapPath @(
            "exhaustive_matcher", "--database_path", $DatabasePath,
            "--FeatureMatching.type", "SIFT_BRUTEFORCE", "--log_path", (Join-Path $LogsRoot "colmap_match")
        ) (Join-Path $LogsRoot "colmap_match_console.log")
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
        $hasModel = @($modelFiles | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0
        if (-not $hasModel) { throw "COLMAP global mapper did not produce a complete sparse model" }
    }

    $analysisLines = & $ColmapPath "model_analyzer" "--path" $SparsePath "--log_target" "stdout" 2>&1
    $analysisText = $analysisLines -join [Environment]::NewLine
    $analysisText | Set-Content -LiteralPath (Join-Path $RunRoot "reconstruction_analysis.txt") -Encoding utf8
    function Read-Metric([string]$Label) {
        $match = [regex]::Match($analysisText, "(?i)" + [regex]::Escape($Label) + "\s*:\s*([0-9.eE+-]+)")
        if ($match.Success) { return [double]::Parse($match.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture) }
        return $null
    }
    $registered = Read-Metric "Registered images"
    if ($null -eq $registered) { $registered = Read-Metric "Images" }
    $points = Read-Metric "Points"
    $reprojectionError = Read-Metric "Mean reprojection error"
    $registrationRate = if ($selectedCount -gt 0 -and $null -ne $registered) { 100.0 * $registered / $selectedCount } else { $null }
    $registrationPass = $null -ne $registrationRate -and $registrationRate -ge 80
    $pointsPass = $null -ne $points -and $points -ge 5000
    $reprojectionPass = $null -ne $reprojectionError -and $reprojectionError -le 1.5
    $reconReport = [ordered]@{
        selected_images = $selectedCount
        registered_images = $registered
        registration_percent = if ($null -ne $registrationRate) { [Math]::Round($registrationRate, 2) } else { $null }
        points = $points
        observations = Read-Metric "Observations"
        mean_track_length = Read-Metric "Mean track length"
        mean_reprojection_error_pixels = $reprojectionError
        image_set_hash = $imageSetHash
        mapper = "COLMAP 4 global_mapper"
        quality_gates = [ordered]@{
            registration = [ordered]@{ pass = $registrationPass; minimum_percent = 80 }
            sparse_points = [ordered]@{ pass = $pointsPass; minimum = 5000 }
            reprojection_error = [ordered]@{ pass = $reprojectionPass; maximum_pixels = 1.5 }
            overall_pass = $registrationPass -and $pointsPass -and $reprojectionPass
        }
        model_path = $SparsePath
    }
    $reconReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $RunRoot "reconstruction_report.json") -Encoding utf8
    Write-Marker "solve" ([ordered]@{
        image_set_hash = $imageSetHash
        registered_images = $registered
        selected_images = $selectedCount
        registration_percent = $registrationRate
        points = $points
        reprojection_error_pixels = $reprojectionError
        quality_gates_pass = $registrationPass -and $pointsPass -and $reprojectionPass
    })
    $qualityText = "registered $registered/$selectedCount ($([Math]::Round($registrationRate, 1))%), points $points, reprojection $reprojectionError px"
    if ($registrationPass -and $pointsPass -and $reprojectionPass) {
        Write-Host "[solve] Quality gates passed: $qualityText" -ForegroundColor Green
    } else {
        Write-Warning "Reconstruction quality warning: $qualityText"
    }
}

if (Test-Stage "train") {
    foreach ($required in @("cameras.bin", "images.bin", "points3D.bin")) {
        if (-not (Test-Path -LiteralPath (Join-Path $SparsePath $required))) { throw "COLMAP model is incomplete; start from solve" }
    }
    $trainMarker = Read-Marker "train"
    $canReuseTraining = (Test-Path -LiteralPath $FinalPly -PathType Leaf) -and $trainMarker -and
        (Test-ObjectProperty $trainMarker "trainer") -and (Test-ObjectProperty $trainMarker "training_steps") -and
        [string]$trainMarker.trainer -eq $Trainer -and [int]$trainMarker.training_steps -eq $TrainingSteps
    if ($canReuseTraining) {
        Write-Host "[train] Reusing $Trainer result at $TrainingSteps steps"
    } elseif ($Trainer -eq "Brush") {
        New-Item -ItemType Directory -Force -Path $BrushOutput, $FinalRoot | Out-Null
        $trainingStarted = Get-Date
        Invoke-LoggedCommand $BrushPath @(
            "--total-steps", "$TrainingSteps", "--export-every", "5000",
            "--export-path", $BrushOutput, "--export-name", "${RunName}_brush_{iter}.ply", $ReconRoot
        ) (Join-Path $LogsRoot "brush_train.log")
        $latestPly = Get-ChildItem -LiteralPath $BrushOutput -Filter "*.ply" |
            Where-Object { $_.LastWriteTime -ge $trainingStarted.AddSeconds(-5) } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $latestPly) { throw "Brush did not export a new PLY" }
        Copy-Item -LiteralPath $latestPly.FullName -Destination $FinalPly -Force
    } else {
        if (-not $Config.ContainsKey("ThreeDGRUT")) {
            throw "3DGRUT is not configured. Add ThreeDGRUT.Repo and ThreeDGRUT.Python to splatitup.local.psd1."
        }
        $ThreeDGrutRepo = [string]$Config.ThreeDGRUT.Repo
        $ThreeDGrutPython = [string]$Config.ThreeDGRUT.Python
        if (-not (Test-Path -LiteralPath (Join-Path $ThreeDGrutRepo "train.py") -PathType Leaf)) { throw "3DGRUT train.py not found in $ThreeDGrutRepo" }
        if (-not (Test-Path -LiteralPath $ThreeDGrutPython -PathType Leaf)) { throw "3DGRUT Python not found: $ThreeDGrutPython" }
        if (-not (Test-Path -LiteralPath (Join-Path $UndistortedRoot "images") -PathType Container)) {
            New-Item -ItemType Directory -Force -Path $UndistortedRoot | Out-Null
            Invoke-LoggedCommand $ColmapPath @(
                "image_undistorter", "--image_path", $ImagesPath, "--input_path", $SparsePath,
                "--output_path", $UndistortedRoot, "--output_type", "COLMAP"
            ) (Join-Path $LogsRoot "colmap_undistort.log")
            $undistortedSparse = Join-Path $UndistortedRoot "sparse"
            $undistortedSparseZero = Join-Path $undistortedSparse "0"
            if ((Test-Path -LiteralPath (Join-Path $undistortedSparse "cameras.bin")) -and -not (Test-Path -LiteralPath $undistortedSparseZero)) {
                New-Item -ItemType Directory -Force -Path $undistortedSparseZero | Out-Null
                Copy-Item -LiteralPath (Join-Path $undistortedSparse "cameras.bin"),(Join-Path $undistortedSparse "images.bin"),(Join-Path $undistortedSparse "points3D.bin") -Destination $undistortedSparseZero
            }
        }
        New-Item -ItemType Directory -Force -Path $ThreeDGrutOutput, $FinalRoot | Out-Null
        $modelConfig = if ($Trainer -eq "3DGUT-MCMC") { "apps/colmap_3dgut_mcmc.yaml" } else { "apps/colmap_3dgut.yaml" }
        Invoke-LoggedCommand $ThreeDGrutPython @(
            "train.py", "--config-name", $modelConfig, "path=$UndistortedRoot",
            "out_dir=$ThreeDGrutOutput", "experiment_name=$RunName", "n_iterations=$TrainingSteps",
            "with_gui=false", "with_viser_gui=false", "log_frequency=500", "val_frequency=999999",
            "export_ply.enabled=true", "export_ply.path=$FinalPly", "checkpoint.iterations=[$TrainingSteps]",
            "hydra.run.dir=$(Join-Path $ThreeDGrutOutput 'hydra')"
        ) (Join-Path $LogsRoot "3dgrut_train.log") $ThreeDGrutRepo
        if (-not (Test-Path -LiteralPath $FinalPly -PathType Leaf)) { throw "3DGRUT completed without creating $FinalPly" }
    }

    Invoke-LoggedCommand $PythonPath @(
        (Join-Path $PSScriptRoot "verify_gaussian_ply.py"), $FinalPly,
        "--json", (Join-Path $RunRoot "gaussian_ply_report.json")
    ) (Join-Path $LogsRoot "verify_ply.log")
    Write-Marker "train" ([ordered]@{ final_ply = $FinalPly; training_steps = $TrainingSteps; trainer = $Trainer })
    Write-Host "[train] $Trainer Gaussian PLY ready: $FinalPly" -ForegroundColor Green
}

if (Test-Stage "view") {
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

Write-Host "`nRun complete: $RunRoot" -ForegroundColor Green
if (Test-Path -LiteralPath $FinalPly) { Write-Host "PLY: $FinalPly" -ForegroundColor Green }
