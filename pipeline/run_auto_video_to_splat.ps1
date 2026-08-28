[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$VideoPath,

    [string]$RunName,
    [ValidateSet("Preview", "Final", "Custom")]
    [string]$AutoPreset = "Custom",
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
    [ValidateSet("train", "blender", "view")]
    [string]$ToStage = "view",
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
$CoreRunnerPath = Join-Path $PSScriptRoot "run_video_to_splat.ps1"
$AutoProfilePath = Join-Path $PSScriptRoot "auto_profile.py"
$ThreeDGrutRuntimeHelperPath = Join-Path $PSScriptRoot "three_dgrut_runtime.ps1"
. $ThreeDGrutRuntimeHelperPath
$OrchestratorVersion = "1.0-bounded-pilot"
$PilotProfile = "AerialExterior"
$PilotSelectedFrames = 180
$PilotCandidateMultiplier = 2
$PilotMaxLongSide = 1280
$AcceptedShotDetectorVersion = "1.1-temporal-nms"
$AcceptedClassifierVersion = "1.0-fail-closed-pose-pilot"
$PilotGatePolicyVersion = "1.0-least-restrictive-production-gates"
$ManualProfiles = @("Object", "Walkthrough", "House", "AerialExterior")
$RequestedTrainer = $Trainer
$ResolvedTrainer = if ($RequestedTrainer -eq "Spirula") { "Spirula" } else { "Brush" }
$TrainerResolutionReason = if ($ResolvedTrainer -eq "Spirula") {
    "Auto retained the requested Spirula backend; its output must pass the same reconstruction, holdout, PLY, and provenance gates."
} else {
    "Auto uses Brush unless Spirula is explicitly selected; 3DGRUT-MCMC is not approved for automatic training."
}

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $ProjectRoot "splatitup.local.psd1"
}
$ConfigPath = [System.IO.Path]::GetFullPath($ConfigPath)
$Config = @{}
if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    $Config = Import-PowerShellDataFile -LiteralPath $ConfigPath
}
if ($ResolvedTrainer -notin @("Brush", "Spirula")) {
    if (-not $Config.ContainsKey("ThreeDGRUT")) {
        throw "3DGRUT is not configured. Add ThreeDGRUT.Repo and ThreeDGRUT.Python before the bounded pilot starts."
    }
    $threeDGrutRepo = [string]$Config.ThreeDGRUT.Repo
    $threeDGrutPython = [string]$Config.ThreeDGRUT.Python
    $threeDGrutConfigName = if ($ResolvedTrainer -eq "3DGUT-MCMC") { "colmap_3dgut_mcmc.yaml" } else { "colmap_3dgut.yaml" }
    $threeDGrutConfig = if ($threeDGrutRepo) { Join-Path $threeDGrutRepo "configs\apps\$threeDGrutConfigName" } else { $null }
    if (-not $threeDGrutRepo -or -not (Test-Path -LiteralPath (Join-Path $threeDGrutRepo "train.py") -PathType Leaf) -or
        -not $threeDGrutPython -or -not (Test-Path -LiteralPath $threeDGrutPython -PathType Leaf) -or
        -not $threeDGrutConfig -or -not (Test-Path -LiteralPath $threeDGrutConfig -PathType Leaf)) {
        throw "3DGRUT official repo, Python, or configs/apps entrypoint is incomplete; the auto pilot was not started."
    }
    $threeDGrutRuntime = Initialize-ThreeDGrutRuntime -ThreeDGrutConfig $Config.ThreeDGRUT
    $threeDGrutRepo = [string]$threeDGrutRuntime.Repo
    $threeDGrutPython = [string]$threeDGrutRuntime.Python
}
if (-not $OutputRoot) {
    if ($Config.ContainsKey("OutputRoot") -and $Config.OutputRoot) {
        $OutputRoot = [string]$Config.OutputRoot
    } else {
        $OutputRoot = Join-Path $ProjectRoot "runs"
    }
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$VideoPath = [System.IO.Path]::GetFullPath($VideoPath)
if (-not (Test-Path -LiteralPath $VideoPath -PathType Leaf)) {
    throw "Video not found: $VideoPath"
}
if ($SelectedFrames -lt 2) { throw "SelectedFrames must be at least 2" }
if ($TrainingSteps -lt 1) { throw "TrainingSteps must be positive" }
if ($CandidateMultiplier -lt 1) { throw "CandidateMultiplier must be at least 1" }
if ($MaxCumulativeFlow -lt 0.0) { throw "MaxCumulativeFlow cannot be negative" }
if ($MaxLongSide -lt 0) { throw "MaxLongSide cannot be negative" }
if ($TrainingMaxResolution -lt 320) { throw "TrainingMaxResolution must be at least 320" }
if ($BrushMaxSplats -lt 10000) { throw "BrushMaxSplats must be at least 10000" }
if ([double]::IsNaN($BrushScaleLossWeight) -or [double]::IsInfinity($BrushScaleLossWeight) -or $BrushScaleLossWeight -lt 0.0) {
    throw "BrushScaleLossWeight must be a finite non-negative number"
}
if ($EvalSplitEvery -lt 2) { throw "EvalSplitEvery must be at least 2" }
if (-not $RunName) {
    $RunName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath).ToLowerInvariant()
    $RunName = ($RunName -replace "[^a-z0-9]+", "-").Trim([char[]]"-")
}
if ($RunName -notmatch "^[a-zA-Z0-9][a-zA-Z0-9_-]*$") {
    throw "RunName may contain only letters, numbers, hyphens, and underscores"
}

$PilotRunName = "$RunName-auto-pilot"
$PilotRoot = Join-Path $OutputRoot $PilotRunName
$FinalRoot = Join-Path $OutputRoot $RunName
$AttemptId = [Guid]::NewGuid().ToString("N")
$AttemptStartedUtc = [DateTime]::UtcNow.ToString("o")
$TerminalDecisionPath = Join-Path $FinalRoot "auto_decision.json"
$LegacyPilotDecisionPath = Join-Path $PilotRoot "auto_decision.json"
$ClassifierDecisionPath = Join-Path $PilotRoot "auto_decision.$AttemptId.classifier.json"
$ClassifierReportPath = Join-Path $PilotRoot "reconstruction_report.auto-classifier.json"

foreach ($staleDecisionPath in @($TerminalDecisionPath, $LegacyPilotDecisionPath)) {
    if (Test-Path -LiteralPath $staleDecisionPath -PathType Leaf) {
        Remove-Item -LiteralPath $staleDecisionPath -Force
    }
}

function Test-JsonProperty($Object, [string]$Name) {
    return $null -ne $Object -and $Object.PSObject.Properties.Name -contains $Name
}

function Write-JsonAtomic([string]$Path, $Value) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    $temporary = "$Path.tmp"
    $Value | ConvertTo-Json -Depth 32 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Get-PilotSourceSha256 {
    $sourcePath = Join-Path $PilotRoot "source.json"
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) { return $null }
    try {
        $source = Get-Content -LiteralPath $sourcePath -Raw | ConvertFrom-Json
        if (Test-JsonProperty $source "source_sha256" -and [string]$source.source_sha256 -match "^[0-9a-fA-F]{64}$") {
            return ([string]$source.source_sha256).ToLowerInvariant()
        }
    } catch {}
    return $null
}

function Write-AutoStop([string]$Code, [string]$Reason, [string]$Stage) {
    $decision = [ordered]@{
        schema_version = 1
        orchestrator_version = $OrchestratorVersion
        classifier_version = $null
        decision_scope = "profile_selection"
        attempt_id = $AttemptId
        attempt_started_utc = $AttemptStartedUtc
        status = "STOP"
        code = $Code
        profile = $null
        reason = $Reason
        evidence = [ordered]@{
            stage = $Stage
            pilot_run_root = $PilotRoot
            video_path = $VideoPath
            source_sha256 = Get-PilotSourceSha256
            auto_preset = $AutoPreset
            requested_selected_frames = $SelectedFrames
            requested_training_steps = $TrainingSteps
            requested_trainer = $RequestedTrainer
            resolved_trainer = $ResolvedTrainer
            trainer_resolution_reason = $TrainerResolutionReason
        }
    }
    Write-JsonAtomic $TerminalDecisionPath $decision
    Write-Warning "$Code`: $Reason"
}

function Get-RequiredJsonObject($Object, [string]$Name, [string]$Label) {
    if (-not (Test-JsonProperty $Object $Name) -or $null -eq $Object.$Name -or
        $Object.$Name -is [string] -or $Object.$Name -is [array]) {
        throw "$Label.$Name must be a JSON object"
    }
    return $Object.$Name
}

function Get-StrictInteger($Value, [string]$Label) {
    if ($Value -is [bool] -or -not ($Value -is [int] -or $Value -is [long])) {
        throw "$Label must be an integer"
    }
    return [long]$Value
}

function Get-FiniteNumber($Value, [string]$Label) {
    if ($null -eq $Value -or $Value -is [bool] -or -not (
        $Value -is [System.Byte] -or $Value -is [System.SByte] -or
        $Value -is [System.Int16] -or $Value -is [System.UInt16] -or
        $Value -is [System.Int32] -or $Value -is [System.UInt32] -or
        $Value -is [System.Int64] -or $Value -is [System.UInt64] -or
        $Value -is [System.Single] -or $Value -is [System.Double] -or
        $Value -is [System.Decimal]
    )) {
        throw "$Label must be numeric"
    }
    try { $number = [double]$Value } catch { throw "$Label must be numeric" }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) {
        throw "$Label must be finite"
    }
    return $number
}

function Get-StrictBoolean($Object, [string]$Name, [string]$Label) {
    if (-not (Test-JsonProperty $Object $Name) -or $Object.$Name -isnot [bool]) {
        throw "$Label.$Name must be a Boolean"
    }
    return [bool]$Object.$Name
}

function New-ClassifierPilotReport {
    $sourceReportPath = Join-Path $PilotRoot "reconstruction_report.json"
    if (-not (Test-Path -LiteralPath $sourceReportPath -PathType Leaf)) {
        throw "The pose pilot did not write reconstruction_report.json"
    }
    $sourceReport = Get-Content -LiteralPath $sourceReportPath -Raw | ConvertFrom-Json
    if ([string]$sourceReport.profile -ne $PilotProfile) {
        throw "The pose pilot reconstruction profile is not $PilotProfile"
    }
    $report = $sourceReport | ConvertTo-Json -Depth 32 | ConvertFrom-Json
    $gates = Get-RequiredJsonObject $report "quality_gates" "reconstruction_report"
    foreach ($gateName in @(
        "model_integrity", "registration", "sparse_points", "mean_track_length",
        "reprojection_error", "verified_match_graph", "maximum_missing_run",
        "capture_loop_closure", "trajectory_continuity"
    )) {
        $gate = Get-RequiredJsonObject $gates $gateName "quality_gates"
        [void](Get-StrictBoolean $gate "pass" "quality_gates.$gateName")
    }

    $registration = $gates.registration
    $registrationActual = Get-FiniteNumber $report.registration_percent "reconstruction_report.registration_percent"
    $registration.pass = $registrationActual -ge 90.0
    $registration.minimum_percent = 90.0

    $points = Get-StrictInteger $report.points "reconstruction_report.points"
    $gates.sparse_points.pass = $points -ge 5000
    $gates.sparse_points.actual = $points
    $gates.sparse_points.minimum = 5000

    $missing = $gates.maximum_missing_run
    $missingActual = Get-StrictInteger $missing.actual "quality_gates.maximum_missing_run.actual"
    $missing.maximum = 10
    $missing.pass = $missingActual -le 10
    if (Test-JsonProperty $missing "shots" -and $null -ne $missing.shots) {
        $shotsPass = $true
        foreach ($shotProperty in $missing.shots.PSObject.Properties) {
            $shot = $shotProperty.Value
            $shotActual = Get-StrictInteger $shot.actual "quality_gates.maximum_missing_run.shots.$($shotProperty.Name).actual"
            $shot.maximum = 10
            $shot.pass = $shotActual -le 10
            if (-not $shot.pass) { $shotsPass = $false }
        }
        $missing.pass = $shotsPass
    }

    $closure = $gates.capture_loop_closure
    $closureApplicable = Get-StrictBoolean $closure "applicable" "quality_gates.capture_loop_closure"
    $closureEdges = Get-StrictInteger $closure.verified_edges "quality_gates.capture_loop_closure.verified_edges"
    $pathRatioValid = $null -ne $closure.first_to_last_over_path
    $stepRatioValid = $null -ne $closure.first_to_last_over_median_step
    $pathRatio = if ($pathRatioValid) {
        Get-FiniteNumber $closure.first_to_last_over_path "quality_gates.capture_loop_closure.first_to_last_over_path"
    } else { $null }
    $stepRatio = if ($stepRatioValid) {
        Get-FiniteNumber $closure.first_to_last_over_median_step "quality_gates.capture_loop_closure.first_to_last_over_median_step"
    } else { $null }
    $closure.pass = [bool](
        $closureApplicable -and $closureEdges -gt 0 -and $pathRatioValid -and $pathRatio -le 0.15 -and
        $stepRatioValid -and $stepRatio -le 30.0
    )
    $closure.blocking = $false
    $closure.maximum_first_to_last_over_path = 0.15
    $closure.maximum_first_to_last_over_median_step = 30.0

    $overallPass = $true
    foreach ($gateProperty in $gates.PSObject.Properties) {
        if ($gateProperty.Name -eq "overall_pass") { continue }
        $gate = $gateProperty.Value
        $gatePass = Get-StrictBoolean $gate "pass" "quality_gates.$($gateProperty.Name)"
        $blocking = if (Test-JsonProperty $gate "blocking") {
            Get-StrictBoolean $gate "blocking" "quality_gates.$($gateProperty.Name)"
        } else { $true }
        if ($blocking -and -not $gatePass) { $overallPass = $false }
    }
    $gates.overall_pass = $overallPass
    $report | Add-Member -NotePropertyName "auto_pilot_gate_policy" -NotePropertyValue ([ordered]@{
        version = $PilotGatePolicyVersion
        source_profile = $PilotProfile
        minimum_registration_percent = 90.0
        minimum_sparse_points = 5000
        maximum_missing_run = 10
        maximum_loop_steps = 30.0
        final_profile_revalidation_required = $true
        source_reconstruction_report_sha256 = (Get-FileHash -LiteralPath $sourceReportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    }) -Force
    Write-JsonAtomic $ClassifierReportPath $report
    return (Get-Item -LiteralPath $ClassifierReportPath).FullName
}

function Assert-ClassifierDecision($Decision, [string]$ClassifierReport) {
    if ((Get-StrictInteger $Decision.schema_version "auto_decision.schema_version") -ne 1) {
        throw "auto_decision.schema_version is not supported"
    }
    if ([string]$Decision.classifier_version -ne $AcceptedClassifierVersion) {
        throw "auto_decision.classifier_version is not supported"
    }
    if ([string]$Decision.status -notin @("PASS", "STOP")) {
        throw "auto_decision.status must be PASS or STOP"
    }
    if (-not (Test-JsonProperty $Decision "code") -or -not [string]$Decision.code -or
        -not (Test-JsonProperty $Decision "reason") -or -not [string]$Decision.reason) {
        throw "auto_decision code and reason are required"
    }
    $evidence = Get-RequiredJsonObject $Decision "evidence" "auto_decision"
    if ([string]$Decision.status -eq "STOP") {
        if (-not (Test-JsonProperty $Decision "profile") -or $null -ne $Decision.profile) {
            throw "A STOP decision cannot select a profile"
        }
        return
    }
    if ([string]$Decision.code -ne "AUTO_PROFILE_SELECTED" -or
        [string]$Decision.profile -notin $ManualProfiles) {
        throw "A PASS decision must select one supported manual profile"
    }

    $report = Get-Content -LiteralPath $ClassifierReport -Raw | ConvertFrom-Json
    foreach ($requiredEvidence in @(
        "pilot_profile", "selected_images", "registered_images", "sparse_points",
        "model_text_sha256", "capture_preflight_pass", "pose_gates_pass",
        "verified_match_graph_pass", "trajectory_continuity_pass", "shot_count",
        "view_convergence"
    )) {
        if (-not (Test-JsonProperty $evidence $requiredEvidence)) {
            throw "auto_decision.evidence.$requiredEvidence is required for PASS"
        }
    }
    if ([string]$evidence.pilot_profile -ne $PilotProfile -or
        (Get-StrictInteger $evidence.selected_images "auto_decision.evidence.selected_images") -ne
            (Get-StrictInteger $report.selected_images "classifier_report.selected_images") -or
        (Get-StrictInteger $evidence.registered_images "auto_decision.evidence.registered_images") -ne
            (Get-StrictInteger $report.registered_images "classifier_report.registered_images") -or
        (Get-StrictInteger $evidence.sparse_points "auto_decision.evidence.sparse_points") -ne
            (Get-StrictInteger $report.points "classifier_report.points")) {
        throw "The PASS decision does not match the current pilot counts or profile"
    }
    foreach ($flagName in @(
        "capture_preflight_pass", "pose_gates_pass", "verified_match_graph_pass",
        "trajectory_continuity_pass"
    )) {
        if (-not (Get-StrictBoolean $evidence $flagName "auto_decision.evidence")) {
            throw "auto_decision.evidence.$flagName must be true for PASS"
        }
    }
    [void](Get-RequiredJsonObject $evidence "view_convergence" "auto_decision.evidence")
    $modelHashes = Get-RequiredJsonObject $evidence "model_text_sha256" "auto_decision.evidence"
    $imagesPath = Join-Path $PilotRoot "recon\sparse_txt\images.txt"
    $pointsPath = Join-Path $PilotRoot "recon\sparse_txt\points3D.txt"
    $actualImagesSha = (Get-FileHash -LiteralPath $imagesPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $actualPointsSha = (Get-FileHash -LiteralPath $pointsPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ([string]$modelHashes.images -ne $actualImagesSha -or [string]$modelHashes.points3d -ne $actualPointsSha) {
        throw "The PASS decision model hashes do not match the current pilot"
    }
    $shotManifest = Get-RequiredJsonObject $report "shot_manifest" "classifier_report"
    if ($shotManifest.shot_ids -isnot [array] -or
        (Get-StrictInteger $evidence.shot_count "auto_decision.evidence.shot_count") -ne @($shotManifest.shot_ids).Count) {
        throw "The PASS decision shot count does not match the current pilot"
    }
}

function Resolve-AutoRunSettings([string]$Profile) {
    $resolvedFrames = $SelectedFrames
    $resolvedSteps = $TrainingSteps
    if ($AutoPreset -eq "Preview") {
        $resolvedFrames = if ($Profile -eq "Object") { 150 } else { 300 }
        $resolvedSteps = if ($Profile -eq "Object") { 7000 } else { 10000 }
    } elseif ($AutoPreset -eq "Final") {
        $resolvedFrames = if ($Profile -eq "Object") { 300 } else { 1200 }
        $resolvedSteps = if ($Profile -eq "Object") { 30000 } else { 40000 }
    }
    return [pscustomobject][ordered]@{
        auto_preset = $AutoPreset
        selected_profile = $Profile
        requested_selected_frames = $SelectedFrames
        requested_training_steps = $TrainingSteps
        resolved_selected_frames = $resolvedFrames
        resolved_training_steps = $resolvedSteps
        requested_trainer = $RequestedTrainer
        resolved_trainer = $ResolvedTrainer
        trainer_resolution_reason = $TrainerResolutionReason
    }
}

function Write-ClassifierTerminalDecision($Decision, [string]$ClassifierReport, $ResolvedRun) {
    $modelTextRoot = Join-Path $PilotRoot "recon\sparse_txt"
    $terminalEvidence = $Decision.evidence | ConvertTo-Json -Depth 32 | ConvertFrom-Json
    if ($null -ne $ResolvedRun) {
        $terminalEvidence | Add-Member -NotePropertyName "resolved_run" -NotePropertyValue $ResolvedRun -Force
    }
    $terminal = [ordered]@{
        schema_version = 1
        orchestrator_version = $OrchestratorVersion
        classifier_version = [string]$Decision.classifier_version
        decision_scope = "profile_selection"
        attempt_id = $AttemptId
        attempt_started_utc = $AttemptStartedUtc
        status = [string]$Decision.status
        code = [string]$Decision.code
        profile = $Decision.profile
        reason = [string]$Decision.reason
        evidence = $terminalEvidence
        binding = [ordered]@{
            video_path = $VideoPath
            source_sha256 = Get-PilotSourceSha256
            pilot_run_root = $PilotRoot
            classifier_report_path = $ClassifierReport
            classifier_report_sha256 = (Get-FileHash -LiteralPath $ClassifierReport -Algorithm SHA256).Hash.ToLowerInvariant()
            images_text_sha256 = (Get-FileHash -LiteralPath (Join-Path $modelTextRoot "images.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
            points3d_text_sha256 = (Get-FileHash -LiteralPath (Join-Path $modelTextRoot "points3D.txt") -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    Write-JsonAtomic $TerminalDecisionPath $terminal
}

function Resolve-PythonPath {
    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($Config.ContainsKey("Tools") -and $Config.Tools.ContainsKey("Python") -and $Config.Tools.Python) {
        $candidates.Add([string]$Config.Tools.Python)
    }
    $environmentValue = [Environment]::GetEnvironmentVariable("SPLATITUP_PYTHON")
    if ($environmentValue) { $candidates.Add($environmentValue) }
    $candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Get-Item -LiteralPath $candidate).FullName
        }
    }
    $command = Get-Command "python.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { return $command.Source }
    throw "Required tool 'Python' was not found. Configure Tools.Python in splatitup.local.psd1."
}

function New-SignedShotManifest([string]$RunRoot) {
    $sourcePath = Join-Path $RunRoot "source.json"
    $extractMarkerPath = Join-Path $RunRoot ".extract.complete.json"
    $selectMarkerPath = Join-Path $RunRoot ".select.complete.json"
    $frameQualityPath = Join-Path $RunRoot "frame_quality.json"
    foreach ($requiredPath in @($sourcePath, $extractMarkerPath, $selectMarkerPath, $frameQualityPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Auto shot evidence is missing: $requiredPath"
        }
    }

    $source = Get-Content -LiteralPath $sourcePath -Raw | ConvertFrom-Json
    $extractMarker = Get-Content -LiteralPath $extractMarkerPath -Raw | ConvertFrom-Json
    $selectMarker = Get-Content -LiteralPath $selectMarkerPath -Raw | ConvertFrom-Json
    $frameQuality = Get-Content -LiteralPath $frameQualityPath -Raw | ConvertFrom-Json
    if (-not (Test-JsonProperty $frameQuality "shot_manifest") -or $null -eq $frameQuality.shot_manifest) {
        throw "frame_quality.json does not contain deterministic shot evidence"
    }
    $embedded = $frameQuality.shot_manifest
    foreach ($requiredProperty in @(
        "schema_version", "kind", "detector_version", "candidate_frame_count",
        "candidate_frame_names_sha256", "selected_frame_count",
        "selected_frame_names_sha256", "shot_ids", "hard_cuts", "shots"
    )) {
        if (-not (Test-JsonProperty $embedded $requiredProperty)) {
            throw "Embedded shot manifest property '$requiredProperty' is missing"
        }
    }
    if ([int]$embedded.schema_version -ne 1 -or [string]$embedded.kind -ne "stitched_video_shots") {
        throw "Embedded shot manifest schema is not supported"
    }
    if (-not $AcceptedShotDetectorVersion -or
        [string]$embedded.detector_version -ne $AcceptedShotDetectorVersion) {
        throw "Shot detector '$($embedded.detector_version)' has not passed the signed golden validation"
    }
    if ([int]$embedded.candidate_frame_count -ne [int]$frameQuality.source_frames -or
        [int]$embedded.candidate_frame_count -ne [int]$extractMarker.raw_frames) {
        throw "Embedded shot manifest does not match the extracted candidate frame count"
    }
    if ([int]$embedded.selected_frame_count -ne [int]$frameQuality.selected_frames -or
        [int]$embedded.selected_frame_count -ne [int]$selectMarker.selected_frames) {
        throw "Embedded shot manifest does not match the selected frame count"
    }
    if ([string]$source.source_sha256 -notmatch "^[0-9a-fA-F]{64}$" -or
        [string]$selectMarker.selected_frame_hash -notmatch "^[0-9a-fA-F]{64}$") {
        throw "Current source or selected-image artifact hash is invalid"
    }

    $standalone = $embedded | ConvertTo-Json -Depth 32 | ConvertFrom-Json
    $standalone | Add-Member -NotePropertyName "source_sha256" -NotePropertyValue ([string]$source.source_sha256).ToLowerInvariant() -Force
    $standalone | Add-Member -NotePropertyName "selected_image_set_sha256" -NotePropertyValue ([string]$selectMarker.selected_frame_hash).ToLowerInvariant() -Force
    $manifestPath = Join-Path $RunRoot "shot_manifest.auto-signed.json"
    Write-JsonAtomic $manifestPath $standalone
    return (Get-Item -LiteralPath $manifestPath).FullName
}

function New-CommonRunnerParameters(
    [string]$Profile,
    [string]$Name,
    [int]$FrameTarget,
    [int]$StepTarget,
    [int]$LongSide,
    [int]$Multiplier,
    [double]$CumulativeFlow,
    [string]$TrainerName,
    [bool]$UseAdaptiveExtraction
) {
    $parameters = @{
        VideoPath = $VideoPath
        RunName = $Name
        SceneType = $Profile
        SelectedFrames = $FrameTarget
        TrainingSteps = $StepTarget
        Trainer = $TrainerName
        CandidateMultiplier = $Multiplier
        MaxCumulativeFlow = $CumulativeFlow
        MaxLongSide = $LongSide
        TrainingMaxResolution = $TrainingMaxResolution
        BrushMaxSplats = $BrushMaxSplats
        BrushScaleLossWeight = $BrushScaleLossWeight
        EvalSplitEvery = $EvalSplitEvery
        OutputRoot = $OutputRoot
        ConfigPath = $ConfigPath
        ViewerPort = $ViewerPort
    }
    if ($UseAdaptiveExtraction) { $parameters["AdaptiveExtraction"] = $true }
    if ($NoAutoRotate) { $parameters["NoAutoRotate"] = $true }
    return $parameters
}

function Invoke-CoreRunner([hashtable]$Parameters) {
    & $CoreRunnerPath @Parameters
}

$pilotSelectParameters = New-CommonRunnerParameters $PilotProfile $PilotRunName $PilotSelectedFrames 1 $PilotMaxLongSide $PilotCandidateMultiplier 0.0 "Brush" $true
$pilotSelectParameters["FromStage"] = "extract"
$pilotSelectParameters["ToStage"] = "select"
$pilotSelectParameters["NoBlender"] = $true
$pilotSelectParameters["NoBrowser"] = $true
try {
    Invoke-CoreRunner $pilotSelectParameters
} catch {
    Write-AutoStop "AUTO_CAPTURE_PILOT_FAILED" $_.Exception.Message "select"
    exit 2
}

try {
    $pilotShotManifestPath = New-SignedShotManifest $PilotRoot
} catch {
    Write-AutoStop "AUTO_SHOT_EVIDENCE_INVALID" $_.Exception.Message "shot_manifest"
    exit 2
}

$pilotSolveParameters = New-CommonRunnerParameters $PilotProfile $PilotRunName $PilotSelectedFrames 1 $PilotMaxLongSide $PilotCandidateMultiplier 0.0 "Brush" $true
$pilotSolveParameters["FromStage"] = "solve"
$pilotSolveParameters["ToStage"] = "solve"
$pilotSolveParameters["ShotManifestPath"] = $pilotShotManifestPath
$pilotSolveParameters["NoBlender"] = $true
$pilotSolveParameters["NoBrowser"] = $true
try {
    Invoke-CoreRunner $pilotSolveParameters
} catch {
    Write-AutoStop "AUTO_POSE_PILOT_FAILED" $_.Exception.Message "solve"
    exit 2
}

try {
    $classifierReport = New-ClassifierPilotReport
} catch {
    Write-AutoStop "AUTO_PILOT_GATE_EVIDENCE_INVALID" $_.Exception.Message "classify-gates"
    exit 2
}

try {
    $PythonPath = Resolve-PythonPath
} catch {
    Write-AutoStop "AUTO_CLASSIFIER_FAILED" $_.Exception.Message "classify"
    exit 2
}
$pilotModelText = Join-Path $PilotRoot "recon\sparse_txt"
try {
    & $PythonPath $AutoProfilePath `
        "--video-probe" (Join-Path $PilotRoot "video_probe.json") `
        "--frame-quality" (Join-Path $PilotRoot "frame_quality.json") `
        "--reconstruction-report" $classifierReport `
        "--images-text" (Join-Path $pilotModelText "images.txt") `
        "--points3d-text" (Join-Path $pilotModelText "points3D.txt") `
        "--json" $ClassifierDecisionPath
    $classifierExitCode = $LASTEXITCODE
} catch {
    Write-AutoStop "AUTO_CLASSIFIER_FAILED" $_.Exception.Message "classify"
    exit 2
}
if (-not (Test-Path -LiteralPath $ClassifierDecisionPath -PathType Leaf)) {
    Write-AutoStop "AUTO_CLASSIFIER_FAILED" "The classifier did not write auto_decision.json" "classify"
    exit 2
}
try {
    $decision = Get-Content -LiteralPath $ClassifierDecisionPath -Raw | ConvertFrom-Json
    Assert-ClassifierDecision $decision $classifierReport
} catch {
    Remove-Item -LiteralPath $ClassifierDecisionPath -Force -ErrorAction SilentlyContinue
    Write-AutoStop "AUTO_CLASSIFIER_FAILED" $_.Exception.Message "classify"
    exit 2
}
Remove-Item -LiteralPath $ClassifierDecisionPath -Force
if ([string]$decision.status -eq "STOP") {
    if ($classifierExitCode -ne 2) {
        Write-AutoStop "AUTO_CLASSIFIER_FAILED" "The classifier returned STOP with exit code $classifierExitCode instead of 2" "classify"
        exit 2
    }
    Write-ClassifierTerminalDecision $decision $classifierReport $null
    Write-Warning "$($decision.code): $($decision.reason)"
    exit 2
}
if ($classifierExitCode -ne 0) {
    Write-AutoStop "AUTO_CLASSIFIER_FAILED" "The classifier returned PASS with exit code $classifierExitCode instead of 0" "classify"
    exit 2
}

$SelectedProfile = [string]$decision.profile
$resolvedRun = Resolve-AutoRunSettings $SelectedProfile
Write-ClassifierTerminalDecision $decision $classifierReport $resolvedRun
Write-Host "[auto] Selected $SelectedProfile; starting the signed production run" -ForegroundColor Green

$finalSelectParameters = New-CommonRunnerParameters $SelectedProfile $RunName $resolvedRun.resolved_selected_frames $resolvedRun.resolved_training_steps $MaxLongSide $CandidateMultiplier $MaxCumulativeFlow $ResolvedTrainer ([bool]$AdaptiveExtraction)
$finalSelectParameters["FromStage"] = "extract"
$finalSelectParameters["ToStage"] = "select"
$finalSelectParameters["NoBlender"] = $true
$finalSelectParameters["NoBrowser"] = $true
Invoke-CoreRunner $finalSelectParameters

$finalShotManifestPath = New-SignedShotManifest $FinalRoot
New-Item -ItemType Directory -Force -Path $FinalRoot | Out-Null

$finalParameters = New-CommonRunnerParameters $SelectedProfile $RunName $resolvedRun.resolved_selected_frames $resolvedRun.resolved_training_steps $MaxLongSide $CandidateMultiplier $MaxCumulativeFlow $ResolvedTrainer ([bool]$AdaptiveExtraction)
$finalParameters["FromStage"] = "solve"
$finalParameters["ToStage"] = $ToStage
$finalParameters["ShotManifestPath"] = $finalShotManifestPath
if ($NoBlender) { $finalParameters["NoBlender"] = $true }
if ($OpenBlender) { $finalParameters["OpenBlender"] = $true }
if ($NoBrowser) { $finalParameters["NoBrowser"] = $true }
Invoke-CoreRunner $finalParameters
