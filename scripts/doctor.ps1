[CmdletBinding()]
param(
    [string]$ConfigPath,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command Import-PowerShellDataFile -ErrorAction SilentlyContinue)) {
    Import-Module -Name (Join-Path $PSHOME "Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1") -Force
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$threeDGrutRuntimeHelperPath = Join-Path $projectRoot "pipeline\three_dgrut_runtime.ps1"
. $threeDGrutRuntimeHelperPath
$productionSafetyPath = Join-Path $projectRoot "pipeline\production_safety.ps1"
. $productionSafetyPath
if (-not $ConfigPath) { $ConfigPath = Join-Path $projectRoot "splatitup.local.psd1" }
$config = if (Test-Path -LiteralPath $ConfigPath) {
    Import-PowerShellDataFile -LiteralPath $ConfigPath
} else {
    @{}
}

function Test-ConfiguredTool([string]$Name, [string]$CommandName, [string]$LocalPath) {
    $configuredPath = $null
    if ($config.ContainsKey("Tools") -and $config.Tools.ContainsKey($Name)) {
        $configuredPath = [string]$config.Tools[$Name]
    }
    $resolved = $null
    foreach ($candidate in @($configuredPath, $LocalPath)) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            $resolved = (Get-Item -LiteralPath $candidate).FullName
            break
        }
    }
    if (-not $resolved -and $CommandName) {
        $command = Get-Command $CommandName -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($command) { $resolved = $command.Source }
    }
    [pscustomobject]@{
        Component = $Name
        Status = if ($resolved) { "Ready" } else { "Missing" }
        Path = $resolved
    }
}

$spirulaConfigured = $config.ContainsKey("Tools") -and $config.Tools.ContainsKey("Spirula") -and [string]$config.Tools.Spirula
$checks = @(
    Test-ConfiguredTool "Python" "python.exe" (Join-Path $projectRoot ".venv\Scripts\python.exe")
    Test-ConfiguredTool "Ffmpeg" "ffmpeg.exe" (Join-Path $projectRoot "tools\ffmpeg\bin\ffmpeg.exe")
    Test-ConfiguredTool "Ffprobe" "ffprobe.exe" (Join-Path $projectRoot "tools\ffmpeg\bin\ffprobe.exe")
    Test-ConfiguredTool "Colmap" "colmap.exe" (Join-Path $projectRoot "tools\colmap\bin\colmap.exe")
    Test-ConfiguredTool "VocabTree" $null (Join-Path $projectRoot "tools\colmap\vocab_tree_faiss_flickr100K_words32K.bin")
    Test-ConfiguredTool "Brush" $null (Join-Path $projectRoot "tools\brush\brush_app.exe")
    Test-ConfiguredTool $(if ($spirulaConfigured) { "Spirula" } else { "Spirula (optional)" }) $null (Join-Path $projectRoot "tools\spirula\v2026.8.28\bin\spirula.exe")
    Test-ConfiguredTool "BlenderBuilder" $null "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    Test-ConfiguredTool "BlenderOpen" $null "C:\Users\mat\Applications\blender-5.2.0-windows-x64\blender.exe"
)

$outputRoot = if ($config.ContainsKey("OutputRoot") -and $config.OutputRoot) { [string]$config.OutputRoot } else { Join-Path $projectRoot "runs" }
$minimumFreeGB = Get-SplatProductionValue -Config $config -Name "MinimumFreeSpaceGB" -Default 20
$perJobReserveGB = Get-SplatProductionValue -Config $config -Name "PerQueuedJobReserveGB" -Default 12
$plannedQueueSize = [int](Get-SplatProductionValue -Config $config -Name "PlannedQueueSize" -Default 1)
$requiredQueueGB = [math]::Max($minimumFreeGB, $perJobReserveGB * $plannedQueueSize)
try {
    $capacity = Assert-SplatDiskCapacity -Path $outputRoot -RequiredGB $requiredQueueGB
    $checks += [pscustomobject]@{
        Component = "Production output capacity"
        Status = "Ready"
        Path = "$($capacity.path) | $($capacity.available_gb) GB free | $requiredQueueGB GB required for $plannedQueueSize queued jobs"
    }
} catch {
    $checks += [pscustomobject]@{ Component = "Production output capacity"; Status = "Insufficient"; Path = $_.Exception.Message }
}

$pythonCheck = $checks | Where-Object Component -eq "Python" | Select-Object -First 1
if ($pythonCheck.Status -eq "Ready") {
    & $pythonCheck.Path -c "import cv2, numpy; from PIL import Image" 2>$null
    $checks += [pscustomobject]@{
        Component = "Python packages"
        Status = if ($LASTEXITCODE -eq 0) { "Ready" } else { "Missing OpenCV/NumPy/Pillow" }
        Path = "opencv-python-headless, numpy, Pillow"
    }
    $localPython = [System.IO.Path]::GetFullPath((Join-Path $projectRoot ".venv\Scripts\python.exe"))
    $checks += [pscustomobject]@{
        Component = "Fork-local Python"
        Status = if ([StringComparer]::OrdinalIgnoreCase.Equals([string]$pythonCheck.Path, $localPython)) { "Ready" } else { "External dependency" }
        Path = [string]$pythonCheck.Path
    }
}

$parserErrors = @()
foreach ($scriptPath in @(
    (Join-Path $projectRoot "SplatItUp888.ps1"),
    (Join-Path $projectRoot "pipeline\run_video_to_splat.ps1"),
    (Join-Path $projectRoot "pipeline\run_auto_video_to_splat.ps1"),
    (Join-Path $projectRoot "pipeline\run_batch_queue.ps1")
)) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$tokens, [ref]$errors) | Out-Null
    $parserErrors += @($errors)
}
$checks += [pscustomobject]@{
    Component = "PowerShell parsers"
    Status = if ($parserErrors.Count -eq 0) { "Ready" } else { "Failed" }
    Path = if ($parserErrors.Count -eq 0) { "GUI, core, auto, and batch runners" } else { ($parserErrors.Message -join " | ") }
}

$smokeOutput = @(& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot "SplatItUp888.ps1") -SmokeTest 2>&1)
$smokeReady = $LASTEXITCODE -eq 0 -and ($smokeOutput -join "`n") -match '"batch_runner_found":true'
$checks += [pscustomobject]@{
    Component = "GUI production smoke"
    Status = if ($smokeReady) { "Ready" } else { "Failed" }
    Path = if ($smokeReady) { "single and multi-video routes" } else { (($smokeOutput | Select-Object -Last 3) -join " | ") }
}

if (-not $SkipTests -and $pythonCheck.Status -eq "Ready") {
    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $testOutput = @(& $pythonCheck.Path -m unittest discover -s (Join-Path $projectRoot "tests") -q 2>&1)
        $testExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    $checks += [pscustomobject]@{
        Component = "Regression suite"
        Status = if ($testExitCode -eq 0) { "Ready" } else { "Failed" }
        Path = if ($testExitCode -eq 0) { (($testOutput | Select-Object -Last 2) -join " | ") } else { (($testOutput | Select-Object -Last 6) -join " | ") }
    }
}

$colmapCheck = $checks | Where-Object Component -eq "Colmap" | Select-Object -First 1
if ($colmapCheck.Status -eq "Ready") {
    $colmapVersion = (& $colmapCheck.Path version 2>&1) -join "`n"
    $checks += [pscustomobject]@{
        Component = "COLMAP 4.1 CUDA"
        Status = if ($LASTEXITCODE -eq 0 -and $colmapVersion -match "COLMAP 4\.1\.1" -and $colmapVersion -match "CUDA") { "Ready" } else { "Wrong build" }
        Path = ($colmapVersion -split "`r?`n")[0]
    }
}

$blenderBuilderCheck = $checks | Where-Object Component -eq "BlenderBuilder" | Select-Object -First 1
if ($blenderBuilderCheck.Status -eq "Ready") {
    & $blenderBuilderCheck.Path --background --factory-startup --disable-autoexec --addons bl_ext.user_default.dgs_render_by_kiri_engine --python-expr "import bpy; assert hasattr(bpy.ops.sna, 'dgs_render_import_ply_e0a3a')" 2>$null | Out-Null
    $kiriOperatorReady = $LASTEXITCODE -eq 0
    $kiriManifest = Join-Path $env:APPDATA "Blender Foundation\Blender\5.0\extensions\user_default\dgs_render_by_kiri_engine\blender_manifest.toml"
    $kiriVersionReady = $false
    if (Test-Path -LiteralPath $kiriManifest -PathType Leaf) {
        $kiriVersionReady = (Get-Content -LiteralPath $kiriManifest -Raw) -match '(?m)^version\s*=\s*"4\.1\.5"\s*$'
    }
    $checks += [pscustomobject]@{
        Component = "KIRI 3DGS Render"
        Status = if ($kiriOperatorReady -and $kiriVersionReady) { "Ready" } else { "Missing/wrong version" }
        Path = if ($kiriVersionReady) { "4.1.5 - $kiriManifest" } else { $kiriManifest }
    }
}

$threeDGrutReady = $false
$threeDGrutPath = $null
$threeDGrutRuntime = $null
$threeDGrutError = $null
$threeDGrutConfigured = $config.ContainsKey("ThreeDGRUT") -and
    ([string]$config.ThreeDGRUT.Repo -or [string]$config.ThreeDGRUT.Python)
if ($threeDGrutConfigured) {
    $threeDGrutPath = [string]$config.ThreeDGRUT.Repo
    try {
        $threeDGrutRuntime = Initialize-ThreeDGrutRuntime -ThreeDGrutConfig $config.ThreeDGRUT
        $threeDGrutReady = $true
    } catch {
        $threeDGrutError = $_.Exception.Message
    }
}
$checks += [pscustomobject]@{
    Component = if ($threeDGrutConfigured) { "3DGRUT" } else { "3DGRUT (optional)" }
    Status = if ($threeDGrutReady) { "Ready" } elseif ($threeDGrutConfigured) { "Incomplete portable runtime" } else { "Not configured" }
    Path = if ($threeDGrutError) { "$threeDGrutPath | $threeDGrutError" } else { $threeDGrutPath }
}
if ($threeDGrutRuntime) {
    $checks += [pscustomobject]@{
        Component = "3DGRUT LPIPS/VGG cache"
        Status = "Ready"
        Path = "$($threeDGrutRuntime.LpipsVggPath) | sha256=$($threeDGrutRuntime.LpipsVggSha256)"
    }
    $checks += [pscustomobject]@{
        Component = "3DGRUT Python UTF-8 runtime"
        Status = if ($threeDGrutRuntime.PythonUtf8 -eq "1" -and $threeDGrutRuntime.PythonIoEncoding -eq "utf-8") { "Ready" } else { "Wrong encoding" }
        Path = "PYTHONUTF8=$($threeDGrutRuntime.PythonUtf8) | PYTHONIOENCODING=$($threeDGrutRuntime.PythonIoEncoding)"
    }
}

$checks | Format-Table -AutoSize
$requiredMissing = @($checks | Where-Object { $_.Component -notin @("3DGRUT (optional)", "Spirula (optional)") -and $_.Status -ne "Ready" })
if ($requiredMissing.Count -gt 0) { exit 1 }
Write-Host "`nSplatItUp888 mechanical production checks passed. Each splat still requires visual approval." -ForegroundColor Green
