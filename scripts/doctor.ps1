[CmdletBinding()]
param(
    [string]$ConfigPath
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
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

$checks = @(
    Test-ConfiguredTool "Python" "python.exe" (Join-Path $projectRoot ".venv\Scripts\python.exe")
    Test-ConfiguredTool "Ffmpeg" "ffmpeg.exe" (Join-Path $projectRoot "tools\ffmpeg\bin\ffmpeg.exe")
    Test-ConfiguredTool "Ffprobe" "ffprobe.exe" (Join-Path $projectRoot "tools\ffmpeg\bin\ffprobe.exe")
    Test-ConfiguredTool "Colmap" "colmap.exe" (Join-Path $projectRoot "tools\colmap\bin\colmap.exe")
    Test-ConfiguredTool "Brush" $null (Join-Path $projectRoot "tools\brush\brush_app.exe")
)

$pythonCheck = $checks | Where-Object Component -eq "Python" | Select-Object -First 1
if ($pythonCheck.Status -eq "Ready") {
    & $pythonCheck.Path -c "import numpy; from PIL import Image" 2>$null
    $checks += [pscustomobject]@{
        Component = "Python packages"
        Status = if ($LASTEXITCODE -eq 0) { "Ready" } else { "Missing NumPy/Pillow" }
        Path = "numpy, Pillow"
    }
}

$threeDGrutReady = $false
$threeDGrutPath = $null
if ($config.ContainsKey("ThreeDGRUT")) {
    $threeDGrutPath = [string]$config.ThreeDGRUT.Repo
    $threeDGrutPython = [string]$config.ThreeDGRUT.Python
    if ($threeDGrutPath -and $threeDGrutPython) {
        $threeDGrutReady = (Test-Path -LiteralPath (Join-Path $threeDGrutPath "train.py") -PathType Leaf) -and
            (Test-Path -LiteralPath $threeDGrutPython -PathType Leaf)
    }
}
$checks += [pscustomobject]@{
    Component = "3DGRUT (optional)"
    Status = if ($threeDGrutReady) { "Ready" } else { "Not configured" }
    Path = $threeDGrutPath
}

$checks | Format-Table -AutoSize
$requiredMissing = @($checks | Where-Object { $_.Component -ne "3DGRUT (optional)" -and $_.Status -ne "Ready" })
if ($requiredMissing.Count -gt 0) { exit 1 }
Write-Host "`nSplatItUp888 stable pipeline is ready." -ForegroundColor Green
