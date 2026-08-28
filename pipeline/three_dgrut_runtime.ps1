Set-StrictMode -Version Latest

function Initialize-ThreeDGrutRuntime {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$ThreeDGrutConfig
    )

    foreach ($name in @("Repo", "Python", "CudaHome", "VcVars", "VcVarsVersion", "RuntimeCacheRoot", "TorchCudaArchList")) {
        if (-not $ThreeDGrutConfig.ContainsKey($name) -or -not [string]$ThreeDGrutConfig[$name]) {
            throw "3DGRUT.$name is required for the portable Windows runtime."
        }
    }

    $repo = [System.IO.Path]::GetFullPath([string]$ThreeDGrutConfig.Repo)
    $python = [System.IO.Path]::GetFullPath([string]$ThreeDGrutConfig.Python)
    $cudaHome = [System.IO.Path]::GetFullPath([string]$ThreeDGrutConfig.CudaHome)
    $vcVars = [System.IO.Path]::GetFullPath([string]$ThreeDGrutConfig.VcVars)
    $vcVarsVersion = [string]$ThreeDGrutConfig.VcVarsVersion
    $runtimeCacheRoot = [System.IO.Path]::GetFullPath([string]$ThreeDGrutConfig.RuntimeCacheRoot)
    $torchCudaArchList = [string]$ThreeDGrutConfig.TorchCudaArchList
    $torchCudaArchMatch = [regex]::Match($torchCudaArchList, "^(?<major>[0-9]+)\.(?<minor>[0-9]+)$")
    if (-not $torchCudaArchMatch.Success) {
        throw "3DGRUT.TorchCudaArchList must contain one CUDA architecture such as 8.6."
    }
    $tcnnCudaArchitectures = "$($torchCudaArchMatch.Groups['major'].Value)$($torchCudaArchMatch.Groups['minor'].Value)"
    $venvScripts = Split-Path -Parent $python
    $nvcc = Join-Path $cudaHome "bin\nvcc.exe"
    $slangc = Join-Path $venvScripts "slangc.exe"

    foreach ($requiredFile in @(
        @{ Name = "3DGRUT train.py"; Path = (Join-Path $repo "train.py") },
        @{ Name = "3DGRUT 3DGUT config"; Path = (Join-Path $repo "configs\apps\colmap_3dgut.yaml") },
        @{ Name = "3DGRUT 3DGUT-MCMC config"; Path = (Join-Path $repo "configs\apps\colmap_3dgut_mcmc.yaml") },
        @{ Name = "3DGRUT Python"; Path = $python },
        @{ Name = "CUDA nvcc"; Path = $nvcc },
        @{ Name = "MSVC vcvars"; Path = $vcVars },
        @{ Name = "Slang compiler"; Path = $slangc }
    )) {
        if (-not (Test-Path -LiteralPath $requiredFile.Path -PathType Leaf)) {
            throw "$($requiredFile.Name) is missing: $($requiredFile.Path)"
        }
    }

    if ($env:NVCC_APPEND_FLAGS -match "--allow-unsupported-compiler" -or
        $env:CUDAFLAGS -match "--allow-unsupported-compiler") {
        throw "Unsupported CUDA compiler flags are present. Launch 3DGRUT with the configured supported MSVC toolset."
    }

    $vcCommand = "`"$vcVars`" -vcvars_ver=$vcVarsVersion >nul 2>&1 && set"
    $vcEnvironment = @(& $env:ComSpec /d /s /c $vcCommand)
    if ($LASTEXITCODE -ne 0 -or $vcEnvironment.Count -eq 0) {
        throw "Failed to initialize the configured MSVC environment: $vcVars"
    }
    $seenEnvironmentNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
    foreach ($line in $vcEnvironment) {
        if ($line -match "^([^=]+)=(.*)$") {
            $environmentName = [string]$matches[1]
            if ($seenEnvironmentNames.Add($environmentName)) {
                [System.Environment]::SetEnvironmentVariable($environmentName, $matches[2], "Process")
            }
        }
    }

    $env:CUDA_HOME = $cudaHome
    $env:CUDA_PATH = $cudaHome
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONUTF8 = "1"
    $env:PYTHONIOENCODING = "utf-8"
    $env:TORCH_CUDA_ARCH_LIST = $torchCudaArchList
    $env:TCNN_CUDA_ARCHITECTURES = $tcnnCudaArchitectures
    $env:WARP_CACHE_PATH = Join-Path $runtimeCacheRoot "warp"
    $env:TORCH_EXTENSIONS_DIR = Join-Path $runtimeCacheRoot "torch-extensions"
    $env:TORCH_HOME = Join-Path $runtimeCacheRoot "torch"
    $env:Path = "$(Join-Path $cudaHome 'bin');$venvScripts;$env:Path"

    foreach ($cachePath in @($env:WARP_CACHE_PATH, $env:TORCH_EXTENSIONS_DIR, $env:TORCH_HOME)) {
        New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
        $probe = Join-Path $cachePath ".splatitup-write-probe-$PID.tmp"
        try {
            [System.IO.File]::WriteAllText($probe, "ok")
        } finally {
            if (Test-Path -LiteralPath $probe) { Remove-Item -LiteralPath $probe -Force }
        }
    }

    $lpipsVggPath = Join-Path $env:TORCH_HOME "hub\checkpoints\vgg16-397923af.pth"
    $lpipsVggExpectedSha256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
    if (-not (Test-Path -LiteralPath $lpipsVggPath -PathType Leaf)) {
        throw "3DGRUT LPIPS validation requires the cached VGG16 checkpoint: $lpipsVggPath"
    }
    $lpipsVggSha256 = (Get-FileHash -LiteralPath $lpipsVggPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($lpipsVggSha256 -ne $lpipsVggExpectedSha256) {
        throw "3DGRUT LPIPS VGG16 checkpoint hash mismatch: $lpipsVggPath"
    }

    $tools = [ordered]@{}
    foreach ($toolName in @("cl.exe", "cmake.exe", "ninja.exe", "slangc.exe", "nvcc.exe")) {
        $tool = Get-Command $toolName -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $tool) { throw "3DGRUT runtime tool is not available on PATH: $toolName" }
        $tools[$toolName] = (Get-Item -LiteralPath $tool.Source).FullName
    }

    $previousErrorAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $nvccVersion = (& $tools["nvcc.exe"] --version 2>&1) -join "`n"
        $nvccExitCode = $LASTEXITCODE
        $slangVersion = ((& $tools["slangc.exe"] -version 2>&1) -join "`n").Trim()
        $slangExitCode = $LASTEXITCODE
        $clVersion = (& $tools["cl.exe"] 2>&1) -join "`n"
        $clExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorAction
    }
    if ($nvccExitCode -ne 0 -or $nvccVersion -notmatch "release 12\.8") {
        throw "3DGRUT requires CUDA 12.8; configured nvcc did not report release 12.8."
    }
    if ($slangExitCode -ne 0 -or $slangVersion -ne "2026.5.2") {
        throw "3DGRUT requires slangc 2026.5.2; found '$slangVersion'."
    }
    $expectedClVersion = $vcVarsVersion -replace "^14\.", "19."
    if ($clExitCode -ne 0 -or $clVersion -notmatch "Version $([regex]::Escape($expectedClVersion))\.") {
        throw "Configured MSVC toolset $vcVarsVersion did not activate compiler $expectedClVersion.x."
    }

    $requiredModules = @(
        "hydra", "omegaconf", "torch", "kaolin", "tinycudann", "ppisp", "fused_ssim",
        "threedgrut", "threedgut_tracer", "threedgrt_tracer", "ncore"
    )
    $moduleLiteral = ($requiredModules | ForEach-Object { "'$_'" }) -join ","
    $environmentCheck = "import importlib.util as u; required=($moduleLiteral,); missing=[name for name in required if u.find_spec(name) is None]; assert not missing, 'missing modules: '+','.join(missing)"
    & $python -c $environmentCheck 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "3DGRUT Python environment is missing one or more required official packages."
    }

    $fingerprintRows = @(
        "runtime_contract==portable-windows-v1",
        "cuda_home==$cudaHome",
        "vcvars_version==$vcVarsVersion",
        "python_utf8==$env:PYTHONUTF8",
        "python_io_encoding==$env:PYTHONIOENCODING",
        "torch_cuda_arch_list==$torchCudaArchList",
        "tcnn_cuda_architectures==$tcnnCudaArchitectures",
        "torch_home==$env:TORCH_HOME",
        "lpips_vgg_sha256==$lpipsVggSha256"
    )
    foreach ($toolName in $tools.Keys) {
        $toolPath = [string]$tools[$toolName]
        $toolHash = (Get-FileHash -LiteralPath $toolPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $fingerprintRows += "tool.$toolName.path==$toolPath"
        $fingerprintRows += "tool.$toolName.sha256==$toolHash"
    }

    return [pscustomobject]@{
        Repo = $repo
        Python = $python
        CudaHome = $cudaHome
        VenvScripts = $venvScripts
        RuntimeCacheRoot = $runtimeCacheRoot
        TorchHome = $env:TORCH_HOME
        PythonUtf8 = $env:PYTHONUTF8
        PythonIoEncoding = $env:PYTHONIOENCODING
        TcnnCudaArchitectures = $tcnnCudaArchitectures
        LpipsVggPath = $lpipsVggPath
        LpipsVggSha256 = $lpipsVggSha256
        RequiredModules = $requiredModules
        FingerprintRows = $fingerprintRows
    }
}
