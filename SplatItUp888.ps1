[CmdletBinding()]
param(
    [switch]$SmokeTest,
    [string]$ScreenshotPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (-not (Get-Command Import-PowerShellDataFile -ErrorAction SilentlyContinue)) {
    Import-Module -Name (Join-Path $PSHOME "Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1") -Force
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$RunnerPath = Join-Path $PSScriptRoot "pipeline\run_video_to_splat.ps1"
$AutoRunnerPath = Join-Path $PSScriptRoot "pipeline\run_auto_video_to_splat.ps1"
$BatchRunnerPath = Join-Path $PSScriptRoot "pipeline\run_batch_queue.ps1"
$ThreeDGrutRuntimeHelperPath = Join-Path $PSScriptRoot "pipeline\three_dgrut_runtime.ps1"
. $ThreeDGrutRuntimeHelperPath
$ConfigPath = Join-Path $PSScriptRoot "splatitup.local.psd1"
$OutputRoot = Join-Path $PSScriptRoot "runs"
if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
    $appConfig = Import-PowerShellDataFile -LiteralPath $ConfigPath
    if ($appConfig.ContainsKey("OutputRoot") -and $appConfig.OutputRoot) {
        $OutputRoot = [string]$appConfig.OutputRoot
    }
}
$SupportedExtensions = @(".mov", ".mp4", ".m4v", ".avi", ".mkv")

$script:ActiveProcess = $null
$script:CompletionHandled = $false
$script:PipelineLogPath = $null
$script:LogStream = $null
$script:LogReader = $null
$script:FinalPly = $null
$script:BlenderFile = $null
$script:RunRoot = $null
$script:ActiveRouteMode = $null
$script:PipelineStartedUtc = $null
$script:AutoDecisionBaseline = $null
$script:TrainingDecisionBaseline = $null
$script:ThreeDGrutConfigurationError = $null
$script:SelectedVideoPaths = @()
$script:QueuePath = $null
$script:ActiveIsBatch = $false

function New-Color([string]$Hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($Hex)
}

function Get-RunName([string]$Path) {
    $name = [System.IO.Path]::GetFileNameWithoutExtension($Path).ToLowerInvariant()
    $name = ($name -replace "[^a-z0-9]+", "-").Trim([char[]]"-")
    if (-not $name) { return "gaussian-run" }
    return $name
}

function ConvertTo-SingleQuotedPowerShell([string]$Value) {
    return "'" + $Value.Replace("'", "''") + "'"
}

function Test-VideoPath([string]$Path) {
    if (-not $Path) { return $false }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    return $SupportedExtensions -contains [System.IO.Path]::GetExtension($Path).ToLowerInvariant()
}

function Test-SelectedVideos {
    return $script:SelectedVideoPaths.Count -gt 0 -and
        @($script:SelectedVideoPaths | Where-Object { -not (Test-VideoPath $_) }).Count -eq 0
}

function Test-ThreeDGRUTConfiguration {
    $script:ThreeDGrutConfigurationError = $null
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { return $false }
    $config = Import-PowerShellDataFile -LiteralPath $ConfigPath
    if (-not $config.ContainsKey("ThreeDGRUT")) { return $false }
    $repo = [string]$config.ThreeDGRUT.Repo
    $python = [string]$config.ThreeDGRUT.Python
    if (-not $repo -or -not $python) { return $false }
    $pathsReady = (Test-Path -LiteralPath $repo -PathType Container) -and
        (Test-Path -LiteralPath $python -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $repo "train.py") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $repo "configs\apps\colmap_3dgut.yaml") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $repo "configs\apps\colmap_3dgut_mcmc.yaml") -PathType Leaf)
    if (-not $pathsReady) { return $false }
    $previousPythonUtf8 = [Environment]::GetEnvironmentVariable("PYTHONUTF8", "Process")
    $previousPythonIoEncoding = [Environment]::GetEnvironmentVariable("PYTHONIOENCODING", "Process")
    try {
        Initialize-ThreeDGrutRuntime -ThreeDGrutConfig $config.ThreeDGRUT | Out-Null
        return $true
    } catch {
        $script:ThreeDGrutConfigurationError = $_.Exception.Message
        return $false
    } finally {
        [Environment]::SetEnvironmentVariable("PYTHONUTF8", $previousPythonUtf8, "Process")
        [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", $previousPythonIoEncoding, "Process")
    }
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "SplatItUp888"
$form.StartPosition = "CenterScreen"
$form.ClientSize = New-Object System.Drawing.Size(820, 800)
$form.MinimumSize = New-Object System.Drawing.Size(720, 680)
$form.BackColor = New-Color "#F3F5F6"
$form.ForeColor = New-Color "#1D2329"
$form.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$form.AllowDrop = $true

$header = New-Object System.Windows.Forms.Panel
$header.Dock = "Top"
$header.Height = 64
$header.BackColor = New-Color "#20262D"
$form.Controls.Add($header)

$titleLabel = New-Object System.Windows.Forms.Label
$titleLabel.Text = "SplatItUp888"
$titleLabel.AutoSize = $true
$titleLabel.Location = New-Object System.Drawing.Point(22, 12)
$titleLabel.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 16)
$titleLabel.ForeColor = [System.Drawing.Color]::White
$header.Controls.Add($titleLabel)

$subtitleLabel = New-Object System.Windows.Forms.Label
$subtitleLabel.Text = "VIDEO  >  MOTION QC  >  COLMAP  >  3DGS  >  BLENDER"
$subtitleLabel.AutoSize = $true
$subtitleLabel.Location = New-Object System.Drawing.Point(24, 40)
$subtitleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 7.5)
$subtitleLabel.ForeColor = New-Color "#AEB7C0"
$header.Controls.Add($subtitleLabel)

$dropPanel = New-Object System.Windows.Forms.Panel
$dropPanel.Location = New-Object System.Drawing.Point(24, 82)
$dropPanel.Size = New-Object System.Drawing.Size(772, 86)
$dropPanel.Anchor = "Top, Left, Right"
$dropPanel.BackColor = [System.Drawing.Color]::White
$dropPanel.BorderStyle = "FixedSingle"
$dropPanel.AllowDrop = $true
$form.Controls.Add($dropPanel)

$dropLabel = New-Object System.Windows.Forms.Label
$dropLabel.Text = "Drop object, walkthrough, or aerial video"
$dropLabel.AutoSize = $true
$dropLabel.Location = New-Object System.Drawing.Point(22, 19)
$dropLabel.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 13)
$dropLabel.ForeColor = New-Color "#20262D"
$dropPanel.Controls.Add($dropLabel)

$dropFormatsLabel = New-Object System.Windows.Forms.Label
$dropFormatsLabel.Text = "MOV, MP4, M4V, AVI, MKV"
$dropFormatsLabel.AutoSize = $true
$dropFormatsLabel.Location = New-Object System.Drawing.Point(24, 50)
$dropFormatsLabel.ForeColor = New-Color "#68727D"
$dropPanel.Controls.Add($dropFormatsLabel)

$videoPathBox = New-Object System.Windows.Forms.TextBox
$videoPathBox.Location = New-Object System.Drawing.Point(24, 181)
$videoPathBox.Size = New-Object System.Drawing.Size(668, 25)
$videoPathBox.Anchor = "Top, Left, Right"
$videoPathBox.ReadOnly = $true
$videoPathBox.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($videoPathBox)

$browseButton = New-Object System.Windows.Forms.Button
$browseButton.Text = "Browse"
$browseButton.Location = New-Object System.Drawing.Point(704, 178)
$browseButton.Size = New-Object System.Drawing.Size(92, 30)
$browseButton.Anchor = "Top, Right"
$browseButton.FlatStyle = "Flat"
$browseButton.BackColor = [System.Drawing.Color]::White
$browseButton.FlatAppearance.BorderColor = New-Color "#B9C1C8"
$form.Controls.Add($browseButton)

$settingsGroup = New-Object System.Windows.Forms.GroupBox
$settingsGroup.Text = "Run settings"
$settingsGroup.Location = New-Object System.Drawing.Point(24, 222)
$settingsGroup.Size = New-Object System.Drawing.Size(772, 175)
$settingsGroup.Anchor = "Top, Left, Right"
$settingsGroup.BackColor = [System.Drawing.Color]::White
$form.Controls.Add($settingsGroup)

$runNameLabel = New-Object System.Windows.Forms.Label
$runNameLabel.Text = "Run name"
$runNameLabel.AutoSize = $true
$runNameLabel.Location = New-Object System.Drawing.Point(16, 24)
$settingsGroup.Controls.Add($runNameLabel)

$runNameBox = New-Object System.Windows.Forms.TextBox
$runNameBox.Location = New-Object System.Drawing.Point(18, 46)
$runNameBox.Size = New-Object System.Drawing.Size(220, 25)
$settingsGroup.Controls.Add($runNameBox)

$framesLabel = New-Object System.Windows.Forms.Label
$framesLabel.Text = "Base frames"
$framesLabel.AutoSize = $true
$framesLabel.Location = New-Object System.Drawing.Point(252, 24)
$settingsGroup.Controls.Add($framesLabel)

$framesInput = New-Object System.Windows.Forms.NumericUpDown
$framesInput.Location = New-Object System.Drawing.Point(254, 46)
$framesInput.Size = New-Object System.Drawing.Size(90, 25)
$framesInput.Minimum = 40
$framesInput.Maximum = 3000
$framesInput.Increment = 10
$framesInput.Value = 300
$settingsGroup.Controls.Add($framesInput)

$stepsLabel = New-Object System.Windows.Forms.Label
$stepsLabel.Text = "Training steps"
$stepsLabel.AutoSize = $true
$stepsLabel.Location = New-Object System.Drawing.Point(350, 84)
$settingsGroup.Controls.Add($stepsLabel)

$stepsInput = New-Object System.Windows.Forms.NumericUpDown
$stepsInput.Location = New-Object System.Drawing.Point(352, 106)
$stepsInput.Size = New-Object System.Drawing.Size(130, 25)
$stepsInput.Minimum = 5000
$stepsInput.Maximum = 100000
$stepsInput.Increment = 5000
$stepsInput.Value = 30000
$stepsInput.ThousandsSeparator = $true
$stepsInput.Enabled = $false
$settingsGroup.Controls.Add($stepsInput)

$trainerLabel = New-Object System.Windows.Forms.Label
$trainerLabel.Text = "Trainer"
$trainerLabel.AutoSize = $true
$trainerLabel.Location = New-Object System.Drawing.Point(360, 24)
$settingsGroup.Controls.Add($trainerLabel)

$trainerInput = New-Object System.Windows.Forms.ComboBox
$trainerInput.Location = New-Object System.Drawing.Point(362, 45)
$trainerInput.Size = New-Object System.Drawing.Size(180, 25)
$trainerInput.DropDownStyle = "DropDownList"
[void]$trainerInput.Items.AddRange([object[]]@("Brush", "Spirula", "3DGUT-MCMC"))
$trainerInput.SelectedIndex = 0
$settingsGroup.Controls.Add($trainerInput)

$sceneTypeLabel = New-Object System.Windows.Forms.Label
$sceneTypeLabel.Text = "Capture type"
$sceneTypeLabel.AutoSize = $true
$sceneTypeLabel.Location = New-Object System.Drawing.Point(550, 24)
$settingsGroup.Controls.Add($sceneTypeLabel)

$sceneTypeInput = New-Object System.Windows.Forms.ComboBox
$sceneTypeInput.Location = New-Object System.Drawing.Point(552, 45)
$sceneTypeInput.Size = New-Object System.Drawing.Size(194, 25)
$sceneTypeInput.DropDownStyle = "DropDownList"
[void]$sceneTypeInput.Items.AddRange([object[]]@(
    "Choose capture type...",
    "Auto - inspect and pilot",
    "Walkthrough - open route",
    "House - closed loop",
    "Object / vehicle orbit",
    "Aerial exterior / building"
))
$sceneTypeInput.SelectedIndex = 1
$settingsGroup.Controls.Add($sceneTypeInput)

$modeLabel = New-Object System.Windows.Forms.Label
$modeLabel.Text = "Mode"
$modeLabel.AutoSize = $true
$modeLabel.Location = New-Object System.Drawing.Point(16, 90)
$settingsGroup.Controls.Add($modeLabel)

$previewMode = New-Object System.Windows.Forms.RadioButton
$previewMode.Text = "Preview"
$previewMode.Appearance = "Button"
$previewMode.FlatStyle = "Flat"
$previewMode.TextAlign = "MiddleCenter"
$previewMode.Location = New-Object System.Drawing.Point(66, 82)
$previewMode.Size = New-Object System.Drawing.Size(82, 30)
$settingsGroup.Controls.Add($previewMode)

$finalMode = New-Object System.Windows.Forms.RadioButton
$finalMode.Text = "Final"
$finalMode.Appearance = "Button"
$finalMode.FlatStyle = "Flat"
$finalMode.TextAlign = "MiddleCenter"
$finalMode.Location = New-Object System.Drawing.Point(147, 82)
$finalMode.Size = New-Object System.Drawing.Size(82, 30)
$finalMode.Checked = $true
$settingsGroup.Controls.Add($finalMode)

$customMode = New-Object System.Windows.Forms.RadioButton
$customMode.Text = "Custom"
$customMode.Appearance = "Button"
$customMode.FlatStyle = "Flat"
$customMode.TextAlign = "MiddleCenter"
$customMode.Location = New-Object System.Drawing.Point(228, 82)
$customMode.Size = New-Object System.Drawing.Size(82, 30)
$settingsGroup.Controls.Add($customMode)

$fastExtractionCheck = New-Object System.Windows.Forms.CheckBox
$fastExtractionCheck.Text = "Fast 1600px decode"
$fastExtractionCheck.AutoSize = $true
$fastExtractionCheck.Location = New-Object System.Drawing.Point(500, 112)
$settingsGroup.Controls.Add($fastExtractionCheck)

$autoRotateCheck = New-Object System.Windows.Forms.CheckBox
$autoRotateCheck.Text = "Autorotate video"
$autoRotateCheck.AutoSize = $true
$autoRotateCheck.Location = New-Object System.Drawing.Point(630, 112)
$autoRotateCheck.Checked = $true
$settingsGroup.Controls.Add($autoRotateCheck)

$openViewerCheck = New-Object System.Windows.Forms.CheckBox
$openViewerCheck.Text = "Open SuperSplat"
$openViewerCheck.AutoSize = $true
$openViewerCheck.Location = New-Object System.Drawing.Point(500, 86)
$openViewerCheck.Checked = $false
$settingsGroup.Controls.Add($openViewerCheck)

$buildBlenderCheck = New-Object System.Windows.Forms.CheckBox
$buildBlenderCheck.Text = "Build Blender scene"
$buildBlenderCheck.AutoSize = $true
$buildBlenderCheck.Location = New-Object System.Drawing.Point(500, 138)
$buildBlenderCheck.Checked = $true
$settingsGroup.Controls.Add($buildBlenderCheck)

$openBlenderCheck = New-Object System.Windows.Forms.CheckBox
$openBlenderCheck.Text = "Open Blender 5.2"
$openBlenderCheck.AutoSize = $true
$openBlenderCheck.Location = New-Object System.Drawing.Point(630, 86)
$openBlenderCheck.Checked = $true
$settingsGroup.Controls.Add($openBlenderCheck)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text = "Choose a video"
$statusLabel.AutoSize = $true
$statusLabel.Location = New-Object System.Drawing.Point(24, 421)
$statusLabel.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 9)
$form.Controls.Add($statusLabel)

$startButton = New-Object System.Windows.Forms.Button
$startButton.Text = "Start Pipeline"
$startButton.Location = New-Object System.Drawing.Point(580, 413)
$startButton.Size = New-Object System.Drawing.Size(124, 34)
$startButton.Anchor = "Top, Right"
$startButton.FlatStyle = "Flat"
$startButton.BackColor = New-Color "#D85D27"
$startButton.ForeColor = [System.Drawing.Color]::White
$startButton.FlatAppearance.BorderSize = 0
$startButton.Enabled = $false
$form.Controls.Add($startButton)

$cancelButton = New-Object System.Windows.Forms.Button
$cancelButton.Text = "Cancel"
$cancelButton.Location = New-Object System.Drawing.Point(712, 413)
$cancelButton.Size = New-Object System.Drawing.Size(84, 34)
$cancelButton.Anchor = "Top, Right"
$cancelButton.FlatStyle = "Flat"
$cancelButton.BackColor = [System.Drawing.Color]::White
$cancelButton.FlatAppearance.BorderColor = New-Color "#B9C1C8"
$cancelButton.Enabled = $false
$form.Controls.Add($cancelButton)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(24, 458)
$progress.Size = New-Object System.Drawing.Size(772, 8)
$progress.Anchor = "Top, Left, Right"
$progress.Style = "Continuous"
$form.Controls.Add($progress)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Location = New-Object System.Drawing.Point(24, 481)
$logBox.Size = New-Object System.Drawing.Size(772, 230)
$logBox.Anchor = "Top, Bottom, Left, Right"
$logBox.Multiline = $true
$logBox.ReadOnly = $true
$logBox.ScrollBars = "Vertical"
$logBox.WordWrap = $false
$logBox.BackColor = New-Color "#11161B"
$logBox.ForeColor = New-Color "#D8DEE4"
$logBox.Font = New-Object System.Drawing.Font("Consolas", 8.5)
$form.Controls.Add($logBox)

$outputCaption = New-Object System.Windows.Forms.Label
$outputCaption.Text = "Output"
$outputCaption.AutoSize = $true
$outputCaption.Anchor = "Bottom, Left"
$outputCaption.Location = New-Object System.Drawing.Point(24, 726)
$outputCaption.ForeColor = New-Color "#68727D"
$form.Controls.Add($outputCaption)

$outputLink = New-Object System.Windows.Forms.LinkLabel
$outputLink.Text = "Not created yet"
$outputLink.AutoEllipsis = $true
$outputLink.Location = New-Object System.Drawing.Point(24, 750)
$outputLink.Size = New-Object System.Drawing.Size(650, 24)
$outputLink.Anchor = "Bottom, Left, Right"
$outputLink.LinkColor = New-Color "#1E5F86"
$outputLink.ActiveLinkColor = New-Color "#D85D27"
$outputLink.Enabled = $false
$form.Controls.Add($outputLink)

$openFolderButton = New-Object System.Windows.Forms.Button
$openFolderButton.Text = "Open Folder"
$openFolderButton.Location = New-Object System.Drawing.Point(692, 740)
$openFolderButton.Size = New-Object System.Drawing.Size(104, 34)
$openFolderButton.Anchor = "Bottom, Right"
$openFolderButton.FlatStyle = "Flat"
$openFolderButton.BackColor = [System.Drawing.Color]::White
$openFolderButton.FlatAppearance.BorderColor = New-Color "#B9C1C8"
$openFolderButton.Enabled = $false
$form.Controls.Add($openFolderButton)

function Get-SceneType {
    switch ($sceneTypeInput.SelectedIndex) {
        1 { return "Auto" }
        2 { return "Walkthrough" }
        3 { return "House" }
        4 { return "Object" }
        5 { return "AerialExterior" }
        default { return $null }
    }
}

function Get-PipelineRoute {
    $sceneType = Get-SceneType
    if (-not $sceneType) { return $null }
    if ($sceneType -eq "Auto") {
        return [pscustomobject][ordered]@{
            mode = "Auto"
            runner_path = $AutoRunnerPath
            scene_type = $null
        }
    }
    return [pscustomobject][ordered]@{
        mode = "Manual"
        runner_path = $RunnerPath
        scene_type = $sceneType
    }
}

function Get-AutoPreset {
    if ($previewMode.Checked) { return "Preview" }
    if ($finalMode.Checked) { return "Final" }
    return "Custom"
}

function Get-ManualTrainerStop([string]$SceneType, [string]$Trainer) {
    if ($Trainer -eq "3DGUT") {
        return [pscustomobject][ordered]@{
            code = "EXPERIMENTAL_TRAINER_UNSUPPORTED"
            reason = "Plain 3DGUT has no measured staged quality gate. Use Brush."
        }
    }
    if ($Trainer -eq "3DGUT-MCMC" -and $SceneType -ne "AerialExterior") {
        return [pscustomobject][ordered]@{
            code = "EXPERIMENTAL_TRAINER_UNSUPPORTED"
            reason = "3DGUT-MCMC is limited to the Aerial exterior 1K visual-review smoke. Use Brush for $SceneType."
        }
    }
    return $null
}

function Get-FileVersion([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $file = Get-Item -LiteralPath $Path
    $sha = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    return "$($file.LastWriteTimeUtc.Ticks):$($file.Length):$sha"
}

function Get-CurrentAutoStopDecision {
    if ($script:ActiveRouteMode -ne "Auto" -or -not $script:RunRoot) { return $null }
    $decisionPath = Join-Path $script:RunRoot "auto_decision.json"
    if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) { return $null }
    $file = Get-Item -LiteralPath $decisionPath
    $currentVersion = Get-FileVersion $decisionPath
    if ($script:AutoDecisionBaseline -and $currentVersion -eq $script:AutoDecisionBaseline) { return $null }
    if ($script:PipelineStartedUtc -and $file.LastWriteTimeUtc -lt $script:PipelineStartedUtc) { return $null }
    try {
        $decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
    $decisionProperties = @($decision.PSObject.Properties.Name)
    if ($decisionProperties -notcontains "status" -or
        $decisionProperties -notcontains "code" -or
        $decisionProperties -notcontains "reason") {
        return $null
    }
    if ([string]$decision.status -ne "STOP" -or -not $decision.code -or -not $decision.reason) { return $null }
    return [pscustomobject][ordered]@{
        path = $decisionPath
        code = [string]$decision.code
        reason = [string]$decision.reason
    }
}

function Get-CurrentTrainingDecision {
    if (-not $script:RunRoot) { return $null }
    $decisionPath = Join-Path $script:RunRoot "training_decision.json"
    if (-not (Test-Path -LiteralPath $decisionPath -PathType Leaf)) { return $null }
    $file = Get-Item -LiteralPath $decisionPath
    $currentVersion = Get-FileVersion $decisionPath
    if ($script:TrainingDecisionBaseline -and $currentVersion -eq $script:TrainingDecisionBaseline) { return $null }
    if ($script:PipelineStartedUtc -and $file.LastWriteTimeUtc -lt $script:PipelineStartedUtc) { return $null }
    try {
        $decision = Get-Content -LiteralPath $decisionPath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
    $decisionProperties = @($decision.PSObject.Properties.Name)
    if ($decisionProperties -notcontains "status" -or
        $decisionProperties -notcontains "code" -or
        $decisionProperties -notcontains "reason") {
        return $null
    }
    if ([string]$decision.status -ne "AWAITING_VISUAL_QC" -or
        [string]$decision.code -ne "THREEDGRUT_SMOKE_AWAITING_VISUAL_QC" -or
        -not $decision.reason) {
        return $null
    }
    $stageReport = if ($decision.PSObject.Properties.Name -contains "evidence" -and
        $decision.evidence -and $decision.evidence.PSObject.Properties.Name -contains "stage_report_path") {
        [string]$decision.evidence.stage_report_path
    } else { $null }
    return [pscustomobject][ordered]@{
        path = $decisionPath
        status = [string]$decision.status
        code = [string]$decision.code
        reason = [string]$decision.reason
        stage_report_path = $stageReport
    }
}

function Apply-ProfileDefaults {
    if ($customMode.Checked) { return }
    $sceneType = Get-SceneType
    if (-not $sceneType) { return }
    if ($previewMode.Checked) {
        $framesInput.Value = if ($sceneType -eq "Object") { 150 } else { 300 }
        $stepsInput.Value = if ($sceneType -eq "Object") { 7000 } else { 10000 }
        $fastExtractionCheck.Checked = $true
    } else {
        $framesInput.Value = if ($sceneType -eq "Object") { 300 } else { 1200 }
        $stepsInput.Value = if ($sceneType -eq "Object") { 30000 } else { 40000 }
        $fastExtractionCheck.Checked = $false
    }
    $stepsInput.Enabled = $false
}

$previewMode.Add_CheckedChanged({ if ($previewMode.Checked) { Apply-ProfileDefaults } })
$finalMode.Add_CheckedChanged({ if ($finalMode.Checked) { Apply-ProfileDefaults } })
$sceneTypeInput.Add_SelectedIndexChanged({
    Apply-ProfileDefaults
    if (-not $script:ActiveProcess) {
        $startButton.Enabled = (Test-SelectedVideos) -and [bool](Get-SceneType)
    }
})
$customMode.Add_CheckedChanged({
    if ($customMode.Checked) { $stepsInput.Enabled = $true }
})
$buildBlenderCheck.Add_CheckedChanged({
    $openBlenderCheck.Enabled = $buildBlenderCheck.Checked
    if (-not $buildBlenderCheck.Checked) { $openBlenderCheck.Checked = $false }
})
Apply-ProfileDefaults

function Set-Videos([string[]]$Paths) {
    $invalid = @($Paths | Where-Object { -not (Test-VideoPath $_) })
    if ($Paths.Count -lt 1 -or $invalid.Count -gt 0) {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Choose only supported video files: MOV, MP4, M4V, AVI, or MKV.",
            "Unsupported file",
            "OK",
            "Warning"
        ) | Out-Null
        return
    }
    $script:SelectedVideoPaths = @($Paths | ForEach-Object { (Get-Item -LiteralPath $_).FullName })
    $script:QueuePath = $null
    if ($script:SelectedVideoPaths.Count -eq 1) {
        $videoPathBox.Text = $script:SelectedVideoPaths[0]
        $runNameBox.Text = Get-RunName $script:SelectedVideoPaths[0]
        $runNameBox.Enabled = $true
        $startButton.Text = "Start Pipeline"
    } else {
        $videoPathBox.Text = "$($script:SelectedVideoPaths.Count) videos selected for the production queue"
        $runNameBox.Text = "automatic-name-plus-source-hash"
        $runNameBox.Enabled = $false
        $startButton.Text = "Start $($script:SelectedVideoPaths.Count) Videos"
    }
    $startButton.Enabled = [bool](Get-SceneType)
    $statusLabel.Text = if (Get-SceneType) { "$($script:SelectedVideoPaths.Count) video(s) ready" } else { "Choose capture type" }
    $statusLabel.ForeColor = New-Color "#1D2329"
}

function Set-Video([string]$Path) { Set-Videos @($Path) }

function Set-RunningState([bool]$Running) {
    $startButton.Enabled = -not $Running -and (Test-SelectedVideos) -and [bool](Get-SceneType)
    $cancelButton.Enabled = $Running
    $browseButton.Enabled = -not $Running
    $runNameBox.Enabled = -not $Running -and $script:SelectedVideoPaths.Count -eq 1
    $framesInput.Enabled = -not $Running
    $stepsInput.Enabled = -not $Running -and $customMode.Checked
    $trainerInput.Enabled = -not $Running
    $sceneTypeInput.Enabled = -not $Running
    $previewMode.Enabled = -not $Running
    $finalMode.Enabled = -not $Running
    $customMode.Enabled = -not $Running
    $fastExtractionCheck.Enabled = -not $Running
    $autoRotateCheck.Enabled = -not $Running
    $openViewerCheck.Enabled = -not $Running
    $buildBlenderCheck.Enabled = -not $Running
    $openBlenderCheck.Enabled = -not $Running -and $buildBlenderCheck.Checked
    $dropPanel.Enabled = -not $Running
    if ($Running) {
        $progress.Style = "Marquee"
        $progress.MarqueeAnimationSpeed = 25
    } else {
        $progress.MarqueeAnimationSpeed = 0
        $progress.Style = "Continuous"
    }
}

function Read-NewLogText {
    if (-not $script:LogReader -and $script:PipelineLogPath -and (Test-Path -LiteralPath $script:PipelineLogPath)) {
        $script:LogStream = New-Object System.IO.FileStream(
            $script:PipelineLogPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        $script:LogReader = New-Object System.IO.StreamReader($script:LogStream)
    }
    if (-not $script:LogReader) { return }

    $chunk = $script:LogReader.ReadToEnd()
    if (-not $chunk) { return }
    $logBox.AppendText($chunk)
    $logBox.SelectionStart = $logBox.TextLength
    $logBox.ScrollToCaret()

    foreach ($line in ($chunk -split "`r?`n")) {
        if ($line -match "^\[(extract|select|solve|train|blender|view)\]") {
            $stage = $Matches[1]
            $statusLabel.Text = "Working: " + (Get-Culture).TextInfo.ToTitleCase($stage)
        }
        if ($line -match "^PLY:\s+(.+\.ply)\s*$") {
            $script:FinalPly = $Matches[1].Trim()
        }
        if ($line -match "^BLEND:\s+(.+\.blend)\s*$") {
            $script:BlenderFile = $Matches[1].Trim()
        }
    }
}

function Close-LogReader {
    if ($script:LogReader) {
        $script:LogReader.Dispose()
        $script:LogReader = $null
    }
    if ($script:LogStream) {
        $script:LogStream.Dispose()
        $script:LogStream = $null
    }
}

function Start-Pipeline {
    if (-not (Test-SelectedVideos)) { return }
    $isBatch = $script:SelectedVideoPaths.Count -gt 1
    $selectedSceneType = Get-SceneType
    if (-not $selectedSceneType) {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Choose Auto, Object / vehicle orbit, Walkthrough, House closed loop, or Aerial exterior before starting.",
            "Capture type required",
            "OK",
            "Warning"
        ) | Out-Null
        return
    }
    if (-not $isBatch -and $runNameBox.Text -notmatch "^[a-zA-Z0-9][a-zA-Z0-9_-]*$") {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Run name may contain only letters, numbers, hyphens, and underscores.",
            "Invalid run name",
            "OK",
            "Warning"
        ) | Out-Null
        return
    }
    $route = Get-PipelineRoute
    $selectedRunnerPath = if ($isBatch) { $BatchRunnerPath } else { [string]$route.runner_path }
    if (-not (Test-Path -LiteralPath $selectedRunnerPath -PathType Leaf)) {
        [System.Windows.Forms.MessageBox]::Show($form, "Pipeline runner not found:`n$selectedRunnerPath", "Missing runner", "OK", "Error") | Out-Null
        return
    }
    $selectedTrainer = [string]$trainerInput.SelectedItem
    $manualTrainerStop = if ($route.mode -eq "Manual") {
        Get-ManualTrainerStop ([string]$route.scene_type) $selectedTrainer
    } else { $null }
    if ($manualTrainerStop) {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Training was not started.`r`n`r`nCode: $($manualTrainerStop.code)`r`nReason: $($manualTrainerStop.reason)",
            "Unsupported experimental trainer",
            "OK",
            "Information"
        ) | Out-Null
        $statusLabel.Text = "$($manualTrainerStop.code) - $($manualTrainerStop.reason)"
        $statusLabel.ForeColor = New-Color "#A34E18"
        return
    }
    if ($route.mode -eq "Manual" -and $selectedTrainer -in @("3DGUT", "3DGUT-MCMC") -and -not (Test-ThreeDGRUTConfiguration)) {
        $runtimeDetail = if ($script:ThreeDGrutConfigurationError) { "`n`n$($script:ThreeDGrutConfigurationError)" } else { "" }
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "3DGRUT is not installed or configured yet. Run scripts\doctor.ps1 after completing its portable runtime paths, or use Brush.$runtimeDetail",
            "3DGRUT unavailable",
            "OK",
            "Information"
        ) | Out-Null
        return
    }

    Close-LogReader
    $script:CompletionHandled = $false
    $script:FinalPly = $null
    $script:BlenderFile = $null
    $script:ActiveIsBatch = $isBatch
    $script:QueuePath = $null
    if ($isBatch) {
        $queueRoot = Join-Path $OutputRoot ".queues"
        New-Item -ItemType Directory -Force -Path $queueRoot | Out-Null
        $script:QueuePath = Join-Path $queueRoot "batch-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))-$([guid]::NewGuid().ToString('N').Substring(0,8)).json"
        $script:RunRoot = $OutputRoot
    } else {
        $script:RunRoot = Join-Path $OutputRoot $runNameBox.Text
    }
    $script:ActiveRouteMode = if ($isBatch) { "Batch" } else { [string]$route.mode }
    $script:PipelineStartedUtc = [datetime]::UtcNow
    $script:AutoDecisionBaseline = if ($script:ActiveRouteMode -eq "Auto") {
        Get-FileVersion (Join-Path $script:RunRoot "auto_decision.json")
    } else {
        $null
    }
    $script:TrainingDecisionBaseline = Get-FileVersion (Join-Path $script:RunRoot "training_decision.json")
    $logsRoot = if ($isBatch) { Join-Path $OutputRoot ".queues\logs" } else { Join-Path $script:RunRoot "logs" }
    New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
    $script:PipelineLogPath = Join-Path $logsRoot "splatitup888-$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ'))-$([guid]::NewGuid().ToString('N').Substring(0,8)).log"

    $runner = ConvertTo-SingleQuotedPowerShell $selectedRunnerPath
    $trainer = ConvertTo-SingleQuotedPowerShell $selectedTrainer
    $logPath = ConvertTo-SingleQuotedPowerShell $script:PipelineLogPath
    $candidateMultiplier = if ($finalMode.Checked) { 8 } else { 2 }
    $maxCumulativeFlow = if ($finalMode.Checked) { "0.0125" } else { "0" }
    if ($isBatch) {
        $videos = "@(" + (($script:SelectedVideoPaths | ForEach-Object { ConvertTo-SingleQuotedPowerShell $_ }) -join ",") + ")"
        $queuePath = ConvertTo-SingleQuotedPowerShell $script:QueuePath
        $captureType = ConvertTo-SingleQuotedPowerShell $(if ($route.mode -eq "Auto") { "Auto" } else { [string]$route.scene_type })
        $mode = ConvertTo-SingleQuotedPowerShell (Get-AutoPreset)
        $command = "& $runner -VideoPaths $videos -QueuePath $queuePath -CaptureType $captureType -Mode $mode"
    } else {
        $video = ConvertTo-SingleQuotedPowerShell $script:SelectedVideoPaths[0]
        $runName = ConvertTo-SingleQuotedPowerShell $runNameBox.Text
        $command = "& $runner -VideoPath $video -RunName $runName"
        if ($route.scene_type) {
            $sceneType = ConvertTo-SingleQuotedPowerShell ([string]$route.scene_type)
            $command += " -SceneType $sceneType"
        }
        if ($route.mode -eq "Auto") {
            $autoPreset = ConvertTo-SingleQuotedPowerShell (Get-AutoPreset)
            $command += " -AutoPreset $autoPreset"
        }
    }
    $command += " -SelectedFrames $([int]$framesInput.Value) -TrainingSteps $([int]$stepsInput.Value) -TrainingMaxResolution 1920 -Trainer $trainer -CandidateMultiplier $candidateMultiplier -MaxCumulativeFlow $maxCumulativeFlow"
    if (-not $isBatch) { $command += " -AdaptiveExtraction" }
    if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
        $configArgument = ConvertTo-SingleQuotedPowerShell $ConfigPath
        $command += " -ConfigPath $configArgument"
    }
    if ($fastExtractionCheck.Checked) { $command += " -MaxLongSide 1600" }
    if (-not $autoRotateCheck.Checked) { $command += " -NoAutoRotate" }
    if (-not $buildBlenderCheck.Checked) { $command += " -NoBlender" }
    if (-not $isBatch -and $buildBlenderCheck.Checked -and $openBlenderCheck.Checked) { $command += " -OpenBlender" }
    if (-not $isBatch -and -not $openViewerCheck.Checked) { $command += " -NoBrowser" }
    $command += " *> $logPath; `$pipelineSucceeded = `$?; `$pipelineExitCode = `$LASTEXITCODE; if (-not `$pipelineSucceeded -and `$pipelineExitCode -eq 2) { exit 2 }; if (`$pipelineSucceeded) { exit 0 }; exit 1"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "powershell.exe"
    $startInfo.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    $logBox.Clear()
    $modeName = if ($previewMode.Checked) { "Preview" } elseif ($finalMode.Checked) { "Final" } else { "Custom" }
    $startLabel = if ($isBatch) { "$($script:SelectedVideoPaths.Count)-video queue" } else { $runNameBox.Text }
    $logBox.AppendText("Starting $startLabel | $selectedSceneType | $modeName | $selectedTrainer...`r`n")
    $outputLink.Text = if ($isBatch) { $script:QueuePath } else { $script:RunRoot }
    $outputLink.Enabled = $true
    $openFolderButton.Enabled = $true
    $statusLabel.Text = "Starting"
    $statusLabel.ForeColor = New-Color "#1D2329"
    Set-RunningState $true

    try {
        $script:ActiveProcess = [System.Diagnostics.Process]::Start($startInfo)
    } catch {
        Set-RunningState $false
        $statusLabel.Text = "Could not start"
        $logBox.AppendText($_.Exception.Message + "`r`n")
        $script:ActiveProcess = $null
        $script:ActiveRouteMode = $null
        $script:ActiveIsBatch = $false
        $script:PipelineStartedUtc = $null
        $script:AutoDecisionBaseline = $null
        $script:TrainingDecisionBaseline = $null
    }
}

function Complete-Pipeline {
    if (-not $script:ActiveProcess -or $script:CompletionHandled) { return }
    $script:CompletionHandled = $true
    $script:ActiveProcess.WaitForExit()
    Read-NewLogText
    $exitCode = $script:ActiveProcess.ExitCode
    Close-LogReader
    Set-RunningState $false
    if ($script:ActiveIsBatch) {
        $queue = $null
        if ($script:QueuePath -and (Test-Path -LiteralPath $script:QueuePath -PathType Leaf)) {
            try { $queue = Get-Content -LiteralPath $script:QueuePath -Raw | ConvertFrom-Json } catch { $queue = $null }
        }
        if ($queue) {
            $complete = @($queue.jobs | Where-Object { $_.status -eq "COMPLETE" }).Count
            $review = @($queue.jobs | Where-Object { $_.status -eq "REVIEW_REQUIRED" }).Count
            $failed = @($queue.jobs | Where-Object { $_.status -eq "FAILED" }).Count
            $statusLabel.Text = "Queue $($queue.status): $complete complete, $review review, $failed failed"
            $statusLabel.ForeColor = if ($failed -eq 0 -and $review -eq 0) { New-Color "#26734D" } else { New-Color "#A34E18" }
            $progress.Value = if ($failed -eq 0 -and $review -eq 0) { 100 } else { 0 }
            $outputLink.Text = $script:QueuePath
            $outputLink.Enabled = $true
        } else {
            $statusLabel.Text = "Queue stopped; queue state is missing or unreadable"
            $statusLabel.ForeColor = New-Color "#A34E18"
            $progress.Value = 0
        }
        $script:ActiveProcess.Dispose()
        $script:ActiveProcess = $null
        $script:ActiveRouteMode = $null
        $script:ActiveIsBatch = $false
        $script:PipelineStartedUtc = $null
        $script:AutoDecisionBaseline = $null
        $script:TrainingDecisionBaseline = $null
        return
    }
    $trainingDecision = Get-CurrentTrainingDecision

    if ($exitCode -eq 0) {
        $script:FinalPly = $null
        $script:BlenderFile = $null
        $manifest = $null
        $manifestPath = Join-Path $script:RunRoot "run_manifest.json"
        if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
            try { $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json } catch { $manifest = $null }
        }
        function Resolve-CurrentArtifact($Artifact) {
            if (-not $Artifact -or -not $Artifact.path -or -not $Artifact.sha256) { return $null }
            $path = [string]$Artifact.path
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
            $file = Get-Item -LiteralPath $path
            if ($Artifact.PSObject.Properties.Name -contains "bytes" -and [int64]$Artifact.bytes -ne $file.Length) { return $null }
            $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($sha -ne [string]$Artifact.sha256) { return $null }
            return $path
        }
        if ($manifest -and [int]$manifest.schema_version -eq 3) {
            $script:FinalPly = Resolve-CurrentArtifact $manifest.artifacts.gaussian_ply
            $script:BlenderFile = Resolve-CurrentArtifact $manifest.artifacts.blender_scene
        }
        $statusLabel.Text = if ($script:BlenderFile) {
            "Mechanical pass - review the result in Blender"
        } elseif ($script:FinalPly) {
            "Mechanical pass - review the splat"
        } else {
            "Complete - no current artifact"
        }
        $reportPath = if ($manifest -and $manifest.reports.reconstruction) { [string]$manifest.reports.reconstruction } else { $null }
        if ($reportPath -and (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
            $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            $registration = [double]$report.registration_percent
            $points = [double]$report.points
            $errorPixels = [double]$report.mean_reprojection_error_pixels
            $prefix = if ($script:BlenderFile) { "Blender mechanical pass" } else { "Mechanical pass" }
            $statusLabel.Text = "$prefix`: $([Math]::Round($registration, 1))% registered | $([Math]::Round($points)) points | $([Math]::Round($errorPixels, 2)) px"
            if ($report.quality_gates.overall_pass) {
                $statusLabel.ForeColor = New-Color "#26734D"
            } else {
                $statusLabel.ForeColor = New-Color "#A34E18"
            }
        }
        $progress.Value = 100
        if ($script:BlenderFile) {
            $outputLink.Text = $script:BlenderFile
            [System.Media.SystemSounds]::Asterisk.Play()
        } elseif ($script:FinalPly) {
            $outputLink.Text = $script:FinalPly
            [System.Media.SystemSounds]::Asterisk.Play()
        } else {
            $outputLink.Text = $script:RunRoot
        }
    } elseif ($exitCode -eq 2 -and $trainingDecision) {
        $reviewMessage = "1K smoke stopped for review: $($trainingDecision.code) - $($trainingDecision.reason)"
        $statusLabel.Text = $reviewMessage
        $statusLabel.ForeColor = New-Color "#A34E18"
        $logBox.AppendText("`r`n$reviewMessage`r`nDecision: $($trainingDecision.path)`r`n")
        if ($trainingDecision.stage_report_path -and (Test-Path -LiteralPath $trainingDecision.stage_report_path -PathType Leaf)) {
            $outputLink.Text = $trainingDecision.stage_report_path
        } else {
            $outputLink.Text = $trainingDecision.path
        }
        $outputLink.Enabled = $true
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "3DGRUT-MCMC stopped after the exact 1K/250K smoke.`r`n`r`nNo 7K, final, or publish stage ran.`r`n`r`nDecision: $($trainingDecision.path)",
            "3DGRUT smoke review required",
            "OK",
            "Information"
        ) | Out-Null
        $progress.Value = 0
        [System.Media.SystemSounds]::Hand.Play()
    } elseif ($exitCode -eq 2 -and $script:ActiveRouteMode -eq "Auto") {
        $decision = Get-CurrentAutoStopDecision
        if ($decision) {
            $stopMessage = "Stopped before training: $($decision.code) - $($decision.reason)"
            $statusLabel.Text = $stopMessage
            $statusLabel.ForeColor = New-Color "#A34E18"
            $logBox.AppendText("`r`n$stopMessage`r`nDecision: $($decision.path)`r`n")
            $outputLink.Text = $decision.path
            [System.Windows.Forms.MessageBox]::Show(
                $form,
                "Stopped before training.`r`n`r`nCode: $($decision.code)`r`nReason: $($decision.reason)",
                "Auto capture check",
                "OK",
                "Information"
            ) | Out-Null
        } else {
            $statusLabel.Text = "Auto stopped before training - current decision evidence is missing"
            $statusLabel.ForeColor = New-Color "#A34E18"
            $logBox.AppendText("`r`nAuto stopped before training, but RunRoot\auto_decision.json was missing, stale, or invalid.`r`n")
        }
        $progress.Value = 0
        [System.Media.SystemSounds]::Hand.Play()
    } else {
        $statusLabel.Text = "Stopped with an error"
        $progress.Value = 0
        [System.Media.SystemSounds]::Hand.Play()
    }
    $script:ActiveProcess.Dispose()
    $script:ActiveProcess = $null
    $script:ActiveRouteMode = $null
    $script:PipelineStartedUtc = $null
    $script:AutoDecisionBaseline = $null
    $script:TrainingDecisionBaseline = $null
}

$pollTimer = New-Object System.Windows.Forms.Timer
$pollTimer.Interval = 300
$pollTimer.Add_Tick({
    Read-NewLogText
    if ($script:ActiveProcess -and $script:ActiveProcess.HasExited) {
        Complete-Pipeline
    }
})
$pollTimer.Start()

$browseButton.Add_Click({
    $dialog = New-Object System.Windows.Forms.OpenFileDialog
    $dialog.Title = "Choose capture video"
    $dialog.Filter = "Video files|*.mov;*.mp4;*.m4v;*.avi;*.mkv|All files|*.*"
    $dialog.Multiselect = $true
    if ($dialog.ShowDialog($form) -eq "OK") { Set-Videos $dialog.FileNames }
    $dialog.Dispose()
})

$dragEnterHandler = {
    param($sender, $eventArgs)
    if ($eventArgs.Data.GetDataPresent([System.Windows.Forms.DataFormats]::FileDrop)) {
        $files = [string[]]$eventArgs.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)
        if ($files.Count -gt 0 -and @($files | Where-Object { -not (Test-VideoPath $_) }).Count -eq 0) {
            $eventArgs.Effect = [System.Windows.Forms.DragDropEffects]::Copy
            return
        }
    }
    $eventArgs.Effect = [System.Windows.Forms.DragDropEffects]::None
}

$dragDropHandler = {
    param($sender, $eventArgs)
    $files = [string[]]$eventArgs.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)
    if ($files.Count -gt 0) { Set-Videos $files }
}

$form.Add_DragEnter($dragEnterHandler)
$form.Add_DragDrop($dragDropHandler)
$dropPanel.Add_DragEnter($dragEnterHandler)
$dropPanel.Add_DragDrop($dragDropHandler)
$startButton.Add_Click({ Start-Pipeline })

$cancelButton.Add_Click({
    if (-not $script:ActiveProcess -or $script:ActiveProcess.HasExited) { return }
    $choice = [System.Windows.Forms.MessageBox]::Show(
        $form,
        "Stop the current pipeline? Completed stages will remain resumable.",
        "Stop pipeline",
        "YesNo",
        "Question"
    )
    if ($choice -eq "Yes") {
        Start-Process -FilePath "taskkill.exe" -ArgumentList "/PID $($script:ActiveProcess.Id) /T /F" -WindowStyle Hidden -Wait | Out-Null
        $statusLabel.Text = "Cancelled"
    }
})

$outputLink.Add_LinkClicked({
    $target = if ($script:BlenderFile) { $script:BlenderFile } elseif ($script:FinalPly) { $script:FinalPly } elseif ($script:QueuePath) { $script:QueuePath } else { $script:RunRoot }
    if ($target -and (Test-Path -LiteralPath $target)) {
        if (Test-Path -LiteralPath $target -PathType Leaf) {
            Start-Process -FilePath "explorer.exe" -ArgumentList "/select,`"$target`"" | Out-Null
        } else {
            Start-Process -FilePath "explorer.exe" -ArgumentList "`"$target`"" | Out-Null
        }
    }
})

$openFolderButton.Add_Click({
    $target = if ($script:RunRoot) { $script:RunRoot } else { $OutputRoot }
    if (Test-Path -LiteralPath $target) {
        Start-Process -FilePath "explorer.exe" -ArgumentList "`"$target`"" | Out-Null
    }
})

$form.Add_FormClosing({
    param($sender, $eventArgs)
    if ($script:ActiveProcess -and -not $script:ActiveProcess.HasExited) {
        $choice = [System.Windows.Forms.MessageBox]::Show(
            $form,
            "The pipeline is still running. Close the app and stop it?",
            "Pipeline running",
            "YesNo",
            "Warning"
        )
        if ($choice -ne "Yes") {
            $eventArgs.Cancel = $true
            return
        }
        Start-Process -FilePath "taskkill.exe" -ArgumentList "/PID $($script:ActiveProcess.Id) /T /F" -WindowStyle Hidden -Wait | Out-Null
    }
    $pollTimer.Stop()
    Close-LogReader
})

if ($SmokeTest) {
    $initialSceneType = Get-SceneType
    $initialFrames = [int]$framesInput.Value
    $initialTrainingSteps = [int]$stepsInput.Value
    $initialMode = if ($finalMode.Checked) { "Final" } else { "Other" }
    $captureChoices = @($sceneTypeInput.Items | ForEach-Object { [string]$_ })
    $trainerChoices = @($trainerInput.Items | ForEach-Object { [string]$_ })
    $routeTable = [ordered]@{}
    foreach ($routeSpec in @(
        [pscustomobject]@{ label = "auto"; index = 1 },
        [pscustomobject]@{ label = "walkthrough"; index = 2 },
        [pscustomobject]@{ label = "house"; index = 3 },
        [pscustomobject]@{ label = "object"; index = 4 },
        [pscustomobject]@{ label = "aerial_exterior"; index = 5 }
    )) {
        $sceneTypeInput.SelectedIndex = $routeSpec.index
        [System.Windows.Forms.Application]::DoEvents()
        $route = Get-PipelineRoute
        $routeTable[$routeSpec.label] = [ordered]@{
            mode = [string]$route.mode
            runner = [System.IO.Path]::GetFileName([string]$route.runner_path)
            scene_type = $route.scene_type
            emits_scene_type = [bool]$route.scene_type
            emits_auto_preset = [string]$route.mode -eq "Auto"
        }
    }

    $sceneTypeInput.SelectedIndex = 1
    $previewMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    $autoPreview = [ordered]@{
        auto_preset = Get-AutoPreset
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        max_long_side = if ($fastExtractionCheck.Checked) { 1600 } else { 0 }
    }
    $finalMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    $autoFinal = [ordered]@{
        auto_preset = Get-AutoPreset
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        max_long_side = if ($fastExtractionCheck.Checked) { 1600 } else { 0 }
    }
    $customMode.Checked = $true
    $framesInput.Value = 777
    $stepsInput.Value = 25000
    [System.Windows.Forms.Application]::DoEvents()
    $autoCustom = [ordered]@{
        auto_preset = Get-AutoPreset
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
    }

    $sceneTypeInput.SelectedIndex = 4
    $previewMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    $objectPreview = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        adaptive_extraction = $true
        candidate_multiplier = 2
        max_cumulative_flow = 0.0
        max_long_side = if ($fastExtractionCheck.Checked) { 1600 } else { 0 }
    }
    $finalMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    $objectFinal = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        candidate_multiplier = 8
        max_cumulative_flow = 0.0125
    }
    $sceneTypeInput.SelectedIndex = 2
    [System.Windows.Forms.Application]::DoEvents()
    $walkthroughFinal = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        candidate_multiplier = 8
        max_cumulative_flow = 0.0125
    }
    $sceneTypeInput.SelectedIndex = 3
    [System.Windows.Forms.Application]::DoEvents()
    $houseFinal = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        candidate_multiplier = 8
        max_cumulative_flow = 0.0125
    }
    $sceneTypeInput.SelectedIndex = 5
    [System.Windows.Forms.Application]::DoEvents()
    $aerialExteriorFinal = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        candidate_multiplier = 8
        max_cumulative_flow = 0.0125
    }
    $sceneTypeInput.SelectedIndex = 1
    [System.Windows.Forms.Application]::DoEvents()
    [ordered]@{
        app = $form.Text
        runner_found = Test-Path -LiteralPath $RunnerPath -PathType Leaf
        core_runner_found = Test-Path -LiteralPath $RunnerPath -PathType Leaf
        auto_runner_found = Test-Path -LiteralPath $AutoRunnerPath -PathType Leaf
        batch_runner_found = Test-Path -LiteralPath $BatchRunnerPath -PathType Leaf
        multi_video_queue_enabled = $true
        output_root = $OutputRoot
        drag_drop_enabled = $form.AllowDrop -and $dropPanel.AllowDrop
        default_frames = $initialFrames
        default_training_steps = $initialTrainingSteps
        default_trainer = [string]$trainerInput.SelectedItem
        trainer_choices = $trainerChoices
        manual_trainer_policy = [ordered]@{
            object_mcmc = Get-ManualTrainerStop "Object" "3DGUT-MCMC"
            walkthrough_mcmc = Get-ManualTrainerStop "Walkthrough" "3DGUT-MCMC"
            house_mcmc = Get-ManualTrainerStop "House" "3DGUT-MCMC"
            aerial_exterior_mcmc = Get-ManualTrainerStop "AerialExterior" "3DGUT-MCMC"
        }
        default_scene_type = $initialSceneType
        capture_selection_required = $null -eq $initialSceneType
        capture_choices = $captureChoices
        default_mode = $initialMode
        route_table = $routeTable
        adaptive_extraction = $true
        candidate_multiplier = 8
        max_cumulative_flow = 0.0125
        max_long_side = if ($fastExtractionCheck.Checked) { 1600 } else { 0 }
        autorotate = $autoRotateCheck.Checked
        build_blender = $buildBlenderCheck.Checked
        open_blender = $openBlenderCheck.Checked
        preview = $objectPreview
        auto_preview = $autoPreview
        auto_final = $autoFinal
        auto_custom = $autoCustom
        object_preview = $objectPreview
        object_final = $objectFinal
        walkthrough_final = $walkthroughFinal
        house_final = $houseFinal
        aerial_exterior_final = $aerialExteriorFinal
        control_count = $form.Controls.Count
    } | ConvertTo-Json -Compress
    $pollTimer.Stop()
    $form.Dispose()
    exit 0
}

$form.Add_FormClosed({ [System.Windows.Forms.Application]::ExitThread() })
$form.Show()
$form.Activate()
$form.Refresh()
if ($ScreenshotPath) {
    $destination = [System.IO.Path]::GetFullPath($ScreenshotPath)
    $parent = Split-Path -Parent $destination
    if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
    $bitmap = New-Object System.Drawing.Bitmap($form.Width, $form.Height)
    $form.DrawToBitmap($bitmap, (New-Object System.Drawing.Rectangle(0, 0, $form.Width, $form.Height)))
    $bitmap.Save($destination, [System.Drawing.Imaging.ImageFormat]::Png)
    $bitmap.Dispose()
    $form.Close()
    exit 0
}
[System.Windows.Forms.Application]::Run()
