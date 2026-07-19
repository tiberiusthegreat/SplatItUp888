[CmdletBinding()]
param(
    [switch]$SmokeTest,
    [string]$ScreenshotPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

[System.Windows.Forms.Application]::EnableVisualStyles()

$RunnerPath = Join-Path $PSScriptRoot "pipeline\run_video_to_splat.ps1"
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
$script:RunRoot = $null

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

function Test-ThreeDGRUTConfiguration {
    if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) { return $false }
    $config = Import-PowerShellDataFile -LiteralPath $ConfigPath
    if (-not $config.ContainsKey("ThreeDGRUT")) { return $false }
    $repo = [string]$config.ThreeDGRUT.Repo
    $python = [string]$config.ThreeDGRUT.Python
    if (-not $repo -or -not $python) { return $false }
    return (Test-Path -LiteralPath $repo -PathType Container) -and
        (Test-Path -LiteralPath $python -PathType Leaf)
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "SplatItUp888"
$form.StartPosition = "CenterScreen"
$form.ClientSize = New-Object System.Drawing.Size(820, 760)
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
$subtitleLabel.Text = "VIDEO  >  FRAMES  >  COLMAP  >  3DGS  >  PLY"
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
$dropLabel.Text = "Drop orbit video"
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
$settingsGroup.Size = New-Object System.Drawing.Size(772, 150)
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
$framesLabel.Text = "Selected frames"
$framesLabel.AutoSize = $true
$framesLabel.Location = New-Object System.Drawing.Point(252, 24)
$settingsGroup.Controls.Add($framesLabel)

$framesInput = New-Object System.Windows.Forms.NumericUpDown
$framesInput.Location = New-Object System.Drawing.Point(254, 46)
$framesInput.Size = New-Object System.Drawing.Size(90, 25)
$framesInput.Minimum = 40
$framesInput.Maximum = 500
$framesInput.Increment = 10
$framesInput.Value = 180
$settingsGroup.Controls.Add($framesInput)

$stepsLabel = New-Object System.Windows.Forms.Label
$stepsLabel.Text = "Training steps"
$stepsLabel.AutoSize = $true
$stepsLabel.Location = New-Object System.Drawing.Point(350, 84)
$settingsGroup.Controls.Add($stepsLabel)

$stepsInput = New-Object System.Windows.Forms.NumericUpDown
$stepsInput.Location = New-Object System.Drawing.Point(352, 106)
$stepsInput.Size = New-Object System.Drawing.Size(120, 25)
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
[void]$trainerInput.Items.AddRange([object[]]@("Brush", "3DGUT", "3DGUT-MCMC"))
$trainerInput.SelectedIndex = 0
$settingsGroup.Controls.Add($trainerInput)

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
$fastExtractionCheck.Text = "Fast 1600px extraction"
$fastExtractionCheck.AutoSize = $true
$fastExtractionCheck.Location = New-Object System.Drawing.Point(500, 86)
$settingsGroup.Controls.Add($fastExtractionCheck)

$autoRotateCheck = New-Object System.Windows.Forms.CheckBox
$autoRotateCheck.Text = "Autorotate phone video"
$autoRotateCheck.AutoSize = $true
$autoRotateCheck.Location = New-Object System.Drawing.Point(500, 112)
$autoRotateCheck.Checked = $true
$settingsGroup.Controls.Add($autoRotateCheck)

$openViewerCheck = New-Object System.Windows.Forms.CheckBox
$openViewerCheck.Text = "Open SuperSplat"
$openViewerCheck.AutoSize = $true
$openViewerCheck.Location = New-Object System.Drawing.Point(566, 48)
$openViewerCheck.Checked = $true
$settingsGroup.Controls.Add($openViewerCheck)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text = "Choose a video"
$statusLabel.AutoSize = $true
$statusLabel.Location = New-Object System.Drawing.Point(24, 396)
$statusLabel.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 9)
$form.Controls.Add($statusLabel)

$startButton = New-Object System.Windows.Forms.Button
$startButton.Text = "Start Pipeline"
$startButton.Location = New-Object System.Drawing.Point(580, 388)
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
$cancelButton.Location = New-Object System.Drawing.Point(712, 388)
$cancelButton.Size = New-Object System.Drawing.Size(84, 34)
$cancelButton.Anchor = "Top, Right"
$cancelButton.FlatStyle = "Flat"
$cancelButton.BackColor = [System.Drawing.Color]::White
$cancelButton.FlatAppearance.BorderColor = New-Color "#B9C1C8"
$cancelButton.Enabled = $false
$form.Controls.Add($cancelButton)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Location = New-Object System.Drawing.Point(24, 433)
$progress.Size = New-Object System.Drawing.Size(772, 8)
$progress.Anchor = "Top, Left, Right"
$progress.Style = "Continuous"
$form.Controls.Add($progress)

$logBox = New-Object System.Windows.Forms.TextBox
$logBox.Location = New-Object System.Drawing.Point(24, 456)
$logBox.Size = New-Object System.Drawing.Size(772, 220)
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
$outputCaption.Location = New-Object System.Drawing.Point(24, 694)
$outputCaption.ForeColor = New-Color "#68727D"
$form.Controls.Add($outputCaption)

$outputLink = New-Object System.Windows.Forms.LinkLabel
$outputLink.Text = "Not created yet"
$outputLink.AutoEllipsis = $true
$outputLink.Location = New-Object System.Drawing.Point(24, 718)
$outputLink.Size = New-Object System.Drawing.Size(650, 24)
$outputLink.Anchor = "Bottom, Left, Right"
$outputLink.LinkColor = New-Color "#1E5F86"
$outputLink.ActiveLinkColor = New-Color "#D85D27"
$outputLink.Enabled = $false
$form.Controls.Add($outputLink)

$openFolderButton = New-Object System.Windows.Forms.Button
$openFolderButton.Text = "Open Folder"
$openFolderButton.Location = New-Object System.Drawing.Point(692, 708)
$openFolderButton.Size = New-Object System.Drawing.Size(104, 34)
$openFolderButton.Anchor = "Bottom, Right"
$openFolderButton.FlatStyle = "Flat"
$openFolderButton.BackColor = [System.Drawing.Color]::White
$openFolderButton.FlatAppearance.BorderColor = New-Color "#B9C1C8"
$openFolderButton.Enabled = $false
$form.Controls.Add($openFolderButton)

$previewMode.Add_CheckedChanged({
    if ($previewMode.Checked) {
        $stepsInput.Value = 7000
        $framesInput.Value = 150
        $stepsInput.Enabled = $false
        $fastExtractionCheck.Checked = $true
    }
})
$finalMode.Add_CheckedChanged({
    if ($finalMode.Checked) {
        $stepsInput.Value = 30000
        $framesInput.Value = 180
        $stepsInput.Enabled = $false
        $fastExtractionCheck.Checked = $false
    }
})
$customMode.Add_CheckedChanged({
    if ($customMode.Checked) { $stepsInput.Enabled = $true }
})

function Set-Video([string]$Path) {
    if (-not (Test-VideoPath $Path)) {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Choose a supported video file: MOV, MP4, M4V, AVI, or MKV.",
            "Unsupported file",
            "OK",
            "Warning"
        ) | Out-Null
        return
    }
    $resolved = (Get-Item -LiteralPath $Path).FullName
    $videoPathBox.Text = $resolved
    $runNameBox.Text = Get-RunName $resolved
    $startButton.Enabled = $true
    $statusLabel.Text = "Ready"
    $statusLabel.ForeColor = New-Color "#1D2329"
}

function Set-RunningState([bool]$Running) {
    $startButton.Enabled = -not $Running -and (Test-VideoPath $videoPathBox.Text)
    $cancelButton.Enabled = $Running
    $browseButton.Enabled = -not $Running
    $runNameBox.Enabled = -not $Running
    $framesInput.Enabled = -not $Running
    $stepsInput.Enabled = -not $Running -and $customMode.Checked
    $trainerInput.Enabled = -not $Running
    $previewMode.Enabled = -not $Running
    $finalMode.Enabled = -not $Running
    $customMode.Enabled = -not $Running
    $fastExtractionCheck.Enabled = -not $Running
    $autoRotateCheck.Enabled = -not $Running
    $openViewerCheck.Enabled = -not $Running
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
        if ($line -match "^\[(extract|select|solve|train|view)\]") {
            $stage = $Matches[1]
            $statusLabel.Text = "Working: " + (Get-Culture).TextInfo.ToTitleCase($stage)
        }
        if ($line -match "^PLY:\s+(.+\.ply)\s*$") {
            $script:FinalPly = $Matches[1].Trim()
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
    if (-not (Test-VideoPath $videoPathBox.Text)) { return }
    if ($runNameBox.Text -notmatch "^[a-zA-Z0-9][a-zA-Z0-9_-]*$") {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "Run name may contain only letters, numbers, hyphens, and underscores.",
            "Invalid run name",
            "OK",
            "Warning"
        ) | Out-Null
        return
    }
    if (-not (Test-Path -LiteralPath $RunnerPath -PathType Leaf)) {
        [System.Windows.Forms.MessageBox]::Show($form, "Pipeline runner not found:`n$RunnerPath", "Missing runner", "OK", "Error") | Out-Null
        return
    }
    $selectedTrainer = [string]$trainerInput.SelectedItem
    if ($selectedTrainer -ne "Brush" -and -not (Test-ThreeDGRUTConfiguration)) {
        [System.Windows.Forms.MessageBox]::Show(
            $form,
            "3DGRUT is not installed or configured yet. Run scripts\doctor.ps1 after adding its repo and Python paths, or use Brush.",
            "3DGRUT unavailable",
            "OK",
            "Information"
        ) | Out-Null
        return
    }

    Close-LogReader
    $script:CompletionHandled = $false
    $script:FinalPly = $null
    $script:RunRoot = Join-Path $OutputRoot $runNameBox.Text
    $logsRoot = Join-Path $script:RunRoot "logs"
    New-Item -ItemType Directory -Force -Path $logsRoot | Out-Null
    $script:PipelineLogPath = Join-Path $logsRoot "splatitup888.log"
    if (Test-Path -LiteralPath $script:PipelineLogPath) {
        Clear-Content -LiteralPath $script:PipelineLogPath
    }

    $runner = ConvertTo-SingleQuotedPowerShell $RunnerPath
    $video = ConvertTo-SingleQuotedPowerShell $videoPathBox.Text
    $runName = ConvertTo-SingleQuotedPowerShell $runNameBox.Text
    $trainer = ConvertTo-SingleQuotedPowerShell $selectedTrainer
    $logPath = ConvertTo-SingleQuotedPowerShell $script:PipelineLogPath
    $command = "& $runner -VideoPath $video -RunName $runName -SelectedFrames $([int]$framesInput.Value) -TrainingSteps $([int]$stepsInput.Value) -Trainer $trainer"
    if (Test-Path -LiteralPath $ConfigPath -PathType Leaf) {
        $configArgument = ConvertTo-SingleQuotedPowerShell $ConfigPath
        $command += " -ConfigPath $configArgument"
    }
    if ($fastExtractionCheck.Checked) { $command += " -AdaptiveExtraction -CandidateMultiplier 2 -MaxLongSide 1600" }
    if (-not $autoRotateCheck.Checked) { $command += " -NoAutoRotate" }
    if (-not $openViewerCheck.Checked) { $command += " -NoBrowser" }
    $command += " *> $logPath; if (`$?) { exit 0 } else { exit 1 }"
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = "powershell.exe"
    $startInfo.Arguments = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded"
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true

    $logBox.Clear()
    $modeName = if ($previewMode.Checked) { "Preview" } elseif ($finalMode.Checked) { "Final" } else { "Custom" }
    $logBox.AppendText("Starting $($runNameBox.Text) | $modeName | $selectedTrainer...`r`n")
    $outputLink.Text = $script:RunRoot
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

    if ($exitCode -eq 0) {
        if (-not $script:FinalPly) {
            $candidate = Join-Path $script:RunRoot ("final\" + $runNameBox.Text + ".ply")
            if (Test-Path -LiteralPath $candidate) { $script:FinalPly = $candidate }
        }
        $statusLabel.Text = "Complete"
        $reportPath = Join-Path $script:RunRoot "reconstruction_report.json"
        if (Test-Path -LiteralPath $reportPath -PathType Leaf) {
            $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
            $registration = [double]$report.registration_percent
            $points = [double]$report.points
            $errorPixels = [double]$report.mean_reprojection_error_pixels
            $statusLabel.Text = "Complete: $([Math]::Round($registration, 1))% registered | $([Math]::Round($points)) points | $([Math]::Round($errorPixels, 2)) px"
            if ($report.quality_gates.overall_pass) {
                $statusLabel.ForeColor = New-Color "#26734D"
            } else {
                $statusLabel.ForeColor = New-Color "#A34E18"
            }
        }
        $progress.Value = 100
        if ($script:FinalPly) {
            $outputLink.Text = $script:FinalPly
            [System.Media.SystemSounds]::Asterisk.Play()
        } else {
            $outputLink.Text = $script:RunRoot
        }
    } else {
        $statusLabel.Text = "Stopped with an error"
        $progress.Value = 0
        [System.Media.SystemSounds]::Hand.Play()
    }
    $script:ActiveProcess.Dispose()
    $script:ActiveProcess = $null
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
    $dialog.Title = "Choose orbit video"
    $dialog.Filter = "Video files|*.mov;*.mp4;*.m4v;*.avi;*.mkv|All files|*.*"
    if ($dialog.ShowDialog($form) -eq "OK") { Set-Video $dialog.FileName }
    $dialog.Dispose()
})

$dragEnterHandler = {
    param($sender, $eventArgs)
    if ($eventArgs.Data.GetDataPresent([System.Windows.Forms.DataFormats]::FileDrop)) {
        $files = [string[]]$eventArgs.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)
        if ($files.Count -gt 0 -and (Test-VideoPath $files[0])) {
            $eventArgs.Effect = [System.Windows.Forms.DragDropEffects]::Copy
            return
        }
    }
    $eventArgs.Effect = [System.Windows.Forms.DragDropEffects]::None
}

$dragDropHandler = {
    param($sender, $eventArgs)
    $files = [string[]]$eventArgs.Data.GetData([System.Windows.Forms.DataFormats]::FileDrop)
    if ($files.Count -gt 0) { Set-Video $files[0] }
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
    $target = if ($script:FinalPly) { $script:FinalPly } else { $script:RunRoot }
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
    $previewMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    $previewSettings = [ordered]@{
        frames = [int]$framesInput.Value
        training_steps = [int]$stepsInput.Value
        fast_extraction = $fastExtractionCheck.Checked
    }
    $finalMode.Checked = $true
    [System.Windows.Forms.Application]::DoEvents()
    [ordered]@{
        app = $form.Text
        runner_found = Test-Path -LiteralPath $RunnerPath -PathType Leaf
        drag_drop_enabled = $form.AllowDrop -and $dropPanel.AllowDrop
        default_frames = [int]$framesInput.Value
        default_training_steps = [int]$stepsInput.Value
        default_trainer = [string]$trainerInput.SelectedItem
        default_mode = if ($finalMode.Checked) { "Final" } else { "Other" }
        autorotate = $autoRotateCheck.Checked
        preview = $previewSettings
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
