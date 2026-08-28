Set-StrictMode -Version Latest

function Get-SplatProductionValue {
    param(
        [hashtable]$Config,
        [Parameter(Mandatory = $true)][string]$Name,
        [double]$Default
    )
    if ($Config.ContainsKey("Production") -and $Config.Production.ContainsKey($Name)) {
        return [double]$Config.Production[$Name]
    }
    return $Default
}

function Get-SplatDiskCapacity {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $driveRoot = [System.IO.Path]::GetPathRoot($fullPath)
    if (-not $driveRoot) { throw "Could not resolve the output drive for $fullPath" }
    $drive = [System.IO.DriveInfo]::new($driveRoot)
    if (-not $drive.IsReady) { throw "Output drive is not ready: $driveRoot" }
    return [pscustomobject]@{
        path = $fullPath
        drive_root = $driveRoot
        available_bytes = [int64]$drive.AvailableFreeSpace
        available_gb = [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
    }
}

function Assert-SplatDiskCapacity {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][double]$RequiredGB
    )

    if ($RequiredGB -le 0) { throw "RequiredGB must be positive" }
    $capacity = Get-SplatDiskCapacity -Path $Path
    $requiredBytes = [int64][math]::Ceiling($RequiredGB * 1GB)
    if ($capacity.available_bytes -lt $requiredBytes) {
        throw "INSUFFICIENT_DISK_SPACE: $($capacity.drive_root) has $($capacity.available_gb) GB free; this operation requires at least $RequiredGB GB."
    }
    return $capacity
}

function Enter-SplatExclusiveLock {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][hashtable]$Metadata
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $parent = Split-Path -Parent $fullPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $stream = $null
    try {
        $stream = [System.IO.File]::Open(
            $fullPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
        $payload = [ordered]@{
            schema_version = 1
            process_id = $PID
            acquired_utc = [DateTime]::UtcNow.ToString("o")
        }
        foreach ($key in $Metadata.Keys) { $payload[$key] = $Metadata[$key] }
        $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes(($payload | ConvertTo-Json -Depth 8))
        $stream.SetLength(0)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
        return [pscustomobject]@{ Path = $fullPath; Stream = $stream }
    } catch {
        if ($stream) { $stream.Dispose() }
        throw "SPLAT_LOCKED: Another SplatItUp888 process owns $fullPath. Wait for it to finish or stop that process before retrying."
    }
}

function Exit-SplatExclusiveLock {
    param($Lock)
    if (-not $Lock) { return }
    try { $Lock.Stream.Dispose() } catch {}
    try { Remove-Item -LiteralPath $Lock.Path -Force -ErrorAction Stop } catch {}
}

function Write-SplatJsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value,
        [int]$Depth = 12
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$Path.tmp-$PID"
    $Value | ConvertTo-Json -Depth $Depth | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}
