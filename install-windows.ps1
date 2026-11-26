[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*)?$')]
    [string] $Tag,
    [switch] $PreflightOnly,
    [string] $ArchivePath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Test-VDRelease([string] $Directory) {
    $python = Join-Path $Directory 'python\python.exe'
    $entryPoint = Join-Path $Directory 'run.py'
    & $python -B $entryPoint preflight
    if ($LASTEXITCODE -ne 0) { throw "VD preflight failed in $Directory." }
}

$root = Join-Path $env:LOCALAPPDATA 'VD-Solver'
$install = Join-Path $root $Tag
if (Test-Path -LiteralPath $install -PathType Container) {
    Write-Host "Checking VD Solver $Tag in $install"
    Test-VDRelease $install
} else {
    $temporary = Join-Path $root ('.install-' + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $temporary -Force | Out-Null
    try {
        if ($ArchivePath) {
            $zip = (Resolve-Path -LiteralPath $ArchivePath).Path
            $checksum = "$zip.sha256"
        } else {
            $base = "https://github.com/jakuza0ae/vd-auto-skill-check/releases/download/$Tag"
            $zip = Join-Path $temporary 'vd-windows-x64.zip'
            $checksum = "$zip.sha256"
            Write-Host "Downloading VD Solver $Tag (Windows x64)"
            Invoke-WebRequest -UseBasicParsing "$base/vd-windows-x64.zip" -OutFile $zip
            Invoke-WebRequest -UseBasicParsing "$base/vd-windows-x64.zip.sha256" -OutFile $checksum
        }
        $hash = ((Get-Content -LiteralPath $checksum -TotalCount 1) -split '\s+')[0]
        if ($hash -notmatch '^[0-9a-f]{64}$' -or
            (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash -ne $hash) {
            throw 'VD download checksum mismatch.'
        }
        $staging = Join-Path $temporary 'app'
        Write-Host 'Verifying the extracted application'
        Expand-Archive -LiteralPath $zip -DestinationPath $staging
        Test-VDRelease $staging
        [System.IO.Directory]::Move($staging, $install)
    } finally {
        Remove-Item -LiteralPath $temporary -Force -Recurse -ErrorAction SilentlyContinue
    }
}

Write-Host "VD Solver $Tag is ready in $install"
if ($PreflightOnly) { return }
Write-Host "Recordings: $(Join-Path $install 'recordings')"
Write-Host 'Switch to fullscreen Roblox on the primary display. Hold LMB to arm. Press Ctrl+C here to stop.'
& (Join-Path $install 'python\python.exe') -B (Join-Path $install 'run.py') run
if ($LASTEXITCODE -ne 0) { throw "VD Solver exited with code $LASTEXITCODE." }
