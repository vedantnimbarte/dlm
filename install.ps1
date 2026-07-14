# dlm installer for Windows — download a prebuilt binary and drop it on your PATH.
#
#   irm https://raw.githubusercontent.com/vedantnimbarte/dlm/main/install.ps1 | iex
#
# Installs the portable CPU build. (The CUDA build is Linux x86-64 only; on
# Windows, build from source with `cargo install --git ... --features cuda-kernels`
# — that needs the CUDA toolkit and MSVC, which build.rs locates via vswhere.)
#
# Env:
#   DLM_INSTALL_DIR   install location (default: %LOCALAPPDATA%\Programs\dlm)

$ErrorActionPreference = 'Stop'

$Repo = 'vedantnimbarte/dlm'
$Bin = 'dlm.exe'
$InstallDir = if ($env:DLM_INSTALL_DIR) { $env:DLM_INSTALL_DIR }
              else { Join-Path $env:LOCALAPPDATA 'Programs\dlm' }

function Die($msg) { Write-Error "error: $msg"; exit 1 }

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    Die "unsupported architecture '$env:PROCESSOR_ARCHITECTURE'. Prebuilt Windows binaries are x86-64 only; build from source: cargo install --git https://github.com/$Repo"
}

$asset = 'dlm-x86_64-pc-windows-msvc.zip'
$base = "https://github.com/$Repo/releases/latest/download"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("dlm-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
    Write-Host "Installing dlm (x86_64-pc-windows-msvc, CPU build)..."

    $zip = Join-Path $tmp $asset
    try { Invoke-WebRequest -Uri "$base/$asset" -OutFile $zip -UseBasicParsing }
    catch { Die "download failed: $base/$asset ($($_.Exception.Message))" }

    # Verify the checksum before extracting and running anything.
    $sumFile = "$zip.sha256"
    try {
        Invoke-WebRequest -Uri "$base/$asset.sha256" -OutFile $sumFile -UseBasicParsing
        $want = ((Get-Content $sumFile -Raw).Trim() -split '\s+')[0]
        $have = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
        if ($want.ToLower() -ne $have) {
            Die "checksum mismatch for ${asset}: expected $want, got $have. The download was corrupted or tampered with."
        }
    } catch [System.Net.WebException] {
        Write-Warning "no published checksum for $asset - skipping verification"
    }

    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $exe = Get-ChildItem -Path $tmp -Filter $Bin -Recurse | Select-Object -First 1
    if (-not $exe) { Die "binary '$Bin' not found in $asset" }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item $exe.FullName (Join-Path $InstallDir $Bin) -Force
    Write-Host "Installed the CPU build to $(Join-Path $InstallDir $Bin)"

    # Put it on PATH for future sessions if it isn't already.
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($userPath -notlike "*$InstallDir*") {
        [Environment]::SetEnvironmentVariable('Path', "$userPath;$InstallDir", 'User')
        Write-Host ""
        Write-Host "  Added $InstallDir to your user PATH."
        Write-Host "  Open a new terminal, or for this session run:"
        Write-Host "    `$env:Path += ';$InstallDir'"
    }

    Write-Host ""
    & (Join-Path $InstallDir $Bin) --version
    Write-Host "Done. Try:  dlm --help"
}
finally {
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}
