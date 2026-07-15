# dlm installer for Windows — download a prebuilt binary and drop it on your PATH.
#
#   irm https://raw.githubusercontent.com/vedantnimbarte/dlm/main/install.ps1 | iex
#
# Installs the GPU (static-CUDA) build when an NVIDIA GPU is detected, otherwise
# the portable CPU build. The GPU build embeds the CUDA runtime, so it needs only
# the NVIDIA driver — no CUDA toolkit install. An AMD GPU gets the CPU build for
# now (AMD GPU support is planned).
#
# Env:
#   DLM_INSTALL_DIR   install location (default: %LOCALAPPDATA%\Programs\dlm)
#   DLM_CPU=1         force the portable CPU build even if a GPU is detected

$ErrorActionPreference = 'Stop'

$Repo = 'vedantnimbarte/dlm'
$Bin = 'dlm.exe'
$InstallDir = if ($env:DLM_INSTALL_DIR) { $env:DLM_INSTALL_DIR }
              else { Join-Path $env:LOCALAPPDATA 'Programs\dlm' }

function Die($msg) { Write-Error "error: $msg"; exit 1 }

if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    Die "unsupported architecture '$env:PROCESSOR_ARCHITECTURE'. Prebuilt Windows binaries are x86-64 only; build from source: cargo install --git https://github.com/$Repo"
}

$target = 'x86_64-pc-windows-msvc'
$cpuAsset = "dlm-$target.zip"
$gpuAsset = "dlm-$target-cuda-static.zip"
$base = "https://github.com/$Repo/releases/latest/download"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("dlm-" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $tmp | Out-Null

# Download $asset into $tmp, verify its published .sha256 before touching it,
# extract it, and return the dlm.exe it contains. Throws on any download,
# checksum, or extract failure so the caller can fall back or abort.
function Get-Asset($asset) {
    $zip = Join-Path $tmp $asset
    Invoke-WebRequest -Uri "$base/$asset" -OutFile $zip -UseBasicParsing

    # Checksum asset is named after the archive *stem* — `dlm-<target>.sha256`
    # (no `.zip`) — and its body is "<hash> *<filename>".
    $stem = [IO.Path]::GetFileNameWithoutExtension($asset)
    $sumFile = Join-Path $tmp "$stem.sha256"
    Invoke-WebRequest -Uri "$base/$stem.sha256" -OutFile $sumFile -UseBasicParsing

    $want = ((Get-Content $sumFile -Raw).Trim().Split()[0]).ToLower()
    $have = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if (-not $want) { throw "empty checksum file for $asset" }
    if ($want -ne $have) {
        throw "checksum mismatch for ${asset}: expected $want, got $have. The download was corrupted or tampered with."
    }

    # Extract into a per-asset subdir so the CPU and GPU binaries never collide.
    $dest = Join-Path $tmp $stem
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    $found = Get-ChildItem -Path $dest -Filter $Bin -Recurse | Select-Object -First 1
    if (-not $found) { throw "binary '$Bin' not found in $asset" }
    return $found
}

try {
    # Pick the build: NVIDIA GPU present -> GPU (static-CUDA), else CPU. DLM_CPU=1
    # forces CPU. The GPU build embeds the CUDA runtime, so it needs only the
    # driver. AMD GPUs land in the CPU branch (AMD GPU support is planned).
    $useGpu = $false
    if ($env:DLM_CPU -eq '1') {
        Write-Host "DLM_CPU=1 set - installing the CPU build."
    } elseif ((Get-Command nvidia-smi -ErrorAction SilentlyContinue) -and (& { nvidia-smi *>$null; $LASTEXITCODE -eq 0 })) {
        $useGpu = $true
        Write-Host "NVIDIA GPU detected - installing the GPU (CUDA) build."
    } else {
        Write-Host "No NVIDIA GPU detected - installing the CPU build. (AMD GPU support is planned; it runs on CPU for now.)"
    }

    $kind = if ($useGpu) { 'GPU (CUDA)' } else { 'CPU' }
    Write-Host "Installing dlm ($target, $kind build)..."

    $exe = $null
    if ($useGpu) {
        # If the GPU asset isn't published in this release (e.g. its CI build
        # failed while CPU succeeded), fall back to CPU instead of aborting.
        try { $exe = Get-Asset $gpuAsset }
        catch {
            Write-Host "GPU build unavailable ($($_.Exception.Message)) - falling back to the CPU build."
            $useGpu = $false
            $kind = 'CPU'
        }
        # Embeds the CUDA runtime but still needs a usable NVIDIA driver. If it
        # won't start, fall back to CPU so we never install a binary that can't run.
        if ($useGpu) {
            & $exe.FullName --version *>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "GPU build won't start here (NVIDIA driver missing or too old?) - falling back to the CPU build."
                Write-Host "Update your NVIDIA driver and re-run to get the GPU build."
                $useGpu = $false
                $kind = 'CPU'
            }
        }
    }
    if (-not $useGpu) {
        try { $exe = Get-Asset $cpuAsset }
        catch { Die "download/verify failed for the CPU build: $($_.Exception.Message)" }
    }

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Copy-Item $exe.FullName (Join-Path $InstallDir $Bin) -Force
    Write-Host "Installed the $kind build to $(Join-Path $InstallDir $Bin)"

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
