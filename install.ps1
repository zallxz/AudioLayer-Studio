# AudioLayer Studio — one-step installer for Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File install.ps1          full install (AI models)
#   powershell -ExecutionPolicy Bypass -File install.ps1 -Lite    no PyTorch (detection-only)
#   ... -Yes                                                      no questions
param([switch]$Lite, [switch]$Yes)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Say($m) { Write-Host "`n==> $m" -ForegroundColor Yellow }
function Die($m) { Write-Host "`nError: $m" -ForegroundColor Red; exit 1 }
function Ask($q) { if ($Yes) { return $true }; $r = Read-Host "$q [Y/n]"; return ($r -eq "" -or $r -match "^[Yy]") }
function Has($c) { return [bool](Get-Command $c -ErrorAction SilentlyContinue) }
function Refresh-Path { $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") }

# --- Python ---------------------------------------------------------------------------
$py = $null
foreach ($c in @("py -3.13", "py -3.12", "py -3.11", "python")) {
  $parts = $c.Split(" ")
  if (Has $parts[0]) {
    try {
      & $parts[0] $parts[1..9] -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>$null
      if ($LASTEXITCODE -eq 0) { $py = $parts; break }
    } catch {}
  }
}
if (-not $py) {
  if ((Has winget) -and (Ask "Python 3.11+ not found. Install Python 3.12 with winget?")) {
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    Refresh-Path; $py = @("py", "-3.12")
  } else { Die "Install Python 3.11+ from https://www.python.org/downloads/ (tick 'Add python.exe to PATH')" }
}
Say ("Using " + (& $py[0] $py[1..9] --version))

# --- FFmpeg ---------------------------------------------------------------------------
if (-not (Has ffmpeg) -or -not (Has ffprobe)) {
  if ((Has winget) -and (Ask "FFmpeg not found. Install with winget?")) {
    winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
    Refresh-Path
    if (-not (Has ffmpeg)) { Die "FFmpeg installed — close this window, open a new PowerShell and run install.ps1 again." }
  } else { Die "FFmpeg is required: winget install Gyan.FFmpeg" }
}
Say ("FFmpeg: " + (Get-Command ffmpeg).Source)

# --- venv + packages ---------------------------------------------------------------------
if (-not (Test-Path .venv)) { Say "Creating virtual environment (.venv)"; & $py[0] $py[1..9] -m venv .venv }
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$al = Join-Path $PSScriptRoot ".venv\Scripts\audiolayer.exe"
& $vpy -m pip install -q --upgrade pip
if ($Lite) {
  Say "Installing AudioLayer (lite, no PyTorch)"; & $vpy -m pip install -q -e .
} else {
  & $vpy -c "import torch" 2>$null
  if ($LASTEXITCODE -ne 0) {
    Say "Installing PyTorch (this is the big download)"
    if (Has nvidia-smi) { & $vpy -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128 }
    else { & $vpy -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu }
  }
  Say "Installing AudioLayer + AI model packages"; & $vpy -m pip install -q -e ".[ml]"
  Say "Downloading model weights (~550 MB, once)"
  & $al models download
  if ($LASTEXITCODE -ne 0) { Write-Host "Model download failed — it will be retried on first use." }
}

# --- web UI (prebuilt copy ships in the package) --------------------------------------------
if ((Has npm) -and -not (Test-Path frontend\dist\index.html)) {
  Say "Building the web UI"
  Push-Location frontend; try { npm ci --no-audit --no-fund --silent; npm run build --silent } catch { Write-Host "UI build skipped (using the prebuilt UI)." }; Pop-Location
}

# --- launcher ------------------------------------------------------------------------------
$bat = Join-Path $PSScriptRoot "AudioLayer.bat"
"@echo off`r`ncd /d `"$PSScriptRoot`"`r`n`"$al`" web`r`n" | Set-Content -Encoding ASCII $bat

# --- Claude ----------------------------------------------------------------------------------
$inputs = Join-Path $env:USERPROFILE "AudioLayer\inputs"
if (Ask "Connect AudioLayer to Claude Desktop (Claude may read media from $inputs)?") {
  $args2 = @("setup-claude", "--input-dir", $inputs)
  if ((Has claude) -and (Ask "Also connect Claude Code?")) { $args2 += "--code" }
  & $al @args2
}

& $al doctor
Write-Host "`nDone!" -ForegroundColor Green
Write-Host "  Start the app:  double-click $bat"
Write-Host "  One command:    .venv\Scripts\audiolayer clean my_clip.mp4 --preset remove_music --loudness youtube --video"
Write-Host "  Claude:         put clips in $inputs, restart Claude Desktop, and ask it to clean them."
