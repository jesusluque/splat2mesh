# splat2mesh core: uv, Python 3.12 and the package's own environment (.venv), all under the install folder.
# Run by the installer wizard and by install_all.bat. Prints "PROGRESS <0-100> core: <message>" lines.
#   powershell -ExecutionPolicy Bypass -File bootstrap.ps1 [-Root <install folder>]
param([string]$Root = (Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
function P($pct, $msg) { Write-Output "PROGRESS $pct core: $msg" }

Set-Location $Root
$uvDir = Join-Path $Root "tools\uv"
$uv = Join-Path $uvDir "uv.exe"
if (-not (Test-Path $uv)) {
  P 5 "downloading uv"
  New-Item -ItemType Directory -Force $uvDir | Out-Null
  $zip = Join-Path $uvDir "uv.zip"
  Invoke-WebRequest "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
  Expand-Archive $zip -DestinationPath $uvDir -Force
  Remove-Item $zip
}
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root "tools\python"
$env:UV_CACHE_DIR = Join-Path $Root "tools\uv-cache"
P 15 "Python 3.12"
& $uv python install 3.12
if ($LASTEXITCODE) { throw "uv python install failed" }
P 30 "environment"
if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) { & $uv venv --python 3.12 "$Root\.venv" }
if ($LASTEXITCODE) { throw "uv venv failed" }
P 40 "splat2mesh dependencies (numpy, open3d, usd-core, OpenEXR...)"
& $uv pip install --python "$Root\.venv\Scripts\python.exe" -r "$Root\requirements.txt"
if ($LASTEXITCODE) { throw "pip install failed" }
P 90 "checking"
& "$Root\.venv\Scripts\python.exe" -c "import splat2mesh, numpy, trimesh, pxr, OpenEXR; print('core ok')"
if ($LASTEXITCODE) { throw "core check failed" }
P 100 "ready"
