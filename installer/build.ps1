# Builds the Windows installer: a staging copy of the code (no outputs, no environments), then Inno Setup.
#   powershell -ExecutionPolicy Bypass -File installer\build.ps1 [-Version 0.4.0] [-Iscc "C:\...\ISCC.exe"]
param([string]$Version = "0.5.0", [string]$Iscc = "")
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Stage = Join-Path $PSScriptRoot "staging"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
New-Item -ItemType Directory $Stage | Out-Null
foreach ($d in "splat2mesh", "scripts", "assets", "tests", "web", "installer") {
  robocopy (Join-Path $Repo $d) (Join-Path $Stage $d) /E /XD __pycache__ staging output /XF *.pyc /NFL /NDL /NJH /NJS | Out-Null
}
New-Item -ItemType Directory (Join-Path $Stage "docs") | Out-Null
Get-ChildItem (Join-Path $Repo "docs") -Filter *.md | Copy-Item -Destination (Join-Path $Stage "docs")
foreach ($f in "README.md", "LICENSE", "NOTICE", "requirements.txt", "splat2mesh.example.toml") {
  Copy-Item (Join-Path $Repo $f) $Stage
}
if (-not $Iscc) {
  foreach ($c in "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe", "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
                 "$env:ProgramFiles\Inno Setup 6\ISCC.exe") { if (Test-Path $c) { $Iscc = $c; break } }
}
if (-not $Iscc) { throw "Inno Setup 6.3+ not found (winget install JRSoftware.InnoSetup)" }
& $Iscc "/DAppVersion=$Version" "/DStaging=staging" (Join-Path $PSScriptRoot "splat2mesh.iss")
if ($LASTEXITCODE) { throw "ISCC failed" }
Get-ChildItem (Join-Path $PSScriptRoot "output")
