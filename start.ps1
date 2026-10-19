param(
    [switch]$SetupOnly,
    [string]$DataDirectory = ""
)
$ErrorActionPreference = 'Stop'
$carePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$careWorkspacePython = Join-Path $PSScriptRoot '..\..\work\.venv-care\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $carePython)) {
    if (Test-Path -LiteralPath $careWorkspacePython) {
        $carePython = (Resolve-Path -LiteralPath $careWorkspacePython).Path
    } else {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.11 -m venv (Join-Path $PSScriptRoot '.venv')
        } else {
            & python -m venv (Join-Path $PSScriptRoot '.venv')
        }
        if ($LASTEXITCODE -ne 0) { throw 'Cannot create environment. Install Python 3.11 first.' }
    }
}
& $carePython -c "import sys; sys.exit(0 if sys.version_info[:2] == (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'This tested environment requires Python 3.11; reinstalling packages will not change Python.' }
& $carePython -c "from importlib.metadata import version; expected={'mediapipe':'0.10.35','opencv-contrib-python':'4.11.0.86','numpy':'2.2.6','PySide6':'6.8.3'}; assert all(version(k)==v for k,v in expected.items())" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $carePython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check network and Python 3.11.' }
}
& $carePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed.' }
Write-Host "Python: $carePython"
if (-not $SetupOnly) {
    if ($DataDirectory) {
        & $carePython (Join-Path $PSScriptRoot 'app.py') --data $DataDirectory
    } else {
        & $carePython (Join-Path $PSScriptRoot 'app.py')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Application exited with an error.' }
}
