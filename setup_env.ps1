param(
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot "scripts\Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath

Write-Host "Using Python: $python"
& $python -m pip install --disable-pip-version-check -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed with exit code $LASTEXITCODE."
}

& $python -c "import torch, torchvision, yaml, sklearn, cv2, open_clip; print('torch', torch.__version__); print('open_clip', open_clip.__version__); print('cuda', torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) {
    throw "Environment import check failed with exit code $LASTEXITCODE."
}
