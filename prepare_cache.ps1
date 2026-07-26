param(
    [string]$Config = "configs\rtx3060ti.yaml",
    [string]$PythonPath = "",
    [string[]]$Set = @(),
    [int]$MaxImages = 0
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot "scripts\Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }

$arguments = @(
    (Join-Path $projectRoot "prepare_cache.py"),
    "--config", $configPath
)
foreach ($override in $Set) {
    $arguments += @("--set", $override)
}
if ($MaxImages -gt 0) {
    $arguments += @("--max-images", $MaxImages)
}

& $python @arguments
exit $LASTEXITCODE
