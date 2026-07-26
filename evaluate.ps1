param(
    [string]$Config = "configs\rtx3060ti_strict_upstream.yaml",
    [string]$Checkpoint = "auto",
    [string]$PythonPath = "",
    [string[]]$Set = @(),
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot "scripts\Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }

$arguments = @(
    (Join-Path $projectRoot "evaluate.py"),
    "--config", $configPath,
    "--checkpoint", $Checkpoint
)
foreach ($override in $Set) {
    $arguments += @("--set", $override)
}
if ($AllowPartial) {
    $arguments += "--allow-partial"
}

& $python @arguments
exit $LASTEXITCODE
