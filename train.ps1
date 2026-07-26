param(
    [string]$Config = "configs\rtx3060ti_strict_upstream.yaml",
    [string]$Resume = "auto",
    [string]$PythonPath = "",
    [string[]]$Set = @(),
    [switch]$ProbeOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot "scripts\Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }

$arguments = @(
    (Join-Path $projectRoot "train.py"),
    "--config", $configPath,
    "--resume", $Resume
)
foreach ($override in $Set) {
    $arguments += @("--set", $override)
}
if ($ProbeOnly) {
    $arguments += "--probe-only"
}

& $python @arguments
exit $LASTEXITCODE
