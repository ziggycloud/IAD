param(
    [string]$Config = "configs\rtx3060ti_strict_upstream.yaml",
    [ValidateSet("metadata", "sample", "full")]
    [string]$Mode = "sample",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
. (Join-Path $projectRoot "scripts\Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }

& $python (Join-Path $projectRoot "validate_data.py") --config $configPath --mode $Mode
exit $LASTEXITCODE
