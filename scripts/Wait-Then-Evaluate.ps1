param(
    [Parameter(Mandatory = $true)]
    [int]$TrainPid,
    [string]$Config = "configs\rtx3060ti_strict_upstream.yaml",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "Resolve-IadPython.ps1")
$python = Resolve-IadPython -PythonPath $PythonPath
$configPath = if ([System.IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $projectRoot $Config
}

$outputDir = Join-Path $projectRoot "outputs\dinomaly2_realiad_variety_b_280_strict_upstream"
$runStatePath = Join-Path $outputDir "run_state.json"
$pipelineStatePath = Join-Path $outputDir "pipeline_state.json"

function Write-PipelineState {
    param(
        [string]$Status,
        [string]$NextAction,
        [string]$LastError = ""
    )
    $payload = [ordered]@{
        status = $Status
        updated_at = [DateTime]::UtcNow.ToString("o")
        train_pid = $TrainPid
        config = $configPath
        next_action = $NextAction
    }
    if ($LastError) {
        $payload.last_error = $LastError
    }
    $payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $pipelineStatePath -Encoding UTF8
}

try {
    Write-PipelineState -Status "waiting_for_training" -NextAction "Wait for the training process to finish."
    if (Get-Process -Id $TrainPid -ErrorAction SilentlyContinue) {
        Wait-Process -Id $TrainPid
    }
    if (-not (Test-Path -LiteralPath $runStatePath -PathType Leaf)) {
        throw "Training run_state.json is missing: $runStatePath"
    }
    $runState = Get-Content -LiteralPath $runStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($runState.status -ne "trained") {
        throw "Training did not finish successfully; status=$($runState.status)"
    }

    Write-PipelineState -Status "evaluating" -NextAction "Run the resumable 160-category paper evaluation."
    & $python (Join-Path $projectRoot "evaluate.py") --config $configPath --checkpoint auto
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation exited with code $LASTEXITCODE"
    }
    Write-PipelineState -Status "complete" -NextAction "Read reports\evaluation_report.md."
}
catch {
    Write-PipelineState -Status "failed" -NextAction "Inspect pipeline stderr and run/evaluation state files." -LastError $_.Exception.Message
    throw
}
