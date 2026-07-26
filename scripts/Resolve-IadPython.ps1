function Resolve-IadPython {
    param(
        [string]$PythonPath = ""
    )

    $candidates = [System.Collections.Generic.List[string]]::new()
    if ($PythonPath) {
        $candidates.Add($PythonPath)
    }
    if ($env:IAD_PYTHON) {
        $candidates.Add($env:IAD_PYTHON)
    }
    $candidates.Add("J:\project\IAD\data\.conda\iad\python.exe")
    $candidates.Add("J:\project\IAD\data\.conda\realiad-variety-py311\python.exe")
    $candidates.Add("J:\project\IAD\data.conda\realiad-variety-py311\python.exe")

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if ($conda) {
        try {
            $environmentJson = & conda env list --json 2>$null
            $environmentData = $environmentJson | ConvertFrom-Json
            foreach ($prefix in $environmentData.envs) {
                if ((Split-Path -Leaf $prefix) -ieq "IAD") {
                    $candidates.Add((Join-Path $prefix "python.exe"))
                }
            }
        }
        catch {
        }
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "IAD Python was not found. Pass -PythonPath or set IAD_PYTHON."
}
