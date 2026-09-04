# R008 补充相位曲线采集：steady/offset 窗口（R005b 检查点，trained/reinit/param_scramble）
param(
    [int]$MaxAttemptsPerRun = 3,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $ResearchRoot
$Python = 'python'
$Script = Join-Path $ResearchRoot 'experiments\bspc_revision\collect_st_sri_curves.py'
$Manifest = Join-Path $ResearchRoot 'results\bspc_revision_v2\r009_window_manifest\phase_windows.csv'
$LogDirectory = Join-Path $ResearchRoot 'logs\bspc_revision_v2'
$RunLog = Join-Path $LogDirectory 'r008_phase_curves.log'
$StatusPath = Join-Path $ResearchRoot 'results\bspc_revision_v2\r008_phase_curves\r008_phase_status.json'

New-Item -ItemType Directory -Force -Path $LogDirectory, (Join-Path $ResearchRoot 'results\bspc_revision_v2\r008_phase_curves') | Out-Null

function Write-Status {
    param([string]$State, [hashtable]$Extra = @{})
    $Payload = @{
        state = $State
        run_name = 'r008_phase_curves'
        process_id = $PID
        updated_at = (Get-Date).ToString('o')
        run_log = $RunLog
    }
    foreach ($Key in $Extra.Keys) { $Payload[$Key] = $Extra[$Key] }
    $TemporaryPath = "$StatusPath.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $TemporaryPath -Encoding utf8
    Move-Item -LiteralPath $TemporaryPath -Destination $StatusPath -Force
}

try {
    for ($Attempt = 1; $Attempt -le $MaxAttemptsPerRun; $Attempt++) {
        Write-Status -State 'running' -Extra @{ attempt = $Attempt; max_attempts = $MaxAttemptsPerRun }
        "[$(Get-Date -Format o)] r008_phase_curves attempt $Attempt/$MaxAttemptsPerRun" |
            Tee-Object -FilePath $RunLog -Append

        & $Python -u $Script `
            --all-subjects `
            --checkpoint-dir checkpoints_bspc_v2\r005b_balanced_full_seed20260815 `
            --output-dir results\bspc_revision_v2\r008_phase_curves `
            --records-csv $Manifest `
            --window-kinds offset steady `
            --variants trained reinit param_scramble `
            --device cuda `
            --threads 2 2>&1 |
            Tee-Object -FilePath $RunLog -Append
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -eq 0) {
            Write-Status -State 'completed' -Extra @{ completed_at = (Get-Date).ToString('o') }
            exit 0
        }

        "[$(Get-Date -Format o)] r008_phase_curves failed with exit code $ExitCode" |
            Tee-Object -FilePath $RunLog -Append
        if ($Attempt -lt $MaxAttemptsPerRun) {
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }

    Write-Status -State 'failed' -Extra @{ error = 'exhausted retry attempts' }
    exit 1
}
catch {
    $_ | Out-String | Tee-Object -FilePath $RunLog -Append
    Write-Status -State 'failed' -Extra @{ error = $_.Exception.Message }
    exit 1
}
