# R012 试点：ResNet1D × 6 人（S1/S10/S15/S17/S27/S40），协议同 R005a
param(
    [int]$MaxAttemptsPerRun = 3,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'python'
$LogDirectory = Join-Path $ResearchRoot 'logs\bspc_revision_v2'
$RunLog = Join-Path $LogDirectory 'r012_pilot.log'
$StatusPath = Join-Path $ResearchRoot 'results\bspc_revision_v2\r012_pilot\r012_status.json'

New-Item -ItemType Directory -Force -Path $LogDirectory, (Join-Path $ResearchRoot 'results\bspc_revision_v2\r012_pilot') | Out-Null

function Write-Status {
    param([string]$State, [hashtable]$Extra = @{})
    $Payload = @{
        state = $State
        run_name = 'r012_pilot'
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
        "[$(Get-Date -Format o)] r012_pilot attempt $Attempt/$MaxAttemptsPerRun" |
            Tee-Object -FilePath $RunLog -Append

        & $Python -u (Join-Path $ResearchRoot 'experiments\bspc_revision\leakage_free_db2.py') `
            --subjects 1 10 15 17 27 40 `
            --arch resnet1d `
            --epochs 60 --patience 15 --min-epochs 30 --batch-size 64 `
            --selection-metric macro_f1 --class-weight-power 0.5 `
            --per-subject-seed --split-seed 20260815 --train-seed 20260820 `
            --resume --skip-window-manifests `
            --output-dir (Join-Path $ResearchRoot 'results\bspc_revision_v2\r012_pilot') `
            --checkpoint-dir (Join-Path $ResearchRoot 'checkpoints_bspc_v2\r012_pilot') 2>&1 |
            Tee-Object -FilePath $RunLog -Append
        $ExitCode = $LASTEXITCODE

        $Summary = Join-Path $ResearchRoot 'results\bspc_revision_v2\r012_pilot\protocol_summary.json'
        $Complete = $false
        if (Test-Path -LiteralPath $Summary) {
            try {
                $S = Get-Content -Raw -LiteralPath $Summary | ConvertFrom-Json
                $Complete = $S.state -eq 'completed' -and $S.training_results.Count -eq 6
            } catch { }
        }
        if ($ExitCode -eq 0 -and $Complete) {
            Write-Status -State 'completed' -Extra @{ completed_at = (Get-Date).ToString('o') }
            exit 0
        }

        "[$(Get-Date -Format o)] r012_pilot failed with exit code $ExitCode" |
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
