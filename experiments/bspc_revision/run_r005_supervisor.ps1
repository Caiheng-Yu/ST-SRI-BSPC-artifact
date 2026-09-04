param(
    [int]$MaxAttempts = 5,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'python'
$Experiment = Join-Path $ResearchRoot 'experiments\bspc_revision\leakage_free_db2.py'
$OutputDirectory = Join-Path $ResearchRoot 'results\bspc_revision_v2\r005_full_s1_s40'
$CheckpointDirectory = Join-Path $ResearchRoot 'checkpoints_bspc_v2\r005_full_s1_s40'
$LogDirectory = Join-Path $ResearchRoot 'logs\bspc_revision_v2'
$LogPath = Join-Path $LogDirectory 'r005_full_s1_s40.log'
$SupervisorStatusPath = Join-Path $OutputDirectory 'supervisor_status.json'

New-Item -ItemType Directory -Force -Path $OutputDirectory, $CheckpointDirectory, $LogDirectory | Out-Null

function Write-AtomicJson {
    param([string]$Path, [hashtable]$Payload)
    $TemporaryPath = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $TemporaryPath -Encoding utf8
    Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
}

for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    Write-AtomicJson -Path $SupervisorStatusPath -Payload @{
        state = 'running'
        attempt = $Attempt
        max_attempts = $MaxAttempts
        process_id = $PID
        started_at = (Get-Date).ToString('o')
        log = $LogPath
    }
    "[$(Get-Date -Format o)] R005 attempt $Attempt/$MaxAttempts" | Tee-Object -FilePath $LogPath -Append
    & $Python -u $Experiment `
        --all-subjects `
        --epochs 40 `
        --patience 12 `
        --batch-size 64 `
        --resume `
        --output-dir $OutputDirectory `
        --checkpoint-dir $CheckpointDirectory 2>&1 | Tee-Object -FilePath $LogPath -Append
    $ExitCode = $LASTEXITCODE

    if ($ExitCode -eq 0) {
        $SummaryPath = Join-Path $OutputDirectory 'protocol_summary.json'
        if (Test-Path -LiteralPath $SummaryPath) {
            $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
            if ($Summary.state -eq 'completed' -and $Summary.training_results.Count -eq 40) {
                Write-AtomicJson -Path $SupervisorStatusPath -Payload @{
                    state = 'completed'
                    attempt = $Attempt
                    process_id = $PID
                    completed_at = (Get-Date).ToString('o')
                    completed_subjects = 40
                    log = $LogPath
                }
                exit 0
            }
        }
    }

    "[$(Get-Date -Format o)] R005 failed with exit code $ExitCode" | Tee-Object -FilePath $LogPath -Append
    if ($Attempt -lt $MaxAttempts) {
        Start-Sleep -Seconds $RetryDelaySeconds
    }
}

Write-AtomicJson -Path $SupervisorStatusPath -Payload @{
    state = 'failed'
    attempts = $MaxAttempts
    process_id = $PID
    failed_at = (Get-Date).ToString('o')
    log = $LogPath
}
exit 1
