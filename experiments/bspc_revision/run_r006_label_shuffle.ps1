# R006 标签打乱阴性对照训练监督器（协议与 R005b 一致，仅置乱训练窗标签）
param(
    [int]$MaxAttemptsPerRun = 3,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'python'
$Script = Join-Path $ResearchRoot 'experiments\bspc_revision\train_label_shuffle.py'
$OutputDirectory = Join-Path $ResearchRoot 'results\bspc_revision_v2\r006_label_shuffle'
$CheckpointDirectory = Join-Path $ResearchRoot 'checkpoints_bspc_v2\r006_label_shuffle'
$LogDirectory = Join-Path $ResearchRoot 'logs\bspc_revision_v2'
$StatusPath = Join-Path $OutputDirectory 'r006_status.json'
$RunLog = Join-Path $LogDirectory 'r006_label_shuffle.log'

New-Item -ItemType Directory -Force -Path $OutputDirectory, $CheckpointDirectory, $LogDirectory | Out-Null

function Write-AtomicJson {
    param([string]$Path, [hashtable]$Payload)
    $TemporaryPath = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $TemporaryPath -Encoding utf8
    Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
}

function Write-Status {
    param([string]$State, [hashtable]$Extra = @{})
    $Payload = @{
        state = $State
        run_name = 'r006_label_shuffle'
        process_id = $PID
        updated_at = (Get-Date).ToString('o')
        run_log = $RunLog
        output_dir = $OutputDirectory
    }
    foreach ($Key in $Extra.Keys) { $Payload[$Key] = $Extra[$Key] }
    Write-AtomicJson -Path $StatusPath -Payload $Payload
}

function Test-CompletedRun {
    $SummaryPath = Join-Path $OutputDirectory 'protocol_summary.json'
    if (-not (Test-Path -LiteralPath $SummaryPath)) { return $false }
    try {
        $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
        return $Summary.state -eq 'completed' -and $Summary.training_results.Count -eq 40
    }
    catch { return $false }
}

try {
    if (Test-CompletedRun) {
        Write-Status -State 'completed' -Extra @{ note = 'already completed' }
        exit 0
    }

    for ($Attempt = 1; $Attempt -le $MaxAttemptsPerRun; $Attempt++) {
        Write-Status -State 'running' -Extra @{
            attempt = $Attempt
            max_attempts = $MaxAttemptsPerRun
        }
        "[$(Get-Date -Format o)] r006_label_shuffle attempt $Attempt/$MaxAttemptsPerRun" |
            Tee-Object -FilePath $RunLog -Append

        & $Python -u $Script `
            --all-subjects `
            --split-seed 20260815 `
            --train-seed 20260815 `
            --label-shuffle-seed 20260816 `
            --epochs 60 `
            --patience 15 `
            --min-epochs 30 `
            --batch-size 64 `
            --selection-metric macro_f1 `
            --class-weight-power 0.5 `
            --resume `
            --output-dir $OutputDirectory `
            --checkpoint-dir $CheckpointDirectory 2>&1 |
            Tee-Object -FilePath $RunLog -Append
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -eq 0 -and (Test-CompletedRun)) {
            Write-Status -State 'completed' -Extra @{ completed_at = (Get-Date).ToString('o') }
            exit 0
        }

        "[$(Get-Date -Format o)] r006_label_shuffle failed with exit code $ExitCode" |
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
