param(
    [int]$MaxAttemptsPerRun = 3,
    [int]$RetryDelaySeconds = 30
)

$ErrorActionPreference = 'Stop'
$ResearchRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python = 'python'
$Experiment = Join-Path $ResearchRoot 'experiments\bspc_revision\leakage_free_db2.py'
$GateScript = Join-Path $ResearchRoot 'experiments\bspc_revision\evaluate_r005a_gate.py'
$DiagnosticRoot = Join-Path $ResearchRoot 'results\bspc_revision_v2\r005a_balanced_diagnostic'
$DiagnosticCheckpointRoot = Join-Path $ResearchRoot 'checkpoints_bspc_v2\r005a_balanced_diagnostic'
$FullOutputDirectory = Join-Path $ResearchRoot 'results\bspc_revision_v2\r005b_balanced_full_seed20260815'
$FullCheckpointDirectory = Join-Path $ResearchRoot 'checkpoints_bspc_v2\r005b_balanced_full_seed20260815'
$LogDirectory = Join-Path $ResearchRoot 'logs\bspc_revision_v2'
$SupervisorLog = Join-Path $LogDirectory 'r005a_supervisor.log'
$SupervisorStatusPath = Join-Path $DiagnosticRoot 'supervisor_status.json'
$DiagnosticSubjects = @(1, 10, 15, 17, 27, 40)
$TrainSeeds = @(20260815, 20260816, 20260817)

New-Item -ItemType Directory -Force -Path `
    $DiagnosticRoot, $DiagnosticCheckpointRoot, $FullOutputDirectory, `
    $FullCheckpointDirectory, $LogDirectory | Out-Null

function Write-AtomicJson {
    param([string]$Path, [hashtable]$Payload)
    $TemporaryPath = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $TemporaryPath -Encoding utf8
    Move-Item -LiteralPath $TemporaryPath -Destination $Path -Force
}

function Write-SupervisorStatus {
    param(
        [string]$State,
        [hashtable]$Extra = @{}
    )
    $Payload = @{
        state = $State
        process_id = $PID
        updated_at = (Get-Date).ToString('o')
        supervisor_log = $SupervisorLog
    }
    foreach ($Key in $Extra.Keys) {
        $Payload[$Key] = $Extra[$Key]
    }
    Write-AtomicJson -Path $SupervisorStatusPath -Payload $Payload
}

function Test-CompletedRun {
    param(
        [string]$OutputDirectory,
        [int]$ExpectedSubjects
    )
    $SummaryPath = Join-Path $OutputDirectory 'protocol_summary.json'
    if (-not (Test-Path -LiteralPath $SummaryPath)) {
        return $false
    }
    try {
        $Summary = Get-Content -Raw -LiteralPath $SummaryPath | ConvertFrom-Json
        return $Summary.state -eq 'completed' -and $Summary.training_results.Count -eq $ExpectedSubjects
    }
    catch {
        return $false
    }
}

function Invoke-TrainingRun {
    param(
        [string]$RunName,
        [int]$TrainSeed,
        [string]$OutputDirectory,
        [string]$CheckpointDirectory,
        [int[]]$Subjects,
        [switch]$AllSubjects,
        [switch]$SkipWindowManifests
    )

    $ExpectedSubjects = if ($AllSubjects) { 40 } else { $Subjects.Count }
    if (Test-CompletedRun -OutputDirectory $OutputDirectory -ExpectedSubjects $ExpectedSubjects) {
        "[$(Get-Date -Format o)] $RunName already completed" | Tee-Object -FilePath $SupervisorLog -Append
        return $true
    }

    $RunLog = Join-Path $LogDirectory "$RunName.log"
    for ($Attempt = 1; $Attempt -le $MaxAttemptsPerRun; $Attempt++) {
        Write-SupervisorStatus -State 'running' -Extra @{
            run_name = $RunName
            train_seed = $TrainSeed
            attempt = $Attempt
            max_attempts = $MaxAttemptsPerRun
            run_log = $RunLog
        }
        "[$(Get-Date -Format o)] $RunName attempt $Attempt/$MaxAttemptsPerRun" |
            Tee-Object -FilePath $SupervisorLog -Append |
            Tee-Object -FilePath $RunLog -Append

        $Arguments = @(
            '-u', $Experiment,
            '--epochs', '60',
            '--patience', '15',
            '--min-epochs', '30',
            '--batch-size', '64',
            '--selection-metric', 'macro_f1',
            '--class-weight-power', '0.5',
            '--per-subject-seed',
            '--split-seed', '20260815',
            '--train-seed', $TrainSeed.ToString(),
            '--resume',
            '--output-dir', $OutputDirectory,
            '--checkpoint-dir', $CheckpointDirectory
        )
        if ($AllSubjects) {
            $Arguments += '--all-subjects'
        }
        else {
            $Arguments += '--subjects'
            $Arguments += $Subjects | ForEach-Object { $_.ToString() }
        }
        if ($SkipWindowManifests) {
            $Arguments += '--skip-window-manifests'
        }

        & $Python @Arguments 2>&1 |
            Tee-Object -FilePath $RunLog -Append |
            Tee-Object -FilePath $SupervisorLog -Append
        $ExitCode = $LASTEXITCODE

        if ($ExitCode -eq 0 -and (
            Test-CompletedRun -OutputDirectory $OutputDirectory -ExpectedSubjects $ExpectedSubjects
        )) {
            return $true
        }

        "[$(Get-Date -Format o)] $RunName failed with exit code $ExitCode" |
            Tee-Object -FilePath $SupervisorLog -Append |
            Tee-Object -FilePath $RunLog -Append
        if ($Attempt -lt $MaxAttemptsPerRun) {
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    return $false
}

try {
    Write-SupervisorStatus -State 'running_diagnostics' -Extra @{
        diagnostic_subjects = $DiagnosticSubjects
        train_seeds = $TrainSeeds
    }

    foreach ($TrainSeed in $TrainSeeds) {
        $OutputDirectory = Join-Path $DiagnosticRoot "seed_$TrainSeed"
        $CheckpointDirectory = Join-Path $DiagnosticCheckpointRoot "seed_$TrainSeed"
        $RunName = "r005a_balanced_seed_$TrainSeed"
        $Completed = Invoke-TrainingRun `
            -RunName $RunName `
            -TrainSeed $TrainSeed `
            -OutputDirectory $OutputDirectory `
            -CheckpointDirectory $CheckpointDirectory `
            -Subjects $DiagnosticSubjects `
            -SkipWindowManifests
        if (-not $Completed) {
            throw "$RunName exhausted all retry attempts"
        }
    }

    Write-SupervisorStatus -State 'evaluating_diagnostic_gate'
    & $Python -u $GateScript --diagnostic-root $DiagnosticRoot 2>&1 |
        Tee-Object -FilePath $SupervisorLog -Append
    $GateExitCode = $LASTEXITCODE
    $GatePath = Join-Path $DiagnosticRoot 'gate_summary.json'
    if (-not (Test-Path -LiteralPath $GatePath)) {
        throw "R005a gate did not produce $GatePath"
    }
    $Gate = Get-Content -Raw -LiteralPath $GatePath | ConvertFrom-Json

    if (-not $Gate.launch_full_run) {
        if ($Gate.PSObject.Properties['manual_hold'] -and $Gate.manual_hold) {
            Write-SupervisorStatus -State 'awaiting_manual_launch' -Extra @{
                gate_summary = $GatePath
                manual_hold = $true
                note = '用户要求门控后暂停；删除 tmp\hold_full_run.flag 后重新运行本监督器即可恢复（已完成种子自动跳过，门控重算后自动启动 R005b）'
            }
            "[$(Get-Date -Format o)] R005b held by user request (hold marker present); supervisor exiting" |
                Tee-Object -FilePath $SupervisorLog -Append
            exit 0
        }
        Write-SupervisorStatus -State 'diagnostic_gate_failed' -Extra @{
            gate_summary = $GatePath
            gate_exit_code = $GateExitCode
        }
        exit 2
    }

    Write-SupervisorStatus -State 'running_full' -Extra @{
        gate_summary = $GatePath
        full_output = $FullOutputDirectory
    }
    $FullCompleted = Invoke-TrainingRun `
        -RunName 'r005b_balanced_full_seed_20260815' `
        -TrainSeed 20260815 `
        -OutputDirectory $FullOutputDirectory `
        -CheckpointDirectory $FullCheckpointDirectory `
        -AllSubjects
    if (-not $FullCompleted) {
        throw 'R005b full balanced run exhausted all retry attempts'
    }

    Write-SupervisorStatus -State 'completed' -Extra @{
        gate_summary = $GatePath
        full_output = $FullOutputDirectory
        full_checkpoints = $FullCheckpointDirectory
        completed_at = (Get-Date).ToString('o')
    }
    exit 0
}
catch {
    $_ | Out-String | Tee-Object -FilePath $SupervisorLog -Append
    Write-SupervisorStatus -State 'failed' -Extra @{
        error = $_.Exception.Message
        failed_at = (Get-Date).ToString('o')
    }
    exit 1
}
