$ErrorActionPreference = 'Stop'
$ExperimentRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$StatusPath = Join-Path $ExperimentRoot 'results\bspc_revision_v2\r005a_balanced_diagnostic\supervisor_status.json'

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class R005APowerState {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@

$Continuous = [Convert]::ToUInt32('80000000', 16)
$SystemRequired = [uint32]0x00000001
$RequestedState = [uint32]($Continuous -bor $SystemRequired)
$TerminalStates = @('completed', 'failed', 'diagnostic_gate_failed')

try {
    if ([R005APowerState]::SetThreadExecutionState($RequestedState) -eq 0) {
        throw '无法设置 R005a 系统唤醒状态'
    }
    Write-Output "R005A_KEEP_AWAKE_STARTED pid=$PID"
    while ($true) {
        if (Test-Path -LiteralPath $StatusPath) {
            try {
                $Status = Get-Content -Raw -LiteralPath $StatusPath | ConvertFrom-Json
                if ($Status.state -in $TerminalStates) {
                    break
                }
                if ($Status.process_id) {
                    $Supervisor = Get-Process -Id $Status.process_id -ErrorAction SilentlyContinue
                    if (-not $Supervisor) {
                        break
                    }
                }
            }
            catch {
                Write-Output "R005A_KEEP_AWAKE_STATUS_RETRY $($_.Exception.Message)"
            }
        }
        Start-Sleep -Seconds 60
    }
}
finally {
    [void][R005APowerState]::SetThreadExecutionState($Continuous)
    Write-Output 'R005A_KEEP_AWAKE_RELEASED'
}
