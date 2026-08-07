param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$SuiteRoot,
    [int]$MaxParallel = 2
)

$ErrorActionPreference = 'Stop'
$scenarios = @(
    'direct-invoke', 'direct-stream', 'local-invoke', 'local-stream',
    'dynamic-model-invoke', 'dynamic-model-stream',
    'react-tool-invoke', 'react-tool-stream',
    'sqlite-durable-invoke', 'sqlite-durable-stream',
    'http-worker-invoke', 'http-worker-stream'
)
$state = [ordered]@{}
$running = @{}
$uv = (Get-Command uv -ErrorAction Stop).Source
$createdAt = [DateTimeOffset]::UtcNow.ToString('o')

New-Item -ItemType Directory -Force -Path $SuiteRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $SuiteRoot 'results') | Out-Null

foreach ($scenario in $scenarios) {
    $state[$scenario] = [ordered]@{
        state = 'pending'
        started_at = $null
        completed_at = $null
        exit_code = $null
        result = $null
        stdout_log = "$scenario.stdout.log"
        stderr_log = "$scenario.stderr.log"
    }
}

function Write-Suite([string]$outcome, [string]$completedAt = $null) {
    $payload = [ordered]@{
        schema = 'pygent.live-suite.v1'
        suite_id = Split-Path $SuiteRoot -Leaf
        created_at = $createdAt
        completed_at = $completedAt
        outcome = $outcome
        max_parallel = $MaxParallel
        scenario_count = $scenarios.Count
        configuration = [ordered]@{
            profile = 'live-long'
            credentials_configured = $true
            gate = 'live-smoke-completed'
        }
        scenarios = $state
    }
    $target = Join-Path $SuiteRoot 'suite.json'
    $temporary = "$target.tmp"
    [System.IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 8),
        [System.Text.UTF8Encoding]::new($false)
    )
    Move-Item -Force -LiteralPath $temporary -Destination $target
}

function Start-Scenario([string]$scenario) {
    $outputRoot = Join-Path (Join-Path $SuiteRoot 'results') $scenario
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $stdout = Join-Path $SuiteRoot $state[$scenario].stdout_log
    $stderr = Join-Path $SuiteRoot $state[$scenario].stderr_log
    $arguments = @(
        'run', '--extra', 'performance', '--env-file', '.env',
        'python', '-m', 'benchmarks', 'run', "live-long-$scenario",
        '--confirm-live', '--output', $outputRoot
    )
    $process = Start-Process -FilePath $uv -ArgumentList $arguments `
        -WorkingDirectory $RepoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $state[$scenario].state = 'running'
    $state[$scenario].started_at = [DateTimeOffset]::UtcNow.ToString('o')
    $running[$scenario] = $process
}

Write-Suite 'running'
try {
    while (($state.Values | Where-Object { $_.state -in @('pending', 'running') }).Count -gt 0) {
        foreach ($scenario in @($running.Keys)) {
            $process = $running[$scenario]
            $process.Refresh()
            if (-not $process.HasExited) { continue }
            # Ensure Start-Process has populated ExitCode before recording it.
            $process.WaitForExit()
            $process.Refresh()
            $resultRoot = Join-Path (Join-Path $SuiteRoot 'results') $scenario
            $result = Get-ChildItem -LiteralPath $resultRoot -Directory -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
            # A missing native exit code must never be coerced from $null to 0.
            $exitCode = if ($null -eq $process.ExitCode) { -1 } else { [int]$process.ExitCode }
            $state[$scenario].completed_at = [DateTimeOffset]::UtcNow.ToString('o')
            $state[$scenario].exit_code = $exitCode
            $state[$scenario].result = if ($null -eq $result) { $null } else { $result.FullName }
            $state[$scenario].state = if ($exitCode -eq 0) { 'completed' } else { 'incomplete' }
            $running.Remove($scenario)
            Write-Suite 'running'
        }
        $slots = $MaxParallel - $running.Count
        if ($slots -gt 0) {
            $pending = @($scenarios | Where-Object { $state[$_].state -eq 'pending' } | Select-Object -First $slots)
            foreach ($scenario in $pending) { Start-Scenario $scenario }
            if ($pending.Count -gt 0) { Write-Suite 'running' }
        }
        Start-Sleep -Seconds 5
    }
    $failed = @($state.Values | Where-Object { $_.state -ne 'completed' })
    $outcome = if ($failed.Count -eq 0) { 'completed' } else { 'incomplete' }
    Write-Suite $outcome ([DateTimeOffset]::UtcNow.ToString('o'))
    if ($failed.Count -gt 0) { exit 3 }
    exit 0
}
catch {
    Write-Suite 'incomplete' ([DateTimeOffset]::UtcNow.ToString('o'))
    throw
}
