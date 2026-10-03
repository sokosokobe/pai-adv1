<#
.SYNOPSIS
    Run the project checks (lint + tests).

.DESCRIPTION
    Default (fast check):
        1. ruff format --check
        2. ruff check
        3. pytest            (isaac-marked tests are excluded by pyproject.toml)
    With -Sim (full check), additionally:
        4. pytest -m isaac   (Isaac Sim integration tests; slow, needs GPU)

    Every step runs even if an earlier one fails, then a summary is printed.
    Exit code is 0 only if all steps passed, otherwise 1.

    NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads BOM-less
    files as ANSI (cp932), which garbles non-ASCII text.

.EXAMPLE
    .\scripts\check.ps1
.EXAMPLE
    .\scripts\check.ps1 -Sim
#>
[CmdletBinding()]
param(
    [switch]$Sim
)

# Native tools report failure via exit codes; do not turn their stderr into terminating errors.
$ErrorActionPreference = 'Continue'

# ---- Resolve paths (works from any current directory) ----
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) {
    Write-Host "[check] Python not found: $Python" -ForegroundColor Red
    exit 1
}

# ---- Step definitions ----
$steps = @(
    @{ Name = 'ruff format --check'; Args = @('-m', 'ruff', 'format', '--check', '.') },
    @{ Name = 'ruff check';          Args = @('-m', 'ruff', 'check', '.') },
    @{ Name = 'pytest';              Args = @('-m', 'pytest') }
)
if ($Sim) {
    $steps += @{ Name = 'pytest -m isaac'; Args = @('-m', 'pytest', '-m', 'isaac') }
    if (-not $env:OMNI_KIT_ACCEPT_EULA) { $env:OMNI_KIT_ACCEPT_EULA = 'YES' }
}

# ---- Run ----
# Python prints UTF-8 (Japanese messages); make the console decode it correctly.
$prevOutputEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$results = @()
Push-Location $RepoRoot
try {
    foreach ($step in $steps) {
        Write-Host ''
        Write-Host "===== $($step.Name) =====" -ForegroundColor Cyan
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $Python @($step.Args)
        $code = $LASTEXITCODE
        $sw.Stop()
        $results += [pscustomobject]@{
            Step     = $step.Name
            Result   = $(if ($code -eq 0) { 'PASS' } else { 'FAIL' })
            ExitCode = $code
            Seconds  = [math]::Round($sw.Elapsed.TotalSeconds, 1)
        }
    }
}
finally {
    Pop-Location
    [Console]::OutputEncoding = $prevOutputEncoding
}

# ---- Summary ----
Write-Host ''
Write-Host '===== summary =====' -ForegroundColor Cyan
foreach ($r in $results) {
    $color = $(if ($r.Result -eq 'PASS') { 'Green' } else { 'Red' })
    Write-Host ('{0,-4}  {1,-22} exit={2,-3} {3,6:N1}s' -f $r.Result, $r.Step, $r.ExitCode, $r.Seconds) -ForegroundColor $color
}

$failed = @($results | Where-Object { $_.Result -ne 'PASS' })
if ($failed.Count -gt 0) {
    Write-Host "[check] FAILED: $($failed.Count) step(s)" -ForegroundColor Red
    exit 1
}
Write-Host '[check] all steps passed' -ForegroundColor Green
exit 0
