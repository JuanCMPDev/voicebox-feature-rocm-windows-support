<#
.SYNOPSIS
  Ejecuta el loop /tts-loop sin supervisión: una tarea por sesión de Claude Code, con contexto limpio cada vez.

.DESCRIPTION
  Antes de cada iteración consulta el estado con loopctl. Se detiene cuando una tarea espera una decisión
  tuya, cuando una tarea queda bloqueada, cuando el backlog se completa o cuando falla varias veces seguidas.
  Cada iteración queda registrada en perf-loop\logs\iter-<fecha>.json.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File perf-loop\run-loop.ps1
  powershell -ExecutionPolicy Bypass -File perf-loop\run-loop.ps1 -MaxIterations 5 -Model opus
#>
param(
  [int]$MaxIterations = 12,
  [int]$MaxConsecutiveFailures = 2,
  [int]$MaxTurns = 300,
  [string]$Model = "",
  [double]$MaxBudgetUsd = 0,
  [string]$PermissionMode = "acceptEdits"
)

# "Continue": en Windows PowerShell 5.1, con "Stop", cualquier línea de stderr de un ejecutable
# redirigida con 2>&1 se convierte en error terminante. Comprobamos $LASTEXITCODE a mano.
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root "backend\venv\Scripts\python.exe"
$ctl = Join-Path $root ".claude\skills\tts-loop\loopctl.py"
$logDir = Join-Path $root "perf-loop\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $py)) { throw "No encuentro $py (el junction backend\venv del worktree)." }
if (-not (Get-Command claude -ErrorAction SilentlyContinue)) { throw "No encuentro el CLI 'claude' en el PATH." }

# perf-loop\STOP detiene el loop entre tareas. Lanzar el runner es la orden de seguir, así que lo quita.
$stopFile = Join-Path $root "perf-loop\STOP"
if (Test-Path $stopFile) {
  Remove-Item -LiteralPath $stopFile -Force
  Write-Host "Quité perf-loop\STOP de una parada anterior."
}

& $py $ctl doctor --quick
if ($LASTEXITCODE -ne 0) { throw "loopctl doctor encontró fallos; corrígelos antes de lanzar el loop." }

Write-Host "Para parar entre tareas sin cortar ninguna: New-Item perf-loop\STOP (desde otra terminal, en esta carpeta)."

$fails = 0
for ($i = 1; $i -le $MaxIterations; $i++) {
  # `next` (no `status`) para que el estado quede guardado: p. ej. una tarea que pide visto bueno
  # para empezar pasa a «awaiting_human» y tu `/tts-loop approve` la encuentra así.
  $st = (& $py $ctl next --json | Out-String) | ConvertFrom-Json
  if ($st.stop) {
    Write-Host ""
    Write-Host "Loop detenido ($($st.reason)): $($st.message)"
    break
  }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $log = Join-Path $logDir "iter-$stamp.json"
  Write-Host "[$i/$MaxIterations] $(Get-Date -Format HH:mm) tarea $($st.task) · $($st.action)"

  $cargs = @("-p", "/tts-loop next", "--permission-mode", $PermissionMode, "--output-format", "json",
             "--max-turns", "$MaxTurns")
  if ($Model) { $cargs += @("--model", $Model) }
  if ($MaxBudgetUsd -gt 0) { $cargs += @("--max-budget-usd", "$MaxBudgetUsd") }

  & claude @cargs 2>&1 | Out-File -FilePath $log -Encoding utf8
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    $fails++
    Write-Warning "La iteración $i terminó con código $code (ver $log)."
    if ($fails -ge $MaxConsecutiveFailures) { Write-Warning "Demasiados fallos seguidos; paro."; break }
  } else {
    $fails = 0
  }
}

Write-Host ""
& $py $ctl status
[Console]::Beep(880, 300)
