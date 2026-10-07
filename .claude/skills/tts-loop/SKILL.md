---
name: tts-loop
description: Orquesta el loop agéntico que aplica el plan de velocidad de Qwen3-TTS (fases 0–4) en ramas locales, con banco, puertas de calidad, revisión independiente y decisiones humanas. Solo se invoca a mano con /tts-loop.
argument-hint: "[init | run [N] | next | status | approve <id> [nota] | reject <id> <motivo> | ab <id> [código] | retry <id> | skip <id> <motivo> | refreeze <motivo> | pause | export | teardown]"
disable-model-invocation: true
allowed-tools: Bash(backend/venv/Scripts/python.exe .claude/skills/tts-loop/loopctl.py *), Agent, Read, Grep, Glob, AskUserQuestion
---

# /tts-loop · orquestador

Estado actual del loop:

!`backend/venv/Scripts/python.exe .claude/skills/tts-loop/loopctl.py brief`

Argumentos recibidos: `$ARGUMENTS`

En adelante, `loopctl` = `backend/venv/Scripts/python.exe .claude/skills/tts-loop/loopctl.py`, ejecutado desde la raíz del proyecto.

## Reglas fijas

1. **Orquestas; no escribes código.** El código lo escribe el subagente `tts-implementer`; la revisión, `tts-reviewer`, en contexto limpio; ramas, estado, banco, comparación de audio y puertas, `loopctl`. No tomes atajos aunque parezcan obvios.
2. **Una tarea cada vez**, en el orden que da `loopctl next`. Ninguna se cierra sin una puerta evaluada por `loopctl verify`.
3. **Las decisiones humanas son del usuario.** No ejecutes approve, reject, ab, skip, retry ni refreeze por iniciativa propia: loopctl las rechaza si no las escribió él en su mensaje. Si te las pide en lenguaje natural, dile el comando exacto (`/tts-loop approve 0.5`).
4. **No toques `perf-loop/` ni `.claude/`** salvo con `loopctl note`. Si un hook te bloquea, no lo rodees: cambia de enfoque o bloquea la tarea con su motivo.
5. **Esperas largas**: `loopctl wait` con timeout 600000 en Bash, repetido mientras imprima `ESTADO: running`. Entre llamadas no hagas otra cosa.
6. Contesta al usuario en español y breve: qué tarea, qué pasó, qué número importa, qué decide él si toca.

## Despacho

| `$ARGUMENTS` | Qué haces |
| --- | --- |
| vacío o `run [N]` | Iteraciones (abajo) hasta N tareas cerradas (por defecto `max_tasks_per_run` de `perf-loop/config.json`) o hasta una parada. |
| `next` | Exactamente una iteración. Es el modo del runner sin supervisión: no preguntes nada; si necesitas al usuario, deja la tarea en `await` o `block`. |
| `init` | Sección «Inicializar». |
| `status` | `loopctl status` y resume en 3–5 líneas. |
| `approve <id> [nota]` | `loopctl approve <id> --note "<nota>"` |
| `reject <id> <motivo>` | `loopctl reject <id> --reason "<motivo>"` |
| `ab <id>` | `loopctl ab <id>` y explica cómo usar la página de escucha ciega. |
| `ab <id> <código>` | `loopctl ab <id> <código>` |
| `retry <id>` | `loopctl retry <id>` |
| `skip <id> <motivo>` | `loopctl skip <id> --reason "<motivo>"` |
| `refreeze <motivo>` | `loopctl refreeze --reason "<motivo>"` |
| `pause` | `loopctl pause` y resume dónde quedó. |
| `export` | `loopctl export` y explica la rama resultante. |
| `teardown` | Solo desde el repo principal y solo si el usuario lo pidió: confirma con AskUserQuestion que quiere borrar el worktree (y sus WAV) y ejecuta `loopctl teardown --yes`. Nunca `git worktree remove` a mano: en Windows seguiría el junction y borraría el venv del repo principal. |

Tras un approve, ab, retry o skip que salga bien, pregunta si sigue con `run`.

## Iteración

1. **Orientarse.** `loopctl doctor --quick`: si hay líneas `FAIL`, para y explícalas. Después `loopctl next --json`.
   - `"stop": true` → cuenta al usuario el `message` (qué tiene que decidir y con qué comando) y termina.
2. **Según `action`:**
   - `begin` → `loopctl begin <id>` y continúa con su `next_action`.
   - `implement` → paso 3 · `verify` → paso 4 (revisión, si la tarea tiene implementador) y luego 5 · `wait` o `decide` → paso 6 (imprime el veredicto) y 7.
   - `await` → `loopctl await <id>`, explica al usuario lo que tiene que hacer y termina.
   - `verify_failed` → `loopctl show <id>` y los logs de `perf-loop/bench/<etiqueta>/logs/`. Si la causa es del entorno (puerto ocupado, GPU, servidor que no arranca, falta un modelo): en modo interactivo, pregunta al usuario si ya está resuelto y, si lo está, paso 5; si no, o en modo `next`, `loopctl block <id> --reason "…"`. Si es del código, paso 3 con el error literal.
3. **Implementar** (solo tareas con implementador). Lanza `tts-implementer` con: «Tarea <id> del loop /tts-loop. Empieza con `loopctl show <id> --for implementer`.» y, si hay, el feedback pendiente literal (hallazgos del revisor o fallos de la última puerta). Lee su JSON:
   - `blocked` → `loopctl block <id> --reason "<notes>"` y termina.
   - `noop` → solo vale en tareas que lo admiten (`loopctl finish` lo comprueba). Paso 4 con la evidencia; si el revisor la acepta, `loopctl finish <id> noop --reason "<noop_evidence>"`.
   - `done` → paso 4.
   - Respuesta incompleta o sin JSON (por ejemplo, se quedó sin turnos): mira `git status` y `git log`, relanza el implementador una vez con «continúa la tarea <id>» y, si vuelve a pasar, `loopctl block`.
4. **Revisar** (siempre que haya cambios de código o un `noop`). Lanza `tts-reviewer` con: «Revisa la tarea <id> del loop /tts-loop. Empieza con `loopctl show <id> --for reviewer`.» Registra `loopctl note <id> "review: <verdict> — <summary>"`. Si `changes_requested`, vuelve a 3 con los hallazgos blocker y major. Máximo 2 rondas de revisión por intento; a la tercera, `loopctl block <id> --reason "revisión sin acuerdo: …"`.
5. **Verificar.** `loopctl verify <id>` (gasta un intento; lanza banco, comparación, tests y lint en segundo plano).
6. **Esperar.** `loopctl wait <id>` (timeout 600000) hasta `ESTADO: done` o un error.
7. **Decidir** según el veredicto impreso:
   - `PASS` → `loopctl finish <id> pass`.
   - `NEEDS_HUMAN` → `loopctl await <id>`. Explica al usuario qué mirar (la página A/B, la rama o las métricas) y el comando exacto. Termina.
   - `FAIL` → `loopctl note <id> "<causa en una frase>"`. Si quedan intentos y la causa se corrige en código, vuelve a 3 con los fallos literales. Si no: experimento o medida → `loopctl finish <id> reject --reason "<causa>"`; tarea obligatoria → `loopctl block <id> --reason "<causa>"` y termina.
8. **Cierre.** Al usuario, 2–3 líneas: tarea, veredicto y la cifra clave (warm medio, speedup o SNR). En modo `run`, siguiente iteración.

Si el usuario pide parar a mitad de tarea: `loopctl pause` y termina.

## Inicializar (`init`, desde el repo principal)

1. `loopctl init --dry-run` y lee el JSON.
2. Pregunta con AskUserQuestion, todo en una sola vez:
   - **Perfil de voz** del banco (de `perfiles`). Si la lista está vacía o ese data dir no es el que usa su app, pídele la ruta del data dir de la app y ejecuta `loopctl profiles --data-dir <ruta>`.
   - **Carpeta de modelos**: si su app guarda los modelos en una carpeta propia (`VOICEBOX_MODELS_DIR`), su ruta; si no, la caché por defecto de Hugging Face.
   - **Cambios locales sin commit** de `backend/`: si se traen al worktree (recomendado: sí; incluye `backend/utils/platform_detect.py`, la detección de GPU AMD).
   - **Ruta del worktree** (por defecto `<repo>-perf`, junto al repo).
3. `loopctl init --profile-id <id> --source-data-dir <ruta> [--models-dir <ruta>] [--carry-backend] [--worktree <ruta>]`.
4. Explica el siguiente paso: abrir Claude Code en la carpeta del worktree y escribir `/tts-loop` (interactivo), o lanzar `powershell -ExecutionPolicy Bypass -File perf-loop\run-loop.ps1` desde ella (sin supervisión).
