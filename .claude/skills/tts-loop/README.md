# /tts-loop · loop agéntico para el plan de velocidad de Qwen3-TTS

Aplica el plan `perf-loop/plan.md` (fases 0–4) tarea a tarea, en ramas locales, midiendo cada cambio con el mismo banco y sin dejar pasar nada que empeore el audio. Se activa con un comando en Claude Code: `/tts-loop`.

## Cómo está montado

```
 tú ──/tts-loop──▶ orquestador (SKILL.md)        no escribe código; sigue un protocolo fijo
                     │
                     ├─▶ tts-implementer         escribe el cambio + tests, commit en la rama de la tarea
                     ├─▶ tts-reviewer            revisa el diff en frío (contexto limpio, sin Edit/Write)
                     └─▶ loopctl.py              determinista: estado, ramas, banco, comparación, puertas
                              │
                              ├─ scripts/bench_tts.py     (lo crea la tarea 0.2, se congela en la 0.5)
                              ├─ scripts/compare_wavs.py  (0.3)  SNR · similitud de locutor · WER · duración
                              └─ gate.json  →  PASS | FAIL | NEEDS_HUMAN
 hooks de Claude Code: guardarraíles (Bash, Edit/Write), contexto al iniciar sesión, no parar a medias,
                       y registrar tus decisiones (approve/ab/…) para que el agente no pueda tomarlas.
```

Prácticas en las que se apoya:

- **El modelo decide, el código verifica.** Las puertas son comprobaciones numéricas que ejecuta `loopctl` a partir de lo que mide el banco; el agente no puede declarar una tarea superada ni editar los umbrales.
- **Una tarea por iteración y estado en archivos** (`perf-loop/backlog.json`, `progress.md`, `RESULTS.md` y git), para retomar con contexto limpio en cualquier momento. Es el patrón del artículo de Anthropic sobre arneses para agentes de larga duración.
- **Escritor y revisor separados.** El revisor no ve cómo se escribió el cambio.
- **Arnés de evaluación congelado.** Tras aprobar la línea base, el banco, el comparador, las frases y la lista de tests que ya fallaban quedan bloqueados (hook y hash SHA-256 comprobado en cada puerta).
- **Ruido medido antes de decidir.** La línea base se mide dos veces (A/A); un speedup dentro del ruido sale como aviso.
- **Persona en los puntos que lo piden**: aprobar la línea base, escuchas ciegas de los cambios de tipo 2, puertas de fase, y empezar la 3.5 (3–5 días).
- **Aislamiento.** Todo ocurre en un `git worktree` aparte; tu copia de trabajo y la app no se tocan, no hay `push` y no se instala nada. El venv es el del repo principal (enlazado), así que los hooks impiden escribir en `backend/venv/` y la congelación guarda además un hash del paquete `qwen_tts`.
- **El evaluador no depende del código evaluado.** El banco habla con el servidor por HTTP y el comparador no importa nada de `backend/`.

## Puesta en marcha

0. **Instalar el skill** (una vez). El paquete está en `.agents\skills\tts-loop\`; Claude Code carga los skills de `.claude\skills\`. Desde `E:\VoiceBox\voicebox`:
   `backend\venv\Scripts\python.exe .agents\skills\tts-loop\loopctl.py install`
1. En el repo principal (`E:\VoiceBox\voicebox`), abre Claude Code y escribe `/tts-loop init`. Te pregunta el perfil de voz del banco, la carpeta de modelos, si traer los cambios locales de `backend/` y dónde crear el worktree. Después:
   - crea `E:\VoiceBox\voicebox-perf` desde tu commit actual, en la rama `perf/qwen-f0`;
   - enlaza `backend\venv` del repo principal (junction; no se duplica nada);
   - copia el andamiaje (`.claude/`, `perf-loop/`) y lo commitea.
   También puedes hacerlo a mano: `backend\venv\Scripts\python.exe .claude\skills\tts-loop\loopctl.py init --dry-run`.
2. Abre Claude Code en `E:\VoiceBox\voicebox-perf` y escribe `/tts-loop`. Avanza hasta 6 tareas o hasta que necesite algo de ti.
3. Sin supervisión (una tarea por sesión, contexto limpio cada vez), desde el worktree:
   `powershell -ExecutionPolicy Bypass -File perf-loop\run-loop.ps1`
   Se detiene solo cuando te toca decidir, cuando algo se bloquea o al terminar.

Antes de medir, cierra la app de Voicebox: compite por la GPU y falsea el banco (`loopctl doctor` avisa).

## Comandos

| En Claude Code | Qué hace |
| --- | --- |
| `/tts-loop` · `/tts-loop run 3` | Avanza tareas (por defecto hasta 6). |
| `/tts-loop next` | Una sola tarea (lo usa el runner). |
| `/tts-loop status` | Estado del backlog y siguiente paso. |
| `/tts-loop approve <id> [nota]` | Aprueba lo que espera tu revisión (línea base, puerta de fase, inicio de la 3.5, tarea manual). |
| `/tts-loop reject <id> <motivo>` | Lo rechaza. Un experimento se descarta; una tarea obligatoria queda bloqueada. |
| `/tts-loop ab <id>` · `/tts-loop ab <id> <código>` | Escucha ciega: abre la página y pega el código que genera. |
| `/tts-loop retry <id>` · `/tts-loop skip <id> <motivo>` | Más intentos para una tarea bloqueada, u omitirla. |
| `/tts-loop refreeze <motivo>` | Aceptas un cambio deliberado en el arnés congelado. |
| `/tts-loop pause` · `/tts-loop export` | Pausar a mitad de tarea · rama limpia final. |
| `/tts-loop teardown` (en el repo principal) | Borra el worktree de forma segura (ver «Si algo va mal»). |

Las decisiones (approve, reject, ab, skip, retry, refreeze) solo valen si las escribes tú: un hook registra tu mensaje y `loopctl` exige ese registro. Desde tu terminal también puedes ejecutarlas con `--i-am-human`.

## Ramas

- `perf/qwen-f0` … `perf/qwen-f4`: una por fase, cada una sale de la anterior (lo que pide el plan: «el diff en una rama por fase»).
- `perf/qwen-task/<id>`: una por tarea. Si pasa, avanza la rama de fase (fast-forward). Si se rechaza, se conserva para consultarla y la fase sigue sin ella.
- `/tts-loop export` crea `perf/qwen-export` sobre tu commit original: un commit por fase, sin `.claude/` ni `perf-loop/`, lista para fusionar donde quieras.

## Puertas de calidad

Cada tarea tiene un tipo de calidad del plan y loopctl aplica sus umbrales:

| Tipo | Qué cambia | Puerta |
| --- | --- | --- |
| 1 | Nada de la aritmética (MIOpen, cachés, colas, calentamiento) | SNR > 50 dB frente al último estado aceptado; duración ±5 % |
| 3 | Menos trabajo del códec con igual o más contexto | SNR > 40 dB; similitud de locutor ≥ base (tolerancia 0,002); WER no empeora |
| 2 | Cómo se ejecuta la misma aritmética (kernels, grafos) | Similitud de locutor −0,01 como mucho; WER no empeora; si la onda cambia, escucha ciega de 6 pares (la base no puede ganar en más de 3) |

Además, cada tarea tiene sus comprobaciones del plan (por ejemplo «Find < 3 s por frase nueva», «etapa reference < 50 ms», «la Find-Db no crece», «≥ 5 % más rápido» en los experimentos de la fase 3), tests sin fallos nuevos frente a la línea base y ruff sin errores graves en los archivos tocados.

La SNR se compara con el último estado aceptado y las métricas de voz y WER con la línea base, para que los cambios pequeños no se acumulen sin control.

Los umbrales viven en `perf-loop/backlog.json` (`defaults.quality` y `gate.checks` de cada tarea). El agente no puede editarlos; tú sí, a mano. Después, `loopctl gate <id>` reevalúa sin volver a medir.

## Duración

La línea base (0.5) mide 12 frases dos veces con una Find-Db vacía: con el código actual, cada frase nueva tarda 137–162 s, así que son 1–2 horas. Después de la fase 1, un banco completo dura unos minutos. `loopctl` mantiene el PC despierto mientras mide.

## Archivos

| Ruta (en el worktree) | Qué es |
| --- | --- |
| `.claude/skills/tts-loop/` | Este skill: `SKILL.md` (orquestador), `loopctl.py`, este README |
| `.claude/agents/` | `tts-implementer`, `tts-reviewer` |
| `.claude/settings.json` | Permisos, hooks y entorno (desactiva el MCP de la app en vivo) |
| `perf-loop/backlog.json` | Tareas, estado y umbrales (solo lo escribe loopctl) |
| `perf-loop/config.json` | Rutas, perfil, puerto del banco (17599), comandos de tests y lint |
| `perf-loop/contracts.md` | Qué deben hacer y escribir `bench_tts.py`, `compare_wavs.py` y las líneas VBPERF |
| `perf-loop/progress.md`, `RESULTS.md`, `results.jsonl` | Bitácora y tabla del banco por tarea |
| `perf-loop/notes/` | Planes y notas de los agentes |
| `perf-loop/bench/<etiqueta>/` | WAV, `bench.json`, comparaciones, `gate.json`, logs (no va a git) |
| `perf-loop/run-loop.ps1` | Runner sin supervisión |

## Si algo va mal

- `loopctl doctor` (sin `--quick`) comprueba Python, git, venv, hooks (incluida una prueba real del hook de Bash), puertos, disco, torch/HIP y el arnés congelado.
- `loopctl show <id>` muestra la ficha, los intentos anteriores y por qué fallaron; los logs están en `perf-loop/bench/<etiqueta>/logs/`.
- Si el banco muere, `loopctl wait` lo detecta. Tras arreglar la causa, `/tts-loop retry <id>` vuelve a medir. Si el proceso en segundo plano no sobrevive en tu máquina, ejecuta `loopctl verify <id> --foreground` desde tu terminal y deja que el loop espere con `wait`.
- Para empezar de cero: desde el repo principal, `/tts-loop teardown` (o `loopctl teardown --yes`), borra las ramas `perf/qwen-*` si ya no las quieres y vuelve a ejecutar `/tts-loop init`. **No uses `git worktree remove` ni borres la carpeta del worktree con el Explorador**: en Windows seguirían el junction `backend\venv` y borrarían el venv del repo principal. `teardown` quita primero el junction y comprueba que el venv sigue ahí.

## Límites conocidos

- Las referencias de línea del plan se escribieron sobre otro checkout; el implementador localiza cada punto por contenido.
- `bench_tts.py` y `compare_wavs.py` todavía no existen: son las tareas 0.2 y 0.3, y las puertas de la fase 0 comprueban que cumplen el contrato.
- La vía WSL2 + vLLM (4.3) es una decisión tuya; el loop no instala nada.
- Los hooks se ejecutan con `backend\venv\Scripts\python.exe`; si ese Python no arranca, los guardarraíles dejan de funcionar sin bloquear la sesión. `loopctl doctor` lo comprueba.
