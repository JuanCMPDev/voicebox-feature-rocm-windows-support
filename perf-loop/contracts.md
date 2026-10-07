# Contratos del arnés de evaluación

`loopctl` lanza estos scripts y lee lo que escriben para evaluar las puertas. Las tareas 0.1–0.4 los implementan; la 0.5 los congela. Si un script no cumple el contrato, la puerta falla.

Rutas relativas a la raíz del worktree. `<python>` es `backend/venv/Scripts/python.exe` (Windows) o `backend/venv/bin/python`.

## §1 · `scripts/bench_phrases.json`

```json
{
  "language": "es",
  "phrases": [
    {"id": "p01", "text": "…", "seed": 1101, "target_s": 3.2},
    {"id": "p02", "text": "…", "seed": 1102, "target_s": 3.8}
  ]
}
```

- Exactamente 12 frases, `id` con formato `pNN` y únicos, `seed` entero, `target_s` entre 3 y 12.
- Longitudes de texto todas distintas.
- `p01` y `p02` (las de humo, `config.bench.smoke_phrases`) son las más cortas.

## §2 · `scripts/bench_tts.py`

### CLI

```
<python> scripts/bench_tts.py --config perf-loop/config.json --tag <tag> --out perf-loop/bench/<tag>
    [--phrases all | p01,p02,…]   # por defecto all
    [--passes cold,warm]          # orden de ejecución; por defecto cold,warm
    [--miopen-db fresh|persist]   # por defecto persist
    [--env KEY=VALUE]…            # variables extra para el servidor (repetible)
    [--scenario contention|batch] [--concurrent N]
<python> scripts/bench_tts.py --self-test
```

Código de salida 0 solo si todas las frases terminaron en `completed`. Cualquier error, con salida distinta de 0 y mensaje claro en stderr.

`dirty` en bench.json es `true` si `git status --porcelain --untracked-files=all -- . ":(exclude)perf-loop"` devuelve algo. Los archivos de `perf-loop/` (estado del loop) no cuentan.

### Ciclo de vida

1. **Comprobaciones previas**: si `config.bench.port` está ocupado, aborta. Si `config.bench.user_app_port` (la app del usuario) responde, avisa en el log y sigue.
2. **Datos del banco**: `config.bench.data_dir` (relativo al worktree). Si no existe, créalo copiando de `config.bench.source_data_dir` la base `voicebox.db` y lo necesario del perfil `config.bench.profile_id` (sus muestras). Nunca escribas en `source_data_dir`. La caché del prompt de voz y la Find-Db propia (desde la 1.2) viven aquí y persisten entre ejecuciones.
3. **Una pasada = un proceso de servidor nuevo**: `<python> -m backend.main --port <port> --data-dir <data_dir>` con cwd en la raíz del worktree, entorno heredado + `config.bench.env` + `--env`, y en Windows `creationflags=CREATE_NO_WINDOW`. Captura stdout+stderr a `<out>/logs/server-<pasada>.log`. Espera a `/health` (máx. `config.bench.server_start_timeout_s`, por defecto 600 s), mide `load_s` hasta que el modelo esté cargado (incluido el calentamiento si existe) y ejecuta las frases **en serie** en el orden del archivo. Al final, apaga el servidor (terminación ordenada y, si no, árbol de procesos) y espera a que libere el puerto.
4. **Cada frase**: `POST /generate` con el texto, `profile_id`, `engine: "qwen"`, `model_size: "1.7B"`, `language: "es"` y su `seed` (comprueba los nombres reales de los campos en `backend/models.py`). Espera a `completed` sondeando el estado, descarga el WAV y guárdalo en `<out>/<pasada>/<id>.wav`. `wall_s` va desde el POST hasta `completed`.
5. **MIOpen**:
   - `fresh`: crea un directorio vacío `<out>/miopen-db/` y pasa `MIOPEN_USER_DB_PATH` y `MIOPEN_CUSTOM_CACHE_DIR` apuntando a él (la app usa `setdefault`, así que esto manda). Sirve para medir el coste real de cada longitud nueva.
   - `persist`: no toca esas variables; vale la ruta que fije la app o, si no fija ninguna, la de MIOpen por defecto (`%USERPROFILE%\.miopen\db`).
   - La Find-Db efectiva es la que diga la línea `VBPERF stage=env` (`miopen_user_db_path`); si aún no existe esa línea, usa la regla anterior. Cuenta las líneas de todos los `*.ufdb.txt` de esa carpeta justo cuando el servidor está listo (tras el calentamiento) y al terminar cada pasada.
6. **Escenarios** (después de las pasadas):
   - `contention`: con el servidor de la última pasada, lanza a la vez 2 generaciones de la frase más larga y un `POST /transcribe` del audio de referencia del perfil. Registra `transcribe_wait_s` (de la petición a la respuesta) y `generation_s_max`.
   - `batch`: encola a la vez `--concurrent N` frases distintas del mismo perfil, mide el tiempo total hasta que todas terminan y compáralo con la suma de sus `wall_s` individuales de la pasada warm: `throughput_gain = suma_individual / total_en_lote`.

### `<out>/bench.json`

```json
{
  "schema": "vb-bench/1",
  "tag": "1.1-a1",
  "commit": "<git rev-parse HEAD>",
  "dirty": false,
  "code_path": "<directorio de backend/ que importó el servidor, según VBPERF env o, si no, el del worktree>",
  "started_at": "2026-10-07T10:00:00-05:00",
  "args": {"phrases": ["p01", "p02"], "passes": ["cold", "warm"], "miopen_db": "fresh", "env": {}, "scenario": null},
  "env_checks": {"…": "payload de la línea VBPERF stage=env, o null"},
  "server": {"cold": {"load_s": 41.2}, "warm": {"load_s": 12.0}},
  "passes": [
    {
      "name": "cold",
      "miopen_db_path": "…",
      "ufdb_lines_before": 0,
      "ufdb_lines_after": 37,
      "phrases": [
        {
          "id": "p01", "seed": 1101, "text_chars": 44,
          "wall_s": 151.3, "audio_s": 3.1, "steps": 39, "rtf": 48.8,
          "wav": "cold/p01.wav",
          "stages": {
            "reference": {"ms": 1830.0},
            "talker": {"ms": 9100.0, "steps": 39, "steps_per_s": 4.3},
            "codec": {"ms": 140000.0, "T": 369, "chunks": 2},
            "normalize": {"ms": 12.0}, "effects": {"ms": 0.0}, "save": {"ms": 30.0}, "total": {"ms": 151200.0}
          }
        }
      ]
    },
    {"name": "warm", "…": "…"}
  ],
  "scenarios": {
    "contention": {"transcribe_wait_s": 9.8, "generation_s_max": 16.1},
    "batch": {"n": 4, "total_s": 31.0, "sum_individual_s": 64.0, "throughput_gain": 2.06}
  },
  "summary": {
    "warm_mean_s": 16.4, "warm_median_s": 16.1, "cold_mean_s": 150.2,
    "rtf_warm_mean": 2.7, "talker_steps_per_s_mean": 4.6,
    "ufdb_lines_added_cold": 37, "ufdb_lines_added_warm": 0, "ufdb_lines_total": 37,
    "stage_ms_mean": {"reference": 1800.0, "talker": 13000.0, "codec": 1500.0},
    "load_s": 41.2, "phrases_n": 12
  }
}
```

- `steps = round(audio_s × 12.5)` si el servidor no los da; `rtf = wall_s / audio_s`.
- `stage_ms_mean` y `talker_steps_per_s_mean` se calculan sobre la pasada **warm** (si no la hay, sobre la última).
- `ufdb_lines_added_<pasada> = ufdb_lines_after − ufdb_lines_before` de esa pasada; `ufdb_lines_total` es el recuento al terminar.
- Campos sin dato: `null`, nunca omitidos. `stages` puede ir vacío hasta la tarea 0.4.

## §3 · `scripts/compare_wavs.py`

### CLI

```
<python> scripts/compare_wavs.py --config perf-loop/config.json
    --ref perf-loop/bench/<tagA> --cand perf-loop/bench/<tagB>
    [--pass warm] [--metrics snr,dur | all] --out <ruta.json>
<python> scripts/compare_wavs.py --self-test
```

Compara la intersección de frases presentes en `<ref>/<pass>/*.wav` y `<cand>/<pass>/*.wav`. `all` = `snr,spk,wer,dur`. Si `--ref` y `--cand` son el mismo directorio, todas las diferencias deben salir exactamente 0 (lo comprueba la puerta de la 0.3).

- **snr**: `10·log10(Σref² / Σ(ref−cand)²)` sobre la longitud mínima común; si la diferencia es 0, `999.0`.
- **spk**: coseno entre `extract_speaker_embedding` del modelo Qwen3-TTS de cada WAV y el del audio de referencia del perfil (de `config.bench.data_dir`). `spk_cos_delta = cand − ref`.
- **wer**: Whisper `small`, español, decodificación determinista, frente al texto pedido (`scripts/bench_phrases.json`). Normaliza ambos: minúsculas, sin puntuación ni tildes duplicadas, espacios colapsados. `wer_delta = cand − ref`.
- **dur**: `dur_dev = |dur_cand / dur_ref − 1|`.

### `--out` (JSON)

```json
{
  "schema": "vb-compare/1",
  "ref": "0.5-a1", "cand": "1.1-a1", "pass": "warm", "metrics": ["snr", "spk", "wer", "dur"],
  "phrases": [
    {"id": "p01", "snr_db": 999.0, "spk_cos_ref": 0.81, "spk_cos_cand": 0.81, "wer_ref": 0.0, "wer_cand": 0.0,
     "dur_ref_s": 3.1, "dur_cand_s": 3.1, "dur_dev": 0.0}
  ],
  "summary": {
    "n": 12, "snr_db_min": 999.0, "snr_db_mean": 999.0,
    "spk_cos_ref_mean": 0.81, "spk_cos_cand_mean": 0.81, "spk_cos_delta_mean": 0.0,
    "wer_ref_mean": 0.02, "wer_cand_mean": 0.02, "wer_delta": 0.0,
    "dur_dev_max": 0.0
  }
}
```

Métricas no pedidas: `null`. Salida 0 si pudo comparar al menos una frase.

## §4 · Líneas `VBPERF`

Solo con `VOICEBOX_PERF_LOG=1` (el banco la pone siempre, vía `config.bench.env`). Una línea por evento, por `logging` a stderr:

```
VBPERF {"gen_id": "…", "stage": "talker", "ms": 9100.0, "steps": 39, "steps_per_s": 4.29}
VBPERF {"stage": "env", "torch": "2.9.1+rocm7.2.1", "hip": "7.2.1", "cudnn_benchmark": false,
        "miopen_immediate": null, "attn_implementation": "sdpa", "codec_dtype": "torch.bfloat16",
        "miopen_env": {"MIOPEN_FIND_MODE": null, "…": "…"}, "miopen_user_db_path": "…", "code_path": "…"}
```

- El JSON va en una sola línea tras el prefijo exacto `VBPERF ` (lo que preceda en la línea, como la fecha del logger, se ignora).
- Etapas: `reference`, `talker` (`steps`, `steps_per_s`), `codec` (`T`, `chunks`), `normalize`, `effects`, `save`, `total`. `gen_id` es el id de la generación de la BD, para que el banco asocie cada etapa con su frase.
- `torch.cuda.synchronize()` antes de cada marca, solo si hay GPU.

## §5 · Independencia del evaluador

`bench_tts.py` y `compare_wavs.py` no importan nada de `backend/`: el banco habla con el servidor por HTTP y el comparador carga `qwen_tts`, `transformers` y `torch` directamente. Así el evaluador no cambia cuando cambia el código que evalúa. Si necesitan código compartido, va en `scripts/bench_lib/`, que se congela con ellos.

## §6 · Directorios

- `perf-loop/bench/<tag>/`: salida de cada verificación (no va a git). `status.json`, `gate.json` y `logs/` los escribe `loopctl`; `bench.json`, `compare-*.json` y los WAV, los scripts.
- `perf-loop/bench/_data/`: data dir del banco (por defecto).
- `perf-loop/bench/_dev/`, `_profile/`: pruebas manuales del agente.
