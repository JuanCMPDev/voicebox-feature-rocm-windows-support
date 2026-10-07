# Qwen3-TTS lento en ROCm/Windows — investigación (solo lectura)

Fecha: 2026-10-06. Checkout: `E:\VoiceBox\voicebox`, commit `901ffcc` (2026-07-21), `v0.4.0-91-g901ffcc`, versión 0.6.0, árbol con cambios locales sin commit. venv: torch 2.9.1+rocm7.2.1, qwen-tts 0.1.1, transformers 4.57.3, rocm_sdk 7.2.1 (MIOpen.dll sin `.kdb`/`.fdb.txt` para gfx1101). Rutas `backend/...` relativas al repo; `qwen_tts/...` a `backend/venv/Lib/site-packages/`.

**Causa principal**: MIOpen ejecuta un Find por capa del decodificador del códec para cada longitud nueva (modo por defecto DYNAMIC_HYBRID); Voicebox no fija `MIOPEN_FIND_MODE` ni agrupa longitudes. Problemas independientes: el prompt de voz se recodifica en cada generación (bug de caché) y el códec corre en BF16.

## Veredictos

| Hipótesis | Veredicto | Evidencia |
| --- | --- | --- |
| 1. Find por forma nueva | Confirmada | Solo `MIOPEN_LOG_LEVEL=4` (`backend/app.py:102-103`); sin `cudnn.benchmark`; `chunked_decode` con anchura dependiente de la longitud (`tokenizer_v2:886-895`). |
| 2. Longitudes sin agrupar | Confirmada | Sin padding; `max_chunk_chars=800` solo trocea texto largo (`chunked_tts.py:73-74`); decodificador causal → agrupar es seguro. |
| 3. Códec en BF16 | Confirmada | `torch_dtype=torch.bfloat16` (`pytorch_backend.py:120`) → tokenizer (`modeling_qwen3_tts.py:1915-1920`). |
| 4. Referencia recodificada | Confirmada (bug) | Caché devuelve `list`; backend acepta solo `dict`/`Tensor` (`pytorch_backend.py:164-171`); `torch.load(weights_only=True)` falla y borra (`utils/cache.py:67-72`). |
| 5. Concurrencia | Confirmada en parte | Sin lock ni bloqueo del loop; misma GPU/stream; «generating» al encolar (`routes/generations.py:101`); no hay «queued». |
| 6. Watchdog/caída | Descartada (watchdog); caída sin determinar | Watchdog solo vigila el PID de Tauri (`server.py:197-232`); ruta failed en `services/generation.py:138-146`. |

## H1 – MIOpen
- Sin `cudnn.benchmark`/`deterministic` en `backend/` ni `qwen_tts/`. torch 2.9.1 tiene `torch.backends.miopen.immediate` y ambos caminos (Find e inmediato) en `torch_hip.dll`; en ambos, con DYNAMIC_HYBRID, una forma nueva lanza búsqueda; solo `MIOPEN_FIND_MODE=FAST` la evita.
- Variables: solo `MIOPEN_LOG_LEVEL=4` (`app.py:102-103`, antes de `import torch` en 105). `HSA_OVERRIDE_GFX_VERSION` se limpia y solo se fija para gfx<1100 (`app.py:41-100`). No existen `MIOPEN_FIND_MODE`, `MIOPEN_FIND_ENFORCE`, `MIOPEN_USER_DB_PATH`, `MIOPEN_DEBUG_*`, `PYTORCH_TUNABLEOP_*`, `HIP_VISIBLE_DEVICES`. `docker-compose.rocm.yml:33-36` (Docker) fija `HSA_OVERRIDE_GFX_VERSION` y `PYTORCH_HIP_ALLOC_CONF`.
- Arranque: `server.py:53-59` detecta variante por nombre del exe. Tauri lanza `{data_dir}/backends/rocm/voicebox-server-rocm.exe` con cwd en esa carpeta y solo `VOICEBOX_MODELS_DIR` (`tauri/src-tauri/src/main.rs:716-727`). Descarga: `services/rocm.py:236-395` vía `POST /backend/download-rocm` (`routes/rocm.py:24-41`). Hooks PyInstaller no tocan MIOpen.
- MIOpen.dll (`_rocm_sdk_libraries_custom/bin/`) reconoce `MIOPEN_FIND_MODE`, `MIOPEN_FIND_ENFORCE`, `MIOPEN_USER_DB_PATH`, `MIOPEN_SYSTEM_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD/BWD`, `MIOPEN_DEBUG_CONV_GEMM`. Sin `*.kdb`/`*.fdb.txt` en el venv.
- Mecanismo: `generate_voice_clone` concatena `ref_code` + códigos (`qwen_tts/inference/qwen3_tts_model.py:612-618`) y decodifica T = ref+gen (620). `chunked_decode(chunk_size=300, left_context_size=25)` (`tokenizer_v2:886-895`): trozo 1 = 300 pasos, trozo 2 = T−275. Cada paso cambia la anchura de ~30 convs en 6 etapas (2·2·8·5·4·3 = 1920). Ej.: 190.720 = 298 × 640 (etapa 8 kHz, 192 canales). Find mide el naive (483 s vs 2,1 s GEMM) → 120–145 s por longitud nueva.

## H2 – Longitudes
- `pytorch_backend.py:234-239` pasa el texto tal cual. Único relleno: conv causal (`tokenizer_v2:183-191`) y `pad_sequence(padding_value=-1)` (`qwen3_tts_tokenizer.py:328`) que `decode` recorta (`tokenizer_v2:1012-1017`) → el −1 ya permite agrupar.
- Encoder de referencia: longitud del WAV de la muestra (633.600 = 26,4 s @24 kHz), constante por perfil.
- `max_chunk_chars` (800, `models.py:93-95`) solo trocea >800 chars (`chunked_tts.py:73-74, 249-260`); `crossfade_ms` (50) solo al concatenar (`chunked_tts.py:172-201`).
- Decodificador causal: conv (159-193), transconv (195-208), ConvNeXt (211-240), pre_transformer causal/sliding (546-551).

## H3 – BF16
- `pytorch_backend.py:108-121`; dispositivo `"cuda"` (`base.py:101-102`). Kwargs al tokenizer (`modeling_qwen3_tts.py:1915-1920`); entrada encoder `.to(self.model.dtype)` (`qwen3_tts_tokenizer.py:248`); speaker encoder (`modeling_qwen3_tts.py:1951`). Sin autocast en Voicebox; RoPE/RMSNorm FP32 internos (`tokenizer_v2:273-279, 384-388`).

## H4 – Prompt de voz
- `services/profiles.py:581-591` → `create_voice_prompt(use_cache=True)`. `pytorch_backend.py:157-171` solo acepta `dict`/`Tensor`; `create_voice_clone_prompt` devuelve `List[VoiceClonePromptItem]` (`qwen3_tts_model.py:361, 447-458`) → recomputa (173-186): encode (418-425) + embedding (435-445).
- Disco: `utils/cache.py:67` `weights_only=True` falla con dataclasses → `unlink` (70-72) → se reescribe (`pytorch_backend.py:189-191`). Arreglo: aceptar `list` o convertir con `_prompt_items_to_voice_clone_prompt` (460-466); `weights_only=False` o `add_safe_globals`.

## H5 – Concurrencia
- Cola serie `task_queue.py:39-66`; `_queued_generation_ids` (26) solo para cancelar (105-117). Estado «generating» en `routes/generations.py:91-104` y `generation.py:73`.
- Whisper `routes/transcription.py:42,79` → `pytorch_backend.py:317-378` (`to_thread`, 378); generación `to_thread` (243). Único lock `_tts_backends_lock` (`backends/__init__.py:203`) protege instanciación. Ambos en `"cuda"` (Whisper `pytorch_backend.py:301`), stream por defecto → serialización en GPU durante el Find.

## H6 – Estabilidad
- Watchdog `server.py:114-235` (registro 278-283): PID padre cada 2 s; `os._exit(0)` (224-228). `/watchdog/disable` (`routes/health.py:47-53`). `/health` solo al arrancar en Tauri (`main.rs:309-333`, `App.tsx:202-224`); frontend sondea 30 s solo para mostrar (`useServer.ts:5-13`); reinicio manual (`GpuPage.tsx:276-281`).
- Rutas failed: (a) excepción `generation.py:138-146`; (b) cancelación 130-137; (c) `task_queue.py:57-62, 69-93`; (d) arranque `app.py:297-310` («Server was shut down during generation»). Consultar `generations.error`.
- Caída: sin determinar; sin log a archivo (solo `logs/watchdog.log`); stderr vía evento `server-log` (`main.rs:936-958`). Menor: `scan_cache_dir()` síncrono en `/health` (`routes/health.py:146-149`).

## Recorrido `POST /generate` (engine=qwen)
1. `routes/generations.py:57` → fila «generating» (91-104), activa (106-110), efectos (113-123), encola (126-144).
2. Worker `task_queue.py:39-66` → `services/generation.py:29`.
3. `PyTorchTTSBackend` (`backends/__init__.py:691-699`); `load_engine_model` (513-521) → `_load_model_sync` (`pytorch_backend.py:90-125`); tokenizer (`modeling_qwen3_tts.py:1891-1920`).
4. Prompt: `profiles.py:516` (581-591) → `pytorch_backend.py:138` → `utils/cache.py` → `qwen3_tts_model.py:356` (418-425, 435-445). `generation.py:73` «generating».
5. `chunked_tts.py:204` → single-shot (249-260).
6. `pytorch_backend.py:226-243`: `manual_seed` (`base.py:187-200`); `generate_voice_clone` (`qwen3_tts_model.py:470`): tokeniza (588-599); talker (603-610 → `modeling_qwen3_tts.py:2022`, 2272; `code_predictor.generate` por paso 1671-1680); cat (612-618); decode (620 → `qwen3_tts_tokenizer.py:361-363` → `tokenizer_v2:993-1017` → 886-895); corte (623-631); float32 CPU (`qwen3_tts_tokenizer.py:364`).
7. `generation.py:87`: `normalize_audio` (90-91); `_save_generate` (168-226): WAV (184-185), versión (189-196), efectos (200-224); completed (122-128); `complete_generation` (150).

## Instrumentación
`torch.cuda.synchronize()` + `perf_counter()` por etapa: referencia (`pytorch_backend.py:173-183`), talker (envolver `self.model.model.generate`; pasos/s), códec (envolver `speech_tokenizer.decode`; T, trozos, líneas nuevas en ufdb), normalización (`generation.py:90-91`), efectos/guardado (185, 213-215), total/RTF (59-128). Además `MIOPEN_ENABLE_LOGGING_CMD=1`, `MIOPEN_LOG_LEVEL=5`, contar líneas de la ufdb, registrar `cudnn.benchmark`, `miopen.immediate`, `MIOPEN_FIND_MODE`, dtype del códec.

## Cambios propuestos (impacto/riesgo)
1. `MIOPEN_FIND_MODE=FAST` por defecto en ROCm (`app.py:102-105`, `setdefault`) — alto / medio (validar que el fallback no elige naive).
2. `MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0`, `_BWD=0` — alto en modo híbrido / bajo (GEMM cubre todas las entradas).
3. Agrupar longitudes: envolver `chunked_decode` para rellenar a múltiplo de 32 pasos y recortar (`pytorch_backend.py` tras 121) — alto / bajo-medio (causal; A/B del WAV).
4. Calentamiento al cargar con longitudes agrupadas (final de `_load_model_sync`) — medio / bajo.
5. Arreglar caché del prompt (`pytorch_backend.py:164-171`, `utils/cache.py:67`) — medio / bajo.
6. `MIOPEN_USER_DB_PATH={data_dir}/cache/miopen` en `server.py:88-93` — bajo-medio / bajo.
7. Códec en FP32/FP16 (`speech_tokenizer.model.float()`) — por determinar / bajo.
8. `/transcribe` en la cola o lock de GPU; exponer `queued` — medio UX / bajo.
9. `scan_cache_dir()` a `to_thread` (`routes/health.py:146-149`) — bajo / bajo.

Validación: (1) misma frase+seed dos veces; (2) frase con pasos distintos → lenta + líneas nuevas; (3) solo FAST → rápida, ufdb no crece; (4) 2+3 → ufdb ≤ ~10 formas/capa; (5) caché → etapa referencia ~0 ms.
