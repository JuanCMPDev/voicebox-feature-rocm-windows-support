# Plan de mejora: velocidad Qwen3-TTS local

Oct 6, 2026 · @Juan Carlos Muñoz

Objetivo: bajar una frase de 6 s con el motor `qwen` (1.7B, voz clonada, RX 7800 XT) de 137–162 s a unos 8–10 s en cuatro fases, sin cambiar el modelo, sus entradas ni la calidad del audio. Parte del informe [Qwen3-TTS lento en ROCm/Windows](https://claude.ai/code/artifact/654c7edf-0848-4016-b495-300acc5eabfe); las cifras de meta son estimaciones que la Fase 0 debe confirmar.

**Qué significa «sin afectar la calidad» en este plan.** Solo entran cambios de tres tipos: (1) los que no tocan la aritmética ni las entradas del modelo (modo de búsqueda de MIOpen, cachés, colas, calentamientos): la onda debe salir idéntica salvo redondeo (SNR > 50 dB frente a la línea base con el mismo `seed`); (2) los que cambian solo cómo se ejecuta la misma aritmética (otro kernel GEMM, grafos, lotes): los logits varían en el último decimal y el muestreo puede divergir, así que se validan con métricas objetivas y escucha ciega, no por identidad de la onda; (3) los que reducen trabajo del códec dando a la parte generada un contexto igual o mayor que el que recibe hoy. Quedan fuera el modelo 0.6B, recortar la referencia, cuantizar el talker o tocar el muestreo (ver la última sección).

| Escenario (frase de 6 s, 75 pasos) | Tiempo | RTF | Qué lo consigue |
| --- | --- | --- | --- |
| Hoy, longitud nueva | 137–162 s | 21–26 | — |
| Hoy, longitud ya vista | 16,5 s | 2,7 | — |
| Tras Fase 1 (MIOpen) | \~15–16 s, cualquier longitud | \~2,5–2,7 | Sin Find por frase nueva |
| Tras Fase 2 (menos trabajo) | \~12–14 s | \~2,0–2,3 | Prompt cacheado, códec a un tercio, sin contienda con Whisper |
| Tras Fase 3 (talker) | \~7–10 s | \~1,2–1,7 | Menos sobrecarga por paso, GEMM afinados, lotes |
| Fase 4, vía Linux/WSL2 + vLLM (opcional) | \~2–3 s | \~0,3–0,5 | Otro stack de inferencia; RTF 0,31 publicado por Qwen |

## Dónde se va el tiempo hoy

Dos costes dominan: la búsqueda de kernels de MIOpen (120–145 s cuando la longitud es nueva) y el talker, que corre a 4,6 pasos/s cuando la GPU podría dar diez veces más. El resto son segundos.

| Etapa | Hoy (estimado) | Techo en esta GPU | Por qué |
| --- | --- | --- | --- |
| Find de MIOpen (solo longitud nueva) | 120–145 s | 0 s | \~30 convoluciones del decodificador × 2 anchuras nuevas; el 99 % del tiempo medido lo gasta midiendo el kernel naive |
| Recodificar la referencia (26 s de audio) | 1–3 s | 0 s | La caché nunca acierta (`pytorch_backend.py:164-171`) |
| Talker, 75 pasos | \~13–15 s (217 ms/paso) | \~0,5–1 s | 1.7B en BF16 = 3,4 GB de pesos; a 624 GB/s un paso cuesta \~5,5 ms de memoria. Cada paso lanza además un `code_predictor.generate` de 15 tokens (5 capas) con toda la sobrecarga de HF generate: \~100 forwards de capa y más de mil lanzamientos de kernel por paso. El coste es de CPU/Python y latencia de lanzamiento en Windows, no de cómputo |
| Decodificación del códec (ref + salida, \~400 pasos) | \~1–2 s | \~0,3 s | Decodifica 32 s de audio para quedarse con 6; el primer trozo de 300 pasos se tira entero |
| Normalización, efectos, WAV | < 0,5 s | — | CPU, despreciable |

El reparto exacto sale de la instrumentación del informe (sección «Plan de instrumentación»); las cifras del talker y del códec son estimaciones a partir de 16,5 s totales en el caso rápido. Datos del modelo que fijan el techo: talker 28 capas, hidden 2048, GQA 16/8, FFN 6144, 16 grupos de códigos; code predictor 5 capas, hidden 1024 (config público del repo `Qwen/Qwen3-TTS-12Hz-1.7B-Base`).

## Fase 0 – Línea base y arnés de calidad (medio día)

Sin línea base no se puede afirmar que la calidad no cambia; esta fase crea el conjunto de prueba y los números contra los que se juzga cada fase.

1. **Conjunto fijo**: 12 frases en español de 3 a 12 s (longitudes distintas a propósito), el mismo perfil clonado y `seed` fijo por frase. Guárdalas en un JSON junto al script.
2. **Script de banco** (`scripts/bench_tts.py`, fuera de la app): lanza las frases por `POST /generate`, espera el estado `completed`, copia los WAV a `bench/<etiqueta>/` y anota tiempo total y pasos (`duración × 12,5`). Primera pasada con el servidor recién arrancado, segunda inmediata: la diferencia entre ambas es el coste del Find.
3. **Instrumentación por etapa**: la del informe (referencia, talker en pasos/s, códec, normalización, guardado), con `torch.cuda.synchronize()` antes de cada marca y una línea de log por etapa. Envíala al log de Tauri (`server-log`) para no tocar la BD.
4. **Métricas de calidad** (`scripts/compare_wavs.py`): por frase, frente a la línea base: SNR de la onda (para cambios del tipo 1 y 3; aceptación > 50 dB); similitud de locutor con el propio `extract_speaker_embedding` del modelo frente al audio de referencia (coseno; aceptación: media no inferior a la base en más de 0,01); WER con Whisper `small` sobre el texto pedido (aceptación: no empeora); duración (±5 %). Para cambios del tipo 2, además escucha ciega A/B de 6 pares.
5. **Registro**: una tabla por fase con tiempo medio, RTF, pasos/s del talker, líneas nuevas en `ufdb.txt` y las cuatro métricas. Las secciones siguientes citan sus puertas de aceptación contra esta tabla.

Comprobaciones rápidas que conviene anotar en la línea base: `torch.backends.cudnn.benchmark`, `torch.backends.miopen.immediate`, `model.config._attn_implementation`, dtype del códec y número de líneas de la ufdb antes de empezar.

## Fase 1 – Eliminar el coste de MIOpen (1 día)

Ganancia esperada: de 137–162 s a \~15–16 s en toda frase nueva. Cambio de tipo 1: la onda debe salir idéntica. Dos variantes; la B es la recomendada porque conserva los kernels medidos, y la A queda como alternativa si el banco la iguala.

**Variante B (recomendada): modo híbrido sin el solver naive.** En `backend/app.py`, junto a `MIOPEN_LOG_LEVEL` (línea 102) y antes de `import torch`:

```python
if os.environ.get("VOICEBOX_BACKEND_VARIANT") == "rocm":
    os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD", "0")
    os.environ.setdefault("MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_BWD", "0")
```

MIOpen sigue buscando por forma nueva, pero solo entre los solvers GEMM (2,1 s de kernel en toda la ufdb frente a 483 s del naive): el Find por frase nueva pasa de minutos a 1–3 s y la Find-Db sigue acumulando el mejor kernel medido. `setdefault` respeta un valor del usuario. `VOICEBOX_BACKEND_VARIANT` lo fija `server.py:55` antes de importar el backend; en modo desarrollo (`python -m backend.main`) no existe, así que añade como segunda condición `torch.version.hip` tras importar torch (MIOpen lee estas variables en la primera convolución, no al importar).

**Variante A: sin búsqueda.** Mismo sitio, `os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")` (o `"2"` si la versión no acepta el nombre). Con una forma ausente de la Find-Db, MIOpen elige el solver por heurística sin medir y no escribe la ufdb. Riesgo: solo de rendimiento (una elección mala para alguna forma); el banco de la Fase 0 lo detecta.

**Find-Db persistente y propia de la app.** En `backend/server.py`, antes de `from backend.main import app` (línea 93), leer `--data-dir` de `sys.argv` y fijar `MIOPEN_USER_DB_PATH={data_dir}/cache/miopen` y `MIOPEN_CUSTOM_CACHE_DIR={data_dir}/cache/miopen` (crear la carpeta). Beneficios: la ufdb sobrevive a reinstalaciones y a cambios de usuario de Windows, se puede borrar desde la app y, más adelante, se puede distribuir una ufdb precalentada para gfx1101 con el binario ROCm (`services/rocm.py`). Sin esto la base vive en `%USERPROFILE%\.miopen\db\`.

**Calentamiento al cargar el modelo.** Al final de `_load_model_sync` (`pytorch_backend.py:123-125`), decodificar códigos ficticios para las longitudes que la Fase 2 va a agrupar (300 y de 32 a 320 de 32 en 32) y ejecutar un paso del talker con un texto corto. Mueve a la carga el primer Find de cada forma y la carga inicial de kernels rocBLAS; con la Find-Db persistente solo cuesta la primera vez (1–3 min con la variante B). Mostrarlo en el estado `loading_model`.

```python
tok = self.model.model.speech_tokenizer
nq = tok.model.config.num_quantizers  # 16
for steps in (300, *range(32, 321, 32)):
    codes = torch.zeros(steps, nq, dtype=torch.long, device=self.device)
    tok.decode([{"audio_codes": codes}])
```

**Puerta de aceptación**: tiempo de frase nueva = tiempo de frase vista ±10 %; con A la ufdb no crece, con B crece como mucho una entrada por capa y anchura nueva y el Find suma < 3 s por frase; SNR > 50 dB frente a la línea base en las 12 frases.

## Fase 2 – Hacer menos trabajo por frase (2–3 días)

Ganancia esperada: de \~16 s a \~12–14 s, y tiempos estables aunque haya transcripciones en paralelo. Todo es de tipo 1 o 3: la parte generada recibe las mismas entradas o más contexto que hoy.

**2.1 Arreglar la caché del prompt de voz** (`pytorch_backend.py:157-193`, `utils/cache.py:64-72`). Convertir la lista que devuelve `create_voice_clone_prompt` a dict con `self.model._prompt_items_to_voice_clone_prompt(items)` antes de cachear (y guardar también `ref_text`, que `generate_voice_clone` no necesita cuando recibe el dict); así el `isinstance(dict)` existente acierta y `torch.load(weights_only=True)` funciona porque solo hay tensores, listas y bools. Mover los tensores a CPU al guardar y a `self.device` al leer. Ahorro: 1–3 s por frase y el Find del encoder tras cada reinicio. Puerta: etapa «referencia» < 50 ms con `cache=True`; SNR > 50 dB.

**2.2 Decodificar solo contexto + salida.** Hoy el códec decodifica `ref_code` (330 pasos) + salida y tira la referencia (`qwen3_tts_model.py:612-631`). Como la decodificación va por trozos de 300 pasos, la parte generada ya se decodifica hoy con solo 25–55 pasos de contexto real (`chunked_decode`, `tokenizer_v2:886-895`). Sustituir en `_generate_sync` la llamada a `generate_voice_clone` por sus cuatro pasos (tokenizar, `self.model.model.generate`, decodificar, cortar) y decodificar `cat(ref_code[-C:], codes)` con `C = min(pasos_ref, 128)`, cortando exactamente `C × 1920` muestras. Con C = 128 (10 s) la salida recibe más contexto que hoy para cualquier referencia de 24 s o más y cubre la ventana de 72 pasos del pre\_transformer; para referencias cortas el contexto es el mismo que hoy (toda la referencia). El códec pasa de \~400 a \~200 pasos por frase y las formas posibles bajan. Puerta: SNR > 40 dB frente a la base (misma aritmética, distinto contexto de arranque) y similitud de locutor igual o mayor.

**2.3 Agrupar longitudes.** Rellenar la secuencia que entra al códec hasta el múltiplo de 32 pasos con filas de `-1`: `decode` ya recorta la salida a la longitud real (`tokenizer_v2:1012-1017`) y todo el decodificador es causal, así que las muestras anteriores no cambian. Con 2.2, T = C + pasos\_gen queda entre \~140 y \~300: una decena de formas por capa en total, todas precalentadas por la Fase 1. Resultado: ninguna frase vuelve a disparar un Find, con la variante A o la B. Puerta: ufdb sin líneas nuevas tras el calentamiento; SNR > 50 dB frente a 2.2.

**2.4 Una sola cola para la GPU.** Pasar `/transcribe` (`routes/transcription.py:79`) por la misma cola serie de `task_queue.py` o por un `asyncio.Lock` compartido con `_generate_sync` y `_create_prompt_sync`. Una transcripción ya no roba tiempo de GPU a la generación ni se queda 45 s esperando a medias. De paso, crear la fila con `status="queued"` en `routes/generations.py:101` y dejar que `generation.py:73` la pase a `generating`; la UI distingue cola de ejecución.

**2.5 Menor.** `/health`: `scan_cache_dir()` a `asyncio.to_thread` o cacheado 60 s (`routes/health.py:146-149`), para que el sondeo de 30 s no frene el loop mientras se genera.

## Fase 3 – Acelerar el talker (1–2 semanas, por experimentos)

Ganancia esperada: de \~13 s a \~7–10 s. El talker está limitado por sobrecarga de CPU y lanzamiento de kernels, no por cómputo, así que las palancas son reducir llamadas, no acelerar FLOPs. Son cambios de tipo 2 salvo 3.2 si se replica el muestreo de HF; cada uno se mide por separado contra la tabla de la Fase 0 y se descarta si no da al menos un 5 %.

| # | Cambio | Dónde | Ganancia estimada | Riesgo y validación |
| --- | --- | --- | --- | --- |
| 3.1 | Perfilar un paso con `torch.profiler` (CPU y GPU) para confirmar el reparto sobrecarga/cómputo antes de tocar nada | `_generate_sync`, una frase | — | Ninguno; decide el orden del resto |
| 3.2 | Sustituir el `code_predictor.generate()` de HF que corre en cada paso (`modeling_qwen3_tts.py:1671-1680`) por un bucle propio de 15 forwards con `DynamicCache` y el mismo muestreo (temperatura → top-k → top-p, `torch.multinomial`) | Parche sobre `talker.code_predictor.generate` desde `pytorch_backend.py` | 10–25 % del tiempo por paso | Si se respeta el orden de los warpers de HF la salida es idéntica (SNR > 50 dB); si no, métricas y escucha |
| 3.3 | Prioridad del proceso del servidor en «por encima de lo normal» (`SetPriorityClass` vía `ctypes` al arrancar, o desde Tauri) y `torch.inference_mode()` en `_generate_sync` | `server.py`, `pytorch_backend.py:226` | 0–5 %, menos variabilidad | Ninguno |
| 3.4 | GEMM afinados con TunableOp: una pasada de afinado offline con el banco (`PYTORCH_TUNABLEOP_ENABLED=1`, `..._TUNING=1`, `..._FILENAME={data_dir}/cache/tunableop.csv`, `..._HIPBLASLT_ENABLED=1`) y después solo lectura (`..._TUNING=0`): las formas sin afinar usan el kernel por defecto, sin coste en caliente | Variables en `app.py`; afinado desde `scripts/bench_tts.py` | 5–15 % de la parte GEMM; modesto porque el cuello es la sobrecarga | Cambia el kernel, no la aritmética: métricas objetivas + escucha. Comprobar que hipBLASLt soporta gfx1101 en ROCm 7.2.1 (si no, desactivar ese flag) |
| 3.5 | Grafos HIP para el paso de decodificación: KV cache estática (`cache_implementation="static"`) y captura del forward del talker y del bucle del code predictor con `torch.cuda.CUDAGraph`; la generación repite el grafo con nuevas entradas | Capa propia de decodificación en `pytorch_backend.py`; no requiere `torch.compile` (sin Triton en Windows) | 2–4× en el talker si la captura funciona | Soporte de grafos HIP en torch ROCm Windows sin confirmar: probar primero con un matmul de 10 líneas; si falla, abandonar. Tipo 2: métricas + escucha |
| 3.6 | Modo lote opcional: cuando hay ≥ 2 trabajos en cola con el mismo perfil, motor, idioma y tamaño, generarlos en una sola llamada (`generate_voice_clone` acepta listas) | `task_queue.py` + `generation.py` | Rendimiento casi ×N para N frases en cola, porque la sobrecarga por paso se reparte | La salida de un `seed` dado deja de coincidir con la de la ejecución individual (el RNG se consume distinto). Opt-in; mantener individual por defecto |
| 3.7 | Comprobar `model.config._attn_implementation`: con `aotriton_v2.dll` presente en el venv, SDPA debería estar activo; forzar `attn_implementation="sdpa"` en `from_pretrained` si sale `eager` | `pytorch_backend.py:116-121` | 0–5 % (solo afecta al prefill de \~450 tokens) | Ninguno si SDPA ya estaba activo |

Orden recomendado: 3.1 → 3.2 → 3.3 → 3.7 → 3.4 → 3.6 → 3.5. El techo práctico de esta fase con el stack de HF en Windows está en 8–12 pasos/s sin grafos y 20–40 con ellos; el techo físico (ancho de banda) sería \~150.

## Fase 4 – Entorno del PC y vía alternativa

Ajustes fuera del código que reducen latencia de lanzamiento y variabilidad, y una vía distinta si el objetivo es acercarse al RTF 0,31 publicado.

**Entorno Windows (una tarde, sin riesgo para la calidad)**

- Activar la programación de GPU acelerada por hardware (Configuración → Pantalla → Gráficos) y el plan de energía de alto rendimiento; en Adrenalin, perfil de ajuste «Rendimiento» o reloj mínimo elevado para evitar que la GPU baje de frecuencia entre lanzamientos cortos.
- Mantener el driver Adrenalin y el HIP SDK en la versión que exige `torch 2.9.1+rocm7.2.1`; revisar en cada actualización de ROCm Windows si MIOpen ya trae bases de datos (`*.kdb`, `*.fdb.txt`) para gfx1101: si llegan, la Fase 1 se vuelve innecesaria.
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` para reducir fragmentación con los workspaces de MIOpen; medir, puede ser neutro.
- No ejecutar en la misma GPU otras cargas (navegador con aceleración, grabación) durante los bancos; en uso normal, el modo cola de la Fase 2 ya evita la contienda interna.

**Vía alternativa: Linux (WSL2 o Docker) con vLLM.** El RTF 0,31 de Qwen se mide con vLLM, que no existe para Windows/ROCm. ROCm en Linux soporta RDNA3 de forma oficial (en WSL2 la lista de GPU depende de la versión: comprobar que gfx1101 esté en la matriz), trae bases de datos de MIOpen, Triton y `torch.compile`, y el repo ya incluye `docker-compose.rocm.yml` y `Dockerfile`. Un servidor `voicebox-server` en WSL2 con el mismo `--data-dir` y el escritorio Tauri apuntando a él (`http://127.0.0.1:17493`) mantendría la app intacta. Coste: instalar y mantener ROCm en WSL2, y validar que vLLM soporta gfx1101 en la versión elegida (hoy es soporte parcial). Es un cambio de stack, no de modelo: misma red, mismos pesos y mismo muestreo, así que la calidad se valida con las métricas de la Fase 0 como un cambio de tipo 2. Solo merece la pena si las Fases 1–3 se quedan cortas.

## Orden de ejecución y puertas de calidad

Cada fase se cierra con el banco de la Fase 0 y entra en la siguiente solo si cumple su puerta; el orden maximiza ganancia temprana con riesgo mínimo.

| Paso | Esfuerzo | Frase de 6 s (meta) | Puerta para seguir |
| --- | --- | --- | --- |
| Fase 0: banco, instrumentación, métricas | 0,5 día | 137–162 s / 16,5 s (medido) | Tabla base completa; reparto por etapa conocido |
| Fase 1: MIOpen (variante B, Find-Db propia, calentamiento) | 1 día | \~15–16 s | Frase nueva = frase vista ±10 %; SNR > 50 dB |
| Fase 2.1 + 2.4: caché del prompt, cola única, estado queued | 1 día | \~13–14 s | Etapa referencia < 50 ms; `/transcribe` nunca espera más que una generación |
| Fase 2.2 + 2.3: contexto acotado y longitudes agrupadas | 1–2 días | \~12–13 s | ufdb sin líneas nuevas; SNR > 40 dB y similitud de locutor ≥ base |
| Fase 3.1–3.3 + 3.7: perfil, code predictor propio, prioridad, SDPA | 2–3 días | \~10–11 s | Cada cambio ≥ 5 %; SNR > 50 dB o métricas + escucha |
| Fase 3.4 + 3.6: TunableOp, modo lote | 2 días | \~9–10 s (lote: ×N en cola) | Métricas objetivas ≥ base; escucha ciega sin preferencia por la base |
| Fase 3.5: grafos HIP | 3–5 días, condicional | \~7–8 s | Prueba de soporte en 1 h antes de invertir; métricas + escucha |
| Fase 4: entorno y, si hace falta, WSL2 + vLLM | 0,5 día / 1–2 semanas | \~2–3 s en la vía Linux | Igual que tipo 2 |

Entregables por fase: el diff en una rama por fase, la fila nueva de la tabla del banco y los WAV del conjunto fijo guardados con etiqueta, para poder volver atrás y comparar escuchas en cualquier momento.

## Qué no hacer

Estos cambios aceleran, pero cambian el modelo, sus entradas o el muestreo, y por tanto la calidad; quedan fuera del plan salvo decisión explícita.

- Pasar al modelo 0.6B: más rápido, pero otra red.
- Recortar o resamplear la muestra de referencia para acortar el prompt ICL: cambia la voz clonada.
- Cuantizar el talker (INT8, FP8) o pasarlo a FP16 sin validar: cambia los logits de forma sistemática, no solo por redondeo.
- Bajar `max_new_tokens`, `top_k`, `top_p` o la `temperature`, o desactivar `repetition_penalty`: cambia el muestreo.
- Decodificar la salida sin ningún contexto de referencia (C = 0) o con menos del que recibe hoy: altera los primeros cientos de ms de audio.
- `torch.backends.cudnn.benchmark = True`: en ROCm fuerza la búsqueda exhaustiva de MIOpen y empeora exactamente el problema de la H1.
- Bajar `max_chunk_chars` para «paralelizar»: cada trozo es una longitud nueva y una generación más; con el talker limitado por sobrecarga no hay ganancia.
