# Progreso del loop

Registro cronológico que escribe loopctl (no editar a mano).

- 2026-10-06T22:23:05-05:00 · Loop inicializado desde release/v0.6.0@901ffcc93b en E:\VoiceBox\voicebox-perf
- 2026-10-06T22:23:59-05:00 · 0.0 empieza en perf/qwen-task/0.0 (base 674497f6)
- 2026-10-06T22:24:04-05:00 · 0.0: verificación 0.0-a1 lanzada
- 2026-10-06T22:24:24-05:00 · 0.0 cerrada: pass
- 2026-10-06T22:24:58-05:00 · nota 0.0: Linea base de tests vacia: pytest se interrumpio por un ImportError al colectar test_profile_duplicate_names.py (relative import beyond top-level package). passed=0, asi que la puerta de ningun test nuevo fallando queda vacia mientras persista. Falta continue-on-collection-errors o ignorar ese modulo en el comando de tests del config. Decide el usuario.
- 2026-10-06T22:25:17-05:00 · 0.1 empieza en perf/qwen-task/0.1 (base 4bf64ba2)
- 2026-10-06T22:27:31-05:00 · nota 0.1: review: approve — solo añade scripts/bench_phrases.json; cumple contracts §1 (es, 12 ids pNN, seeds 1101-1112 distintos, target_s 3.1-11.9, longitudes distintas, p01/p02 las más cortas = smoke)
- 2026-10-06T22:27:32-05:00 · 0.1: verificación 0.1-a1 lanzada
- 2026-10-06T22:27:36-05:00 · 0.1 cerrada: pass
- 2026-10-06T22:27:54-05:00 · 0.2 empieza en perf/qwen-task/0.2 (base ec2ce0ab)
- 2026-10-06T23:03:32-05:00 · nota 0.2: Prueba _dev (persist): p01 cold 103.5 s, warm (proceso nuevo, ufdb sin lineas nuevas) 65.8 s, y la misma p01 en el mismo proceso 11-12 s. La pasada warm del contrato (un proceso nuevo por pasada) paga ~50 s por longitud la primera vez en cada proceso, asi que warm_mean_s no es la 'longitud ya vista' del plan (16,5 s) y batch.throughput_gain sale inflado (5.37 con 2 frases) porque compara warm de proceso nuevo con un lote ya caliente. Conviene decidirlo antes de congelar el arnes en 0.5.
- 2026-10-06T23:15:22-05:00 · nota 0.2: review: approve — solo añade scripts/bench_tts.py y su test; cumple contracts §2. Menores: batch throughput_gain inflado (warm en proceso nuevo vs lote ya caliente; decidir antes de congelar en 0.5), ufdb delta entre carpetas distintas si cambia la Find-Db, sondeo a 0,2 s.
- 2026-10-06T23:15:22-05:00 · 0.2: verificación 0.2-a1 lanzada
- 2026-10-06T23:19:11-05:00 · 0.2 cerrada: pass
