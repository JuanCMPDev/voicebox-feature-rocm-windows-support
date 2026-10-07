# Resultados del banco

Tabla generada por loopctl a partir de `results.jsonl` (no editar a mano).

Hot = misma frase en el mismo proceso (régimen estable); warm = proceso nuevo con la Find-Db ya poblada (tras un reinicio); cold = primera vez. Reinicio = warm − hot por frase. Speedup en hot (en warm si falta la pasada hot, marcado con *).

| Tarea | Etiqueta | Resultado | Hot medio (s) | Warm medio (s) | Cold medio (s) | Reinicio (s) | RTF hot | Talker pasos/s | ufdb + cold/warm | SNR mín. vs ant. (dB) | Δ sim. locutor | Δ WER | Desv. duración | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.0-a1 | pass | — | — | — | — | — | — | —/— | — | — | — | — | — |
| 0.1 | 0.1-a1 | pass | — | — | — | — | — | — | —/— | — | — | — | — | — |
| 0.2 | 0.2-a1 | pass | — | 45.10 | 44.91 | — | — | — | 0/0 | — | — | — | — | — |
| 0.0 | 0.0-a2 | pass | — | — | — | — | — | — | —/— | — | — | — | — | — |
| 0.2 | 0.2-a2 | pass | 11.18 | 44.85 | 45.06 | 33.67 | 3.90 | — | 0/0 | — | — | — | — | +0.6 %* |
| 0.3 | 0.3-a1 | pass | — | — | — | — | — | — | —/— | — | — | — | — | — |
| 0.4 | 0.4-a1 | pass | 11.20 | 44.69 | 44.95 | 33.48 | 3.91 | 7.53 | 0/0 | 999.0 | — | — | — | -0.2 % |
| 0.5 | 0.5-a1 | pass | 15.53 | 32.64 | 76.66 | 17.11 | 3.06 | 8.20 | 244/0 | — | — | — | — | -0.4 % |
| 1.3 | — | skip | — | — | — | — | — | — | —/— | — | — | — | — | — |
| 1.1 | 1.1-a2 | pass | 8.80 | 9.06 | 10.08 | 0.26 | 1.65 | 8.21 | 24/0 | 45.3 | 0.0000 | 0.000 | 0.000 | +43.3 % |
| 1.2 | 1.2-a1 | pass | 8.80 | 9.05 | 9.12 | 0.26 | 1.65 | 8.17 | 0/0 | 999.0 | 0.0000 | 0.000 | 0.000 | +0.1 % |
| 1.4 | 1.4-a1 | pass | 8.79 | 8.89 | 9.85 | 0.11 | 1.65 | 8.31 | 24/0 | 999.0 | 0.0000 | 0.000 | 0.000 | +0.1 % |
| 1.G | 1.G-a2 | pass | 8.96 | 9.12 | 9.85 | 0.17 | 1.68 | 8.11 | 24/0 | 999.0 | 0.0000 | 0.000 | 0.000 | -1.9 % |
| 2.2 | — | skip | — | — | — | — | — | — | —/— | — | — | — | — | — |
