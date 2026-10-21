# Preregistro v2

Fecha: 2026-10-19. Congelado con SHA-256 en `extension/protocol/FREEZE.sha256` antes de correr el modelo de decisión sobre el test.

## Relación con CACIC

El test final (27 grupos de familias y 50 000 benignos) ya se usó una vez en CACIC, en julio de 2026. Los autores conocen sus resultados agregados y varios análisis post-hoc, entre ellos la cascada Logit→ModernBERT (F1 0.9223 en el test). Las configuraciones nuevas se fijaron solo con `router_train` y `gate1`. El paper declara este reuso, y las afirmaciones sobre las configuraciones nuevas son más débiles que las de CACIC.

## Datos

| Papel | Conjunto | Familias |
|---|---|---|
| Entrenar el enrutador | `router_train` | 54 familias conocidas por los expertos |
| Decisiones de diseño | `gate1` | 25 familias no vistas |
| Evaluación (una corrida) | `final_test` | 18 familias nuevas y 9 variantes declaradas |

## Sistemas

1. **Los sistemas de CACIC**, reconstruidos exactamente (la diferencia máxima con los CSV de CACIC es 1e-16): expertos fijos, ensamble, cascada Logit→DomURLs_BERT (t = 0.80), enrutador (5 semillas), control aleatorio emparejado y oráculos.
2. **C1, enrutador con costo por falso positivo.** Es el mismo enrutador de CACIC; los benignos de entrenamiento en los que algún experto da falsa alarma pesan × (1 + fp_weight). Se eligió fp_weight = 2.0 en la grilla {0, 1, 2, 4, 8}, con la regla: menor FPR media en desarrollo entre los valores con F1 medio ≥ F1(fp_weight = 0) − 0.005. En desarrollo: F1 0.8658, FPR 0.0420, latencia 0.513 ms. El puntaje es el `p_dga` del experto elegido; los umbrales por semilla para FPR 0.001 y 0.01 están en `extension/config/frozen_v2.json`.
3. **C2, cascada por tipo con regla de diccionario.** Logit decide si su confianza es ≥ 0.80. Si no, el dominio va a ModernBERT cuando la cobertura de palabras de su etiqueta más larga es ≥ τ, y si no, a DomURLs_BERT. τ = 0.40, elegido en `router_train` por la mayor exactitud balanceada WL contra NW (grilla 0.20 a 0.95). Lista de palabras: `extension/data/american-english.txt`, SHA-256 `9f513f1ceadb6a01c5485b7dbdfd5118dc66cd70b59cae2851292112d4066a32`. En desarrollo: F1 0.8682, FPR 0.0340, latencia 0.760 ms.
4. **Modelo de decisión.** d1-3B (`LiquidAI/d1-3B`, revisión `051bcc464b01b9f92942b364d9586b0ef5912432`), en float16 en la T4 (la T4 no tiene bf16). Se eligió entre Julia-1, Laya y d1-3B por el mayor AUC WL contra NW en 400 dominios de desarrollo (0.962). Las dos preguntas tipadas tienen dos opciones, con la opción aleatoria o benigna primero; ese orden también se eligió con esa muestra. Corre solo sobre los dominios dudosos (confianza de Logit < 0.80), una llamada por dominio.
5. **C3, cascada por tipo con modelo de decisión.** Igual que C2, pero d1-3B responde "¿es WL?": el dominio va a ModernBERT si la probabilidad de WL es ≥ τ_dm = 0.35. τ_dm se eligió por la mayor exactitud balanceada WL contra NW en una muestra de 4000 WL y 4000 NW de `router_train` (grilla 0.05 a 0.95). En desarrollo: F1 0.8703, FPR 0.0353, latencia 21.096 ms.
6. **C4, juez de dudosos.** Logit decide los dominios con confianza ≥ 0.80; en los dudosos decide d1-3B, que marca DGA si su probabilidad de DGA es ≥ θ = 0.10. θ se eligió por el mayor F1 en desarrollo (grilla 0.05 a 0.95). En desarrollo: F1 0.8633, FPR 0.1000, latencia 20.407 ms.
7. **Controles.** Cada sistema que elige expertos tiene su control aleatorio emparejado, con las proporciones de ruta medidas en el conjunto evaluado, como en CACIC, y 5 semillas.

## Métricas

- F1, FPR y latencia media por dominio, en global, en WL, en NW y por nivel (familia nueva o variante).
- Recall a FPR 0.001 y 0.01 con umbrales elegidos en desarrollo, junto con la FPR que se logra en el test.
- Precisión ajustada a 1:100.
- Intervalos de confianza del 95% con bootstrap pareado en dos etapas (se remuestrean las familias DGA y los dominios benignos), con 1000 réplicas y semilla 7, para estas diferencias de F1: C1 contra el enrutador de CACIC (voto mayoritario de las 5 semillas de cada uno), C2 contra la cascada de CACIC, C3 contra C2, C4 contra la cascada de CACIC, y C2 y C3 contra cada una de las 5 semillas de su control aleatorio.
- La latencia de C3 y C4 suma, en los dominios dudosos, la de Logit, la del modelo de decisión y la del experto elegido.

## Reglas de decisión

- **C1:** se afirma una mejora si su FPR media en el test es menor que la del enrutador de CACIC (0.0455) y su F1 medio es ≥ 0.9175 − 0.005.
- **C2:** se afirma una mejora si (a) su F1 supera al de su control aleatorio y el intervalo de la diferencia queda por encima de 0, (b) su F1 es ≥ al de la cascada de CACIC (0.8703) y (c) su FPR es ≤ a la de la cascada de CACIC (0.0320).
- **C3:** se afirma una mejora sobre su control si los 5 intervalos contra las semillas del control quedan por encima de 0, y una mejora sobre la regla de diccionario si el intervalo de C3 menos C2 queda por encima de 0.
- **C4:** se afirma una mejora si el intervalo de C4 menos la cascada de CACIC queda por encima de 0 y su FPR es ≤ 0.0320.
- En C2, la parte (a) exige que los 5 intervalos contra las semillas de su control queden por encima de 0.
- Si no se cumple ninguna regla, el paper reporta la frontera sin afirmar ninguna mejora.

## Ejecución

El script `extension/scripts/run_test_v2.py` se probó con pruebas unitarias y con un ensayo completo sobre desarrollo (`--dry-run`). Antes de correr verifica los SHA-256 de `FREEZE.sha256`, corre una sola vez y se niega a correr de nuevo si ya existe su resumen.
