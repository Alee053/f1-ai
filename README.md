# Agente de conducción en carreras 2D

Avance del proyecto de Inteligencia Artificial SIS-252, UCB Cochabamba, gestión II-2026.
El objetivo principal es entrenar y evaluar una política neuronal PPO en un entorno
existente. Gymnasium CarRacing aporta el circuito y la física Box2D. No se modela un
vehículo real de Fórmula 1.

La primera etapa incluye observaciones numéricas, un controlador de referencia,
separación de circuitos por semillas, entrenamiento reproducible y evaluación.
A* y el desgaste de neumáticos quedan para etapas posteriores.

Se utiliza Python 3.12 y `uv`. Las dependencias se instalan en `.venv`; PyTorch usa
la distribución CPU porque la política MLP es pequeña. No se requieren servicios
pagados, CUDA ni instalaciones globales.

Los informes y documentos permanecen en `docs/` y no se versionan. Los experimentos
y modelos se guardan en `runs/`, también excluido de Git.

## Instalación y comprobaciones

```bash
uv sync --locked
uv run f1-ai check
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Las pruebas incluyen reproducción por semilla, compatibilidad Gymnasium y SB3,
igualdad de la dinámica con CarRacing original, clasificación de finales de
episodio, una vuelta del controlador y actualización y restauración de PPO.

## Datos y entorno

El proyecto genera transiciones en simulación; no usa un dataset de imágenes.
Las semillas de `configs/ppo.toml` separan 20 circuitos de entrenamiento, 5 de
validación y 5 de prueba. La partición de prueba se reserva para la evaluación
final. Consultar un circuito de validación no lo convierte en una prueba final.

La adaptación numérica usa CarRacing de Gymnasium 1.2.3, con Box2D a 50 Hz.
Sobrescribe la salida interna `state_pixels`, que en el entorno original se
renderiza incluso sin ventana. No modifica `car_dynamics.py`. Una prueba compara
posiciones y recompensas nativas de ambos entornos para las mismas acciones.
La simulación visual sigue disponible al construir el entorno con
`render_mode="rgb_array"` o `render_mode="human"`.

La observación contiene 16 valores `float32` con escalas fijas y límites
`[-1, 1]`. No se ajusta un normalizador con datos de validación o prueba.

| Componentes | Preparación |
| --- | --- |
| Velocidad longitudinal y lateral | División por 60 y 30 |
| Velocidad angular | División por 5 |
| Orientación respecto al eje del circuito | Seno y coseno |
| Distancia lateral al eje | División por la semianchura de la pista |
| Ruedas en contacto con carretera | Fracción entre 0 y 1 |
| Dirección actual de las ruedas delanteras | División por 0,4 radianes |
| Cuatro puntos futuros a 8, 16, 32 y 48 unidades de recorrido | Coordenadas relativas al coche divididas por 50 |

Estos valores provienen del mapa y del estado interno. Son observaciones
instrumentadas, con información privilegiada de simulación. No equivalen a
percepción visual, LiDAR real ni localización con incertidumbre. Las distancias
pertenecen a las unidades del simulador.

La acción tiene dos valores continuos en `[-1, 1]`: dirección y aceleración
firmada. La aceleración positiva se convierte en acelerador y la negativa en
freno. Así se conservan los tres actuadores originales sin aplicar freno y
acelerador simultáneamente.

La recompensa propia es `1000 * avance_nuevo / longitud - 0.1` por paso.
El avance nuevo mide la máxima distancia neta alcanzada en el sentido del
circuito. Retroceder y volver a avanzar sobre el mismo tramo no vuelve a
premiarse. Se suman 100 puntos por completar una vuelta y se descuentan
50 por abandono sostenido de la pista o 100 por salir del área global.
La recompensa nativa de CarRacing se registra por separado; su umbral de
900 puntos no se aplica a esta tarea adaptada.

Una vuelta completada exige cruzar la línea de partida hacia adelante,
avanzar al menos el 95 % de la longitud y visitar al menos el 95 % de los
tramos de carretera. Un salto de más de 10 unidades entre pasos no cuenta como
avance. La tarea admite 3000 pasos, equivalentes a 60 segundos simulados,
frente a los 1000 pasos del registro original. Un contacto con carretera de
menos de dos ruedas durante 75 pasos consecutivos termina en fallo.
Este criterio del simulador no representa una vuelta homologada de F1.

## Referencias y evaluación

El agente aleatorio usa una semilla fija por circuito. El controlador de
referencia usa pure pursuit y control proporcional de velocidad a partir de
la misma observación que recibe PPO. Sus parámetros se comprueban con los
circuitos de entrenamiento. No aprende ni planifica una trayectoria óptima.

```bash
uv run f1-ai benchmark --steps 5000 --output runs/benchmark
uv run f1-ai evaluate --split validation --output runs/baselines
uv run f1-ai plot --evaluation runs/baselines
```

Se registra un episodio por agente y circuito, con retorno, cobertura, avance,
finalización, causa del final, pasos fuera de pista, tiempo simulado y tiempo
real. El tiempo de vuelta se calcula solo para éxitos. Si no existen vueltas
completadas, el resultado es `null`. La salida incluye CSV por paso, geometría
del circuito y JSON con versiones, configuración y revisión de Git. Las carpetas
de experimentos nuevas no sobrescriben las anteriores.

## PPO

La red contiene dos capas de 64 unidades para el actor y otras dos para el
crítico. Los hiperparámetros y la semilla se definen en `configs/ppo.toml`.
Se usa CPU y un hilo de PyTorch. Cuatro entornos vectorizados ejecutan sus pasos
secuencialmente mediante `DummyVecEnv`.

Para comprobar la integración sin realizar un entrenamiento prolongado:

```bash
uv run f1-ai train --smoke --output runs/ppo-smoke
uv run f1-ai evaluate --agents random pursuit ppo --split validation \
  --model runs/ppo-smoke/model.zip --output runs/validation-smoke
uv run f1-ai plot --evaluation runs/validation-smoke
```

La prueba breve ejecuta exactamente 512 interacciones, con un entorno, rollouts
de 128 pasos y dos épocas por actualización. Comprueba cambios de parámetros,
valores finitos y predicciones idénticas después de guardar y cargar.
No demuestra aprendizaje de conducción ni superioridad sobre el controlador.

El entrenamiento configurado de 500000 pasos se inicia solo cuando se decida
realizar ese experimento:

```bash
uv run f1-ai train --output runs/ppo-500k
uv run f1-ai evaluate --agents random pursuit ppo --split validation \
  --model runs/ppo-500k/best_model.zip --output runs/validation-500k
```

PPO completa rollouts enteros, por lo que puede superar ligeramente el número
solicitado de pasos. Se registra el total real. Cada 25000 interacciones se
evalúan los circuitos de validación y se guarda un checkpoint. El mejor modelo
se elige por retorno medio de validación. La interrupción con Ctrl+C conserva
un modelo parcial y el número de interacciones realizadas.

Solo deben cargarse checkpoints locales de confianza: SB3 usa serialización
que puede ejecutar código al abrir archivos de terceros. La carga también
comprueba versión de observaciones y configuración del entorno.

## Fuentes de implementación

- [Código de CarRacing 1.2.3](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/box2d/car_racing.py)
- [Dinámica original del coche](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/box2d/car_dynamics.py)
- [PPO en Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html)
- [Comparación de métodos de conducción en F1TENTH](https://arxiv.org/abs/2402.18558v2)
