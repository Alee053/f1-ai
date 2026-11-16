# Agente de conducción en carreras 2D

Avance del proyecto de Inteligencia Artificial SIS-252, UCB Cochabamba, gestión II-2026.
El objetivo principal es entrenar y evaluar una política neuronal mediante imitación
y ajuste con PPO en un entorno existente. Gymnasium CarRacing aporta el circuito
y la física Box2D. No se modela un vehículo real de Fórmula 1.

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

La configuración original contiene dos capas de 64 unidades para el actor y
otras dos para el crítico. Las variantes permiten configurar ese tamaño y
emplean 128 unidades. Los hiperparámetros y la semilla se definen en los TOML de `configs/`.
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
uv run f1-ai plot-training --run runs/ppo-500k
uv run f1-ai evaluate --agents random pursuit ppo --split validation \
  --model runs/ppo-500k/best_model.zip --output runs/validation-500k
```

PPO completa rollouts enteros, por lo que puede superar ligeramente el número
solicitado de pasos. Se registra el total real. Cada 25000 interacciones se
evalúan los circuitos de validación y se guarda un checkpoint. El mejor modelo
se elige por vueltas completas, cobertura y tiempo, con retorno como último desempate.
La interrupción con Ctrl+C conserva
un modelo parcial y el número de interacciones realizadas.

Solo deben cargarse checkpoints locales de confianza: SB3 usa serialización
que puede ejecutar código al abrir archivos de terceros. La carga también
comprueba versión de observaciones y configuración del entorno.

## Imitación y ajuste con PPO

La variante corregida usa `steering_scale=0.4`: el intervalo normalizado se
convierte al rango físico de la dirección. Los modelos anteriores conservan
la escala 1 y no se cargan bajo la interpretación nueva. El perfil `track`
no premia el avance fuera de carretera y pondera el avance por centrado y
alineación. Las escalas y perfiles se registran en los metadatos. Sus retornos
no son directamente comparables con los del perfil original `progress`.

El controlador etiqueta observaciones exclusivamente en los circuitos de
entrenamiento. BC ajusta el actor por regresión supervisada. DAgger agrega
etiquetas del experto para estados visitados por el alumno. Se guarda y evalúa
cada etapa antes de iniciar PPO. En validación e inferencia actúa únicamente
la red; no existe un controlador de rescate.

```bash
uv run f1-ai --config configs/ppo-track-sde.toml demonstrations \
  --target-speed 40 --lateral-acceleration 12 --output runs/demos
uv run f1-ai --config configs/ppo-track-sde.toml clone \
  --dataset runs/demos --epochs 60 --dagger-rounds 2 --seed 42 --output runs/clonado
uv run f1-ai --config configs/ppo-imitation-gamma995.toml train \
  --initial-model runs/clonado/best_model.zip --seed 42 --output runs/ajuste
uv run f1-ai --config configs/evaluation-expanded.toml evaluate --agents bc-ppo \
  --model runs/ajuste/best_model.zip --min-completion 0.95 --output runs/evaluacion
uv run f1-ai plot --evaluation runs/evaluacion
```

Los ejemplos requieren carpetas de salida nuevas. `--seed` permite repetir
el procedimiento con otra inicialización y `--timesteps` cambiar el presupuesto
de PPO sin modificar el TOML. La validación ampliada contiene 20 circuitos,
semillas 100 a 119. Las semillas 200 a 204 siguen reservadas para prueba final.

La política gSDE tiene acciones acotadas mediante tanh y remuestrea los pesos
de exploración cada cuatro pasos; no mantiene una acción fija durante esos
cuatro pasos. `log_std_init` controla los pesos del ruido, cuya desviación de
acción también depende de las características latentes. La escala de recompensa
de entrenamiento se aplica fuera de Monitor: las métricas conservan el retorno
sin reescalar. No se utiliza normalización adaptativa con datos de validación.

`configs/ppo-imitation-conservative.toml` conserva la configuración de
`ppo-imitation-gamma995.toml` y reduce únicamente la tasa de aprendizaje de
0.00003 a 0.00001. Permite comparar el tamaño de las actualizaciones desde el
mismo checkpoint de imitación y con la misma semilla. Para evaluar la variación
se repiten tanto la imitación como PPO, con semillas 42, 43 y 44; las pistas
de evaluación siguen siendo las mismas.

`plot-training` muestra retorno, vueltas completas, cobertura y tiempo medio
de las vueltas completadas. Los puntos sin éxitos carecen de tiempo de vuelta.
El punto inicial de un ajuste corresponde a la política de imitación antes de
PPO. Si ninguna actualización la supera, la selección conserva ese punto:
no debe atribuirse a PPO una mejora que no ocurrió. Los archivos históricos
que solo contienen retornos siguen generando la curva de retorno.

Las etiquetas `bc`, `bc-ppo` y `ppo` distinguen imitación, PPO inicializado por
imitación y PPO desde cero. El evaluador rechaza etiquetas incompatibles con
la procedencia del checkpoint. `--min-completion` hace fallar el comando si
algún agente solicitado no alcanza la tasa requerida; sus mediciones se conservan.
Los checkpoints intermedios usan los metadatos de su directorio de corrida.

## Vídeo de la política

```bash
uv run f1-ai --config configs/ppo-imitation-gamma995.toml replay \
  --model runs/ajuste/best_model.zip --seed 100 --output runs/replays/agente.mp4
```

La exportación usa ffmpeg si está instalado y produce MP4, una captura PNG y
un JSON con el modelo, semilla y resultado. Ejecuta la red cargada sin invocar
al experto. La puntuación del HUD es la nativa de CarRacing; la evaluación
registra por separado el perfil de recompensa configurado.

## Fuentes de implementación

- [Código de CarRacing 1.2.3](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/box2d/car_racing.py)
- [Dinámica original del coche](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.3/gymnasium/envs/box2d/car_dynamics.py)
- [PPO en Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html)
- [Comparación de métodos de conducción en F1TENTH](https://arxiv.org/abs/2402.18558v2)
- [DAgger y distribución de estados inducida por el alumno](https://proceedings.mlr.press/v15/ross11a.html)
- [Exploración suave gSDE](https://proceedings.mlr.press/v164/raffin22a.html)
