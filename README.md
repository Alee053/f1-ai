# Agente de conducción en carreras 2D

Avance del proyecto de Inteligencia Artificial SIS-252, UCB Cochabamba, gestión II-2026.
El objetivo principal es entrenar y evaluar una política neuronal mediante imitación
y ajuste con PPO en un entorno existente. Gymnasium CarRacing aporta el circuito
y la física Box2D del perfil `classic`, utilizado en los experimentos iniciales.
La etapa actual usa `formula-v1`, con dinámica planar en unidades SI, mandos
directos y práctica de frenada. Sus parámetros aproximan un monoplaza.

- [Política con mandos directos y currículo](#política-con-mandos-directos)
- [Conducción con mando Xbox](#conducción-manual-con-xbox)
- [Reproducción cenital o en perspectiva](#v%C3%ADdeo-de-la-pol%C3%ADtica)

La primera etapa incluye observaciones numéricas, un controlador de referencia,
separación de circuitos por semillas, entrenamiento reproducible y evaluación.
A* y el desgaste de neumáticos quedan para etapas posteriores.

Se utiliza Python 3.12 y `uv`. Las dependencias se instalan en `.venv`. En Linux y
Windows se utiliza la distribución PyTorch con CUDA 13.0 del índice oficial;
los perfiles existentes conservan CPU como dispositivo predeterminado. La ejecución
CUDA requiere una GPU NVIDIA y un controlador compatible. No se requieren servicios
pagados ni instalaciones globales de Python o del toolkit CUDA.

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

La adaptación numérica del perfil `classic` usa CarRacing de Gymnasium 1.2.3, con Box2D a 50 Hz.
Sobrescribe la salida interna `state_pixels`, que en el entorno original se
renderiza incluso sin ventana. No modifica el paquete instalado. Una prueba del perfil clásico compara
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
El dispositivo se configura con `device="cpu"` o `device="cuda"`. Una solicitud
CUDA falla si PyTorch no puede usar la GPU; no se sustituye silenciosamente por CPU.
Los perfiles históricos usan un hilo de PyTorch y cuatro entornos secuenciales
mediante `DummyVecEnv`. `vec_env="subproc"` ejecuta esos entornos en procesos
separados con `spawn`, sin heredar el contexto CUDA. La física Box2D sigue en CPU.

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
La interrupción con Ctrl+C o SIGTERM durante el aprendizaje conserva
`interrupted_model.zip` y registra el motivo y las interacciones en `interrupted.json`.
Al recibir SIGTERM, el comando termina con código 143 después de guardar el modelo.

Para continuar ese estado, incluso pasando de CPU a GPU:

```bash
uv run f1-ai --config configs/formula-refine-cuda.toml train \
  --resume-model runs/formula-refine-42-2m/interrupted_model.zip \
  --output runs/formula-refine-cuda-42-2m
```

`--resume-model` conserva el actor, el crítico, el optimizador, la exploración
aprendida y el contador. `total_timesteps` es el presupuesto acumulado, por lo que
solo se ejecutan los pasos restantes. Se reinician los episodios y los generadores
aleatorios porque no se guarda el estado de Box2D. La reanudación conserva los
hiperparámetros de aprendizaje; permite cambiar el dispositivo y el backend de
entornos. `--initial-model` transfiere pesos e inicia un optimizador nuevo.
Ambas opciones son excluyentes.

Los metadatos registran el dispositivo real, la GPU y la ubicación de los estados
del optimizador. Evaluación y vídeo usan el dispositivo del TOML y registran dónde
se ejecutó la inferencia. La instalación sigue las guías oficiales de
[PyTorch](https://pytorch.org/get-started/locally/) y
[uv](https://docs.astral.sh/uv/guides/integration/pytorch/).

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
uv run f1-ai --config configs/ppo-imitation-conservative.toml train \
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
uv run f1-ai --config configs/ppo-imitation-conservative.toml replay \
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


## Conducción manual con Xbox

```bash
uv sync --locked
uv run f1-ai controllers
uv run f1-ai play
```

`play` utiliza `configs/formula.toml` por defecto. Los demás subcomandos mantienen
`configs/ppo.toml` como configuración predeterminada. El mando se identifica con
el mapeo estándar SDL2, compatible con Xbox por USB o Bluetooth cuando SDL lo
reconoce. El nombre mostrado por Linux puede ser "Xbox 360 Controller" incluso
para un dispositivo conectado por Bluetooth.

La ventana comienza en pausa. Se debe hacer clic en ella y pulsar START o espacio.
Conviene arrancar con RT progresivamente: el modelo permite patinamiento al aplicar
potencia excesiva y bloqueo al frenar. No tiene ABS ni control de tracción físicos.

| Acción | Mando Xbox | Teclado |
| --- | --- | --- |
| Dirección | Stick izquierdo | Flechas izquierda/derecha |
| Acelerador | RT | Flecha arriba |
| Freno | LT | Flecha abajo |
| Pausa y reanudación | START o B | Espacio o P |
| Reiniciar la pista | A | R |
| Siguiente pista | X | N |
| Abrir ajustes | Y | TAB |
| Seleccionar parámetro | Cruceta arriba/abajo | W/S |
| Modificar parámetro | Cruceta izquierda/derecha | A/D |
| Comparar carga aero encendida/apagada | LB | B |
| Guardar configuración | RB | F5 |
| Mostrar el eje de referencia | Sin asignación | G |
| Salir | BACK | ESC |

Si se pulsan ambos gatillos, el freno tiene prioridad. La zona muerta y la curva
del stick ajustan la entrada del mando y no modifican la adherencia física.
Se pueden configurar al arrancar:

```bash
uv run f1-ai play --deadzone 0.08 --steering-curve 1.6
uv run f1-ai play --controller 0 --seed 3
```

La pérdida de foco o de conexión pausa la sesión y exige reanudación explícita.
La práctica utiliza solamente las semillas de entrenamiento. Permite recuperar
el coche en césped sin terminar el episodio tras una salida breve; los límites
de práctica se registran aparte y no sustituyen los del TOML exportado.

Cada sesión crea un directorio nuevo `runs/manual-...` con metadatos, telemetría
CSV, captura final y configuración. F5/RB escribe `saved-001.toml`, etc. Los cambios
de parámetros reinician la misma pista y dejan la sesión en pausa. El ajuste LB
cambia únicamente downforce y mantiene drag para aislar el efecto del agarre;
no representa DRS ni aerodinámica activa del reglamento real.

```bash
uv run f1-ai --config runs/manual-.../saved-001.toml play
uv run f1-ai --config configs/ppo-imitation-conservative.toml play
```

El segundo comando permite probar el coche clásico con los mismos controles.
Las velocidades de ese perfil conservan unidades del simulador, sin etiqueta
km/h. La física siempre avanza a 50 Hz con un acumulador independiente del render.
La cámara muestra más pista por delante al aumentar la velocidad.

## Modelo Formula v1

El nuevo coche conserva el trazado procedural de CarRacing y sustituye la dinámica
antes del primer paso físico. Unidades del perfil: metros, segundos, kg y N.
Se reutiliza la forma de la carrocería, redimensionada a la batalla y anchura
seleccionadas. El código del motor físico instalado permanece intacto.

- Masa total de 770 kg, incluidos cuatro cuerpos de rueda de 12 kg.
- Potencia agregada pico de 750 kW, aproximadamente 1006 hp mecánicos, con eficiencia
  de transmisión de 0.90. No se simula el despliegue energético de una batería.
- Batalla de 3.4 m y anchura exterior de 1.9 m.
- Propulsión trasera, giro delantero de hasta 0.4 rad y límite de velocidad del actuador.
- Dinámica angular de las ruedas con par motor, freno y reacción de contacto.
- Fuerzas longitudinales y laterales por slip, con saturación combinada, pérdida
  de agarre por deslizamiento y sensibilidad a carga del neumático.
- Transferencia de carga cuasiestática longitudinal y lateral. No hay suspensión 3D.
- Downforce y drag proporcionales al cuadrado de velocidad, más resistencia de rodadura.

La aerodinámica utiliza `F = 0.5 * rho * coeficiente_por_area * v²`. Downforce
incrementa la carga normal y el agarre de las ruedas; la masa inercial del coche
permanece fija. Drag aplica una fuerza contraria a la velocidad. Se usan diez
subpasos para integrar la rotación de las ruedas por cada paso físico externo.

El preset usa densidad del aire 1.225 kg/m³, ClA 5.0 m² y CdA 1.2 m². Estos
coeficientes, el agarre base 1.65, el centro de gravedad y los límites de par son
elecciones de ingeniería editables, no mediciones de un equipo F1. A 100, 200 y
300 km/h, el downforce calculado es aproximadamente 2.36, 9.45 y 21.27 kN.
La densidad elegida tampoco representa una medición de la atmósfera local.

Los ajustes están bajo `[environment.formula]` en `configs/formula.toml`. Los
campos omitidos utilizan los valores explícitos de `FormulaConfig`. El editor
manual expone masa, potencia, agarre, ClA, CdA, reparto de freno y densidad del aire.

Las referencias públicas 2026 son [F1 sobre dimensiones y masa](https://corp.formula1.com/f1-2026-regulations-terminology-update/),
[FIA sobre las unidades de potencia](https://www.fia.com/news/f1s-new-era-everything-you-need-know-about-how-fia-making-formula-1-more-competitive-more)
y las ecuaciones de [sustentación](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/lift-equation/)
y [drag de NASA](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/drag-equation/).
La suma de 750 kW es una simplificación de potencia pico. Los límites de despliegue
eléctrico y las condiciones de masa de la [FIA, sección C, Issue 21](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_c_technical_-_iss_21_-_2026-10-19.pdf)
son más detallados. No se simulan marchas, diferencial, temperatura de neumáticos,
desgaste, combustible variable ni aerodinámica activa.

Los resultados de los entrenamientos anteriores corresponden a `classic`.
La carga de un checkpoint compara el perfil y todos sus parámetros: rechaza
utilizarlo con Formula o con una configuración física distinta. Los modelos
necesitan entrenarse nuevamente para esta dinámica; las comprobaciones breves de
integración no demuestran conducción aprendida en el perfil nuevo.

### Entrenamiento de Formula

`configs/formula-training.toml` mantiene la física del preset manual y amplía el
horizonte a 6000 pasos, equivalentes a 120 segundos. Permite evaluar al experto
inicial, que tarda más de 60 segundos en algunas pistas. Las comparaciones de
Formula utilizan este mismo horizonte y las mismas condiciones de salida de pista.

La referencia usa la batalla y las unidades del perfil. En Formula añade
amortiguación de guiñada, anticipa la curva con los cuatro puntos observados y
convierte la aceleración deseada en fuerza y demanda de pedal. Compensa drag y
rodadura y limita su demanda de acelerador y freno. Estos límites pertenecen al
controlador de referencia; el coche conserva patinamiento y bloqueo. Los parámetros
de la referencia se seleccionaron en las semillas de entrenamiento.

```bash
uv run f1-ai --config configs/formula-training.toml demonstrations \
  --target-speed 28 --lateral-acceleration 12 --output runs/formula-demos
uv run f1-ai --config configs/formula-training.toml clone \
  --dataset runs/formula-demos --epochs 60 --dagger-rounds 2 \
  --seed 42 --output runs/formula-clonado
uv run f1-ai --config configs/formula-training.toml train \
  --initial-model runs/formula-clonado/best_model.zip \
  --seed 42 --output runs/formula-ajuste
uv run f1-ai --config configs/formula-training.toml evaluate --agents bc-ppo \
  --model runs/formula-ajuste/best_model.zip --split validation \
  --output runs/formula-validacion
uv run f1-ai --config configs/formula-evaluation-expanded.toml evaluate --agents bc-ppo \
  --model runs/formula-ajuste/best_model.zip --split validation \
  --output runs/formula-validacion-ampliada
```

La selección utiliza las cinco semillas 100 a 104. La validación ampliada agrega
las semillas 105 a 119 para describir el checkpoint elegido; no añade etiquetas
de imitación. Ninguna de estas dos evaluaciones es la prueba final.

El piloto `configs/formula-cold-pilot.toml` permite comprobar PPO desde cero con
100000 transiciones solicitadas y evaluación cada 25000. Comparte el entorno del
ajuste por imitación, pero usa otra exploración y tasa de aprendizaje. Un piloto
corto sin vueltas completas no permite descartar PPO ni comparar métodos con el
mismo presupuesto. Las semillas 200 a 204 siguen reservadas para la prueba final.

`configs/formula-training-kl002.toml` prueba `target_kl=0.002` frente a `0.01` desde
el mismo checkpoint DAgger y la misma semilla. Conserva los demás parámetros de
optimización, solicita 500000 pasos y evalúa cada 50000. El límite KL puede detener
antes las épocas de PPO; no limita la distancia acumulada respecto a la imitación
ni garantiza conservar su conducción. La comparación utiliza el tramo común de
500000 interacciones y distingue los checkpoints que completan vueltas con salidas.

### Referencia global de trazada y velocidad

La nueva referencia prepara un experto orientado al tiempo de vuelta. Se comparan
trazadas desplazadas del eje y candidatos que reducen longitud o curvatura mediante
una búsqueda acotada. Cada candidato recibe un perfil de velocidad; se elige el de
menor tiempo estimado. La cercanía al centro no forma parte de ese criterio.
Esto no demuestra una trazada óptima: se exploran pocos candidatos con un modelo
aproximado y aún se deben medir vueltas completas en Box2D.

El perfil incorpora carga aerodinámica, drag, sensibilidad del neumático a la
carga, propulsión trasera, reparto de freno y un presupuesto combinado de fuerzas
por eje. Las pasadas de aceleración y frenado incluyen el cierre de la pista para
anticipar las curvas siguientes. La estimación corresponde a una vuelta lanzada;
los ensayos físicos actuales salen desde parado. No son tiempos equivalentes.

`RacingController` sigue la trazada con pure pursuit y regula la velocidad usando
el perfil. Limita la rapidez de cambio del volante y del pedal, reduce acelerador
ante slip trasero excesivo y modera el freno al detectar bloqueo. Esa regulación
pertenece al experto; la física mantiene su capacidad de patinar y bloquear ruedas.
Los umbrales son parámetros iniciales pendientes de ajuste en conducción.

Este experto recibe el mapa completo, los parámetros físicos y el estado SI del
coche, incluidos slip y bloqueo. Tiene más información que la red actual de 16
entradas. Los artefactos lo identifican como `full_map_and_physical_state`; no se
mezcla con las evaluaciones de PPO o BC. La preparación de demostraciones nuevas
requerirá decidir qué información y anticipación recibirá el alumno.

Para preparar un plan desde un mapa ya guardado, sin generar pistas ni ejecutar
pasos físicos, se puede utilizar:

```bash
uv run f1-ai plan-reference --track runs/formula-reference-01/track-0.json \
  --samples 64 --iterations 4 --output runs/reference-plan-0
```

El JSON de entrada contiene las coordenadas `[[x, y], ...]` en metros, ordenadas
según el sentido de marcha, sin repetir el primer punto al final. `--half-width`
indica la semianchura de pista y por defecto usa la de CarRacing, 6.667 m.
El corredor reserva media anchura del coche y `--road-margin` adicional. Se
comprueban puntos de cada segmento; no se certifica la huella completa de la
carrocería, la ausencia de cruces ni la dinámica transitoria del neumático.

El comando de conducción permite una comprobación acotada o, cuando se decida
evaluarlo, un episodio físico completo. Solo admite semillas de entrenamiento
mientras se desarrolla esta referencia:

```bash
uv run f1-ai reference --seed 0 --steps 12 \
  --samples 64 --iterations 2 --output runs/reference-smoke-0
uv run f1-ai --config configs/formula-training.toml reference \
  --seed 0 --output runs/reference-driving-0
```

Se guardan configuración, revisión Git, plan y trayectoria con slip y bloqueo.
Los fallos no reciben un tiempo de vuelta. No se entrenan redes ni se modifica
la recompensa existente. El modelo del plan omite transferencia longitudinal y
rotación de ruedas; sus límites de velocidad se deben contrastar y ajustar con
los ensayos físicos antes de usarlo como profesor de BC o DAgger.

`--time-search-passes 3` prueba desplazamientos laterales suaves y ordena
las alternativas por un tiempo aproximado que incluye aceleración y frenada.
Cada alternativa final pasa por el corredor, el límite de giro y el perfil
completo de velocidad antes de competir con las referencias geométricas. En
un TOML se activa con `time_search_passes = 3` en
`[environment.racing_reference.plan]`.

### Aprendizaje de la referencia global

`configs/formula-racing-cuda.toml` define una generación independiente. Su perfil
`racing-v2` contiene 34 entradas: el estado base, ocho puntos de pista entre 6 y
180 m normalizados por su distancia, slip y bloqueo de las cuatro ruedas, y los
dos mandos aplicados en el paso anterior. El alumno observa la carretera, sin
recibir la trazada planificada ni el perfil de velocidad del profesor. Por ello
el experto sigue disponiendo de información privilegiada. Los modelos anteriores
conservan el contrato de 16 entradas y se rechaza cargarlos como esta generación.

La recompensa `lap-time-v1` asigna una cantidad fija por avance válido y un coste
por paso. No pondera el avance por centrado o alineación con el eje. Añade costes
menores por cambios bruscos de mando, slip longitudinal por encima de 0.2 y deriva
de carrocería por encima de 0.15 rad. Son umbrales iniciales del experimento, no
límites identificados de un F1 real. No se considera éxito cruzar la meta fuera
de carretera. La selección de modelos prioriza las vueltas completas sin salidas.

```bash
uv run f1-ai --config configs/formula-racing-cuda.toml demonstrations \
  --teacher racing --output runs/racing-demos
uv run f1-ai --config configs/formula-racing-cuda.toml clone \
  --dataset runs/racing-demos --epochs 60 --dagger-rounds 2 \
  --output runs/racing-dagger
uv run f1-ai --config configs/formula-racing-cuda.toml train \
  --initial-model runs/racing-dagger/best_model.zip --output runs/racing-ppo
```

DAgger vuelve a planificar cada pista de entrenamiento y etiqueta los estados
visitados por el alumno. Las rampas del profesor se calculan desde el mando que
realmente se aplicó. La red se ajusta en CUDA; la recolección y Box2D siguen en
CPU. El preset usa dos capas de 256 unidades y solicita un millón de interacciones
PPO, con tres épocas por actualización, tasa 3e-6 y KL objetivo 0.003. Estos
parámetros prueban un ajuste conservador; no garantizan evitar regresiones ni
encontrar la trazada óptima. Cada etapa conserva sus checkpoints y metadatos.

`train --rehearsal-dataset RUTA` activa una variante que alterna una actualización
PPO estándar con cuatro minibatches supervisados del actor. Utiliza un Adam
separado a 1e-4 y lotes de 256 demostraciones. El crítico y la exploración solo
reciben actualizaciones PPO. El repaso intenta reducir la pérdida de imitación
durante el ajuste; no impone una cota respecto al profesor ni garantiza estabilidad.
Es una alternancia PPO/BC, no una pérdida conjunta modificada de PPO.

El dataset de repaso debe corresponder al entorno y contener únicamente etiquetas
de entrenamiento. Se registran ruta, SHA y parámetros. El corpus y su optimizador
supervisado se excluyen del checkpoint. Para reanudar esta variante se debe indicar
el mismo dataset: se restaura el optimizador PPO y se reinician el Adam supervisado
y su muestreador. La predicción de un modelo guardado utiliza solamente la red.

El preset `formula-racing-rehearsal-cuda.toml` reduce `log_std_init` a -5 y eleva
`gamma` a 0.999. Se prueba junto al repaso; el experimento cambia varios factores
y no permite atribuir un resultado a uno solo. Ejemplo de inicio:

```bash
uv run f1-ai --config configs/formula-racing-rehearsal-cuda.toml train \
  --initial-model runs/racing-dagger/best_model.zip \
  --rehearsal-dataset runs/racing-dagger-dataset \
  --output runs/racing-ppo-rehearsal
```

### PPO con correcciones de trazada y velocidad

`formula-residual-cuda.toml` usa el contrato de observación 3. La política recibe
la trazada, sus velocidades y curvaturas hasta 100 m por delante y el estado
del coche. Cada 0.1 s propone desplazamientos de hasta 1.6 m y correcciones de
hasta ±25 % de la velocidad objetivo. Se filtran las consignas y el seguidor
actúa a 50 Hz. Con correcciones cero se reproduce el controlador original.

La recompensa se suma por paso físico. El presupuesto de PPO cuenta decisiones
de la red, con hasta cinco pasos físicos por decisión. La selección prioriza
vueltas limpias, vueltas completas y tiempo medio de las vueltas limpias, en ese
orden. El tiempo entre métodos debe compararse en las mismas pistas completadas.
La inferencia de este perfil requiere el planificador y el seguidor, además de la
red. Es una aplicación de la idea de un controlador residual, descrita para coches
de carreras por [Trumpp et al. (2023)](https://arxiv.org/abs/2302.07035).

```bash
uv run f1-ai --config configs/formula-residual-cuda.toml train \
  --output runs/formula-residual-ppo
uv run f1-ai --config configs/formula-residual-cuda.toml evaluate \
  --agents reference ppo --split validation \
  --model runs/formula-residual-ppo/best_model.zip --output runs/residual-comparison
```

`formula-residual-time-cuda.toml` mantiene la política y pondera el avance por
el tiempo previsto de cada sector. Se estima el tiempo de cada segmento con
el perfil de velocidad y se reparte el bono de progreso según ese tiempo. Una
vuelta limpia conserva el mismo bono total. Durante la vuelta, un metro en una
curva lenta aporta más que un metro en una recta rápida.

`--braking-fraction` reduce el frenado supuesto al calcular el perfil de
velocidad, para empezar a frenar con más distancia disponible. Conserva el
límite del actuador en la simulación. En el control residual se pueden fijar
las opciones del planificador en `[environment.racing_reference.plan]` y las
del seguidor en `[environment.racing_reference.follower]`. Se registran en
los metadatos y forman parte del contrato de cada checkpoint.

`formula-residual-braking-cuda.toml` usa una fracción de agarre de 0.88, un
factor de frenado planificado de 0.50, 0.28 s de anticipación del seguidor y
ganancia de velocidad de 1.2. Para medir esa referencia con la red desactivada:

```bash
uv run f1-ai --config configs/formula-residual-braking-cuda.toml evaluate \
  --agents reference --split validation --output runs/reference-early-braking
```

### Política con mandos directos

`formula-direct-cuda.toml` activa `planned-v3`, con 41 entradas y contrato 4.
La red recibe el estado físico y la trazada prevista con velocidades y curvaturas.
Aplica directamente giro y pedal firmado a 50 Hz. Durante imitación, el seguidor
calibrado etiqueta acciones en train. Durante la evaluación de la red, el
planificador proporciona geometría y el seguidor queda fuera del control.
El efecto del ajuste PPO se mide frente al checkpoint de imitación del que parte.

```bash
uv run f1-ai --config configs/formula-direct-cuda.toml demonstrations \
  --teacher racing --output runs/direct-demos
uv run f1-ai --config configs/formula-direct-cuda.toml clone \
  --dataset runs/direct-demos --epochs 40 --dagger-rounds 2 --output runs/direct-bc
uv run f1-ai --config configs/formula-direct-cuda.toml train \
  --initial-model runs/direct-bc/best_model.zip --output runs/direct-ppo
```

### Observaciones del corredor

`formula-corridor-cuda.toml` añade los bordes izquierdo y derecho del asfalto
a 6, 15, 30, 60 y 100 m. Las coordenadas se expresan respecto al coche. La
red conserva las entradas de la trazada y el mando directo sobre giro y pedal.
Se puede iniciar desde `planned-v3`. Las columnas nuevas del actor y del
crítico empiezan en cero.

```bash
uv run f1-ai --config configs/formula-corridor-cuda.toml train \
  --initial-model runs/ppo-stable/model.zip --output runs/corridor-ppo
```

`time_progress_mix` en `[ppo]` acepta valores entre 0 y 1. El valor 1
distribuye el premio de avance según el tiempo de referencia por sector. El
total de avance en una vuelta completa se conserva y la evaluación usa el
mismo perfil `lap-time-v1`.

### Currículo de curvas

`formula-curriculum-cuda.toml` parte de un checkpoint de mandos directos y sube
`log_std_init` de -5 a -4,4. Durante el entrenamiento mezcla vueltas completas
con sectores de hasta 120 m o 9 s. Los sectores empiezan de 25 a 55 m antes de
una curva, con velocidad inicial basada en el perfil de la trazada.

La probabilidad de iniciar un sector baja del 70 % al 55 % a los 32.768 pasos
y al 25 % a los 81.920. En esas transiciones aumenta la variación de velocidad
y orientación. La validación usa vueltas completas desde la meta. El fin de un
sector solo trunca el episodio.

El ajuste opcional `rear_slip_penalty` en `[ppo]` añade un coste al superar
0,30 de deslizamiento longitudinal en una rueda trasera sobre el asfalto.
La política conserva el giro y el pedal que haya elegido.

```bash
uv run f1-ai --config configs/formula-curriculum-cuda.toml train \
  --initial-model runs/direct-ppo/best_model.zip --output runs/corner-ppo
```

`formula-braking-ppo-cuda.toml` prioriza sectores con una caída prevista de
velocidad antes del ápice. Su coste adicional `front_lock_penalty` se aplica al
deslizamiento longitudinal delantero durante el frenado y al bloqueo de una
rueda delantera. Para un primer punto de control de 196.608 pasos:

```bash
uv run f1-ai --config configs/formula-braking-ppo-cuda.toml train \
  --initial-model runs/ppo-stable/model.zip --timesteps 196608 \
  --output runs/braking-pilot
```

Se puede reanudar desde un checkpoint evaluado con `--resume-model`. El
presupuesto de la configuración es de 1.998.848 interacciones, alineado con
los lotes de 2.048 pasos. La evaluación emplea vueltas completas desde la meta
con la física y observación de la política de mandos directos.

### Vista de persecución

La cámara `chase` muestra el circuito plano desde atrás con una inclinación de
27,5 grados hacia abajo. La cuadrícula está anclada al mundo, con divisiones de
un metro y líneas principales cada cinco. En conducción manual se puede cambiar
de vista con `C` o pulsando el stick derecho del mando. La vista es independiente
de las observaciones de la política y de la configuración física.

```bash
uv run f1-ai play --camera chase --grid-meters 1
uv run f1-ai --config configs/formula-direct-cuda.toml replay \
  --model runs/direct-ppo/best_model.zip --seed 117 --camera chase \
  --grid-meters 1 --device cpu --output runs/replays/formula-chase.mp4
```

El MP4 toma 25 cuadros por segundo de los 50 pasos físicos del entorno. En
`chase` incluye velocidad, gas, freno y deslizamiento. `top` mantiene la vista
cenital.
