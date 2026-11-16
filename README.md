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
