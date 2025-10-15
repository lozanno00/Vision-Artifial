https://github.com/lozanno00/Vision-Artifial.git
##🧠 Proyecto CNN con CIFAR-10

Este repositorio implementa una red neuronal convolucional (CNN) para la clasificación de imágenes del dataset CIFAR-10, utilizando TensorFlow y Keras.

El desarrollo se divide en tres hitos principales:

Preparación del dataset visual

Construcción del “Córtex Visual” (modelo CNN)

Entrenamiento y evaluación

##📦 Milestone 1: Preparación del Dataset Visual
🎯 Objetivo

Cargar, explorar y preprocesar el dataset CIFAR-10 para su uso en una CNN.

🔹 Pasos principales

Cargar el dataset desde TensorFlow y dividirlo en entrenamiento y prueba.

Explorar visualmente el contenido mostrando ejemplos por clase.

Normalizar los valores de los píxeles al rango entre 0 y 1.

Convertir las etiquetas numéricas al formato One-Hot.

Validar que las dimensiones, tipos y rangos de los datos sean correctos.

✅ Resultado esperado

Dataset limpio, normalizado y listo para usarse en la red neuronal.

##🧩 Milestone 2: Construcción del “Córtex Visual” (Modelo CNN)
🎯 Objetivo

Diseñar la arquitectura CNN con capas convolucionales para la extracción de características y capas densas para la clasificación.

⚙️ Componentes

Inicialización de un modelo secuencial vacío.

Primer bloque convolucional con reducción mediante pooling.

Segundo bloque convolucional para detectar patrones más complejos.

Capas densas encargadas de la clasificación final.

Compilación del modelo con optimizador Adam, pérdida categorical crossentropy y métrica de precisión.

🧠 Análisis

La CNN replica las etapas clásicas de la visión por computadora: entrada, preprocesamiento, extracción de características y clasificación, automatizando el proceso de detección de patrones.

✅ Resultado esperado

Modelo construido, compilado y preparado para el entrenamiento.

##🚀 Milestone 3: Entrenamiento y Evaluación
🎯 Objetivo

Entrenar el modelo, visualizar el proceso de aprendizaje y evaluar su rendimiento final.

🔹 Pasos principales

Compilar el modelo antes del entrenamiento.

Entrenar la red con al menos diez épocas y reservar una parte del conjunto para validación.

Visualizar las métricas de precisión y pérdida durante el entrenamiento.

Evaluar el rendimiento final del modelo en el conjunto de prueba.

Analizar los resultados, identificando sobreajuste o subajuste, y proponer mejoras como regularización o aumento de datos.

✅ Resultado esperado

Curvas de entrenamiento claras, modelo evaluado y análisis técnico del rendimiento.

##🧰 Tecnologías Utilizadas

Python 3

TensorFlow / Keras

Matplotlib

NumPy

📁 Estructura del Proyecto

dataset/: contiene los datos CIFAR-10

notebooks/: desarrollo paso a paso de los milestones

models/: definición del modelo CNN

results/: gráficos y métricas obtenidas

README.md: documentación del proyecto

📜 Licencia

Proyecto de uso educativo.
Libre para modificar y distribuir citando la fuente original.
