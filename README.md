https://github.com/lozanno00/Vision-Artifial.git


Proyecto CNN con CIFAR-10

Este proyecto implementa una red neuronal convolucional (CNN) paso a paso para clasificar imágenes del dataset CIFAR-10 utilizando TensorFlow y Keras.

El trabajo se divide en tres milestones principales:

Preparación del Dataset Visual

Construcción del “Córtex Visual” (modelo CNN)

Entrenamiento y Evaluación

📍 Milestone 1: Preparación del Dataset Visual (CIFAR-10)
🎯 Objetivo

Cargar, explorar y preprocesar el dataset CIFAR-10 para dejarlo listo para una CNN.

🧩 Issue 1: Cargar el dataset CIFAR-10 desde TensorFlow

Descripción:
Importar y cargar el dataset cifar10 usando tensorflow.keras.datasets. Separar los conjuntos de entrenamiento y prueba.

Tareas:

Importar el módulo tensorflow.keras.datasets.

Cargar los datos CIFAR-10.

Imprimir las dimensiones de los datasets.

Criterios de aceptación:

El dataset se carga sin errores.

Se muestran correctamente las dimensiones de x_train, y_train, x_test, y_test.

🧩 Issue 2: Explorar visualmente las imágenes del dataset

Descripción:
Visualizar algunas imágenes de cada categoría para comprender la variedad dentro de las clases.

Tareas:

Crear una función para mostrar imágenes aleatorias por clase.

Mostrar 2-3 ejemplos por clase con su etiqueta.

Criterios de aceptación:

Las imágenes se visualizan correctamente.

La función puede reutilizarse para análisis posteriores.

🧩 Issue 3: Normalizar los valores de píxeles

Descripción:
Escalar los valores de los píxeles al rango [0, 1].

Tareas:

Dividir los valores por 255.0.

Verificar los rangos tras la normalización.

Criterios de aceptación:

Los valores están entre 0 y 1.

No se altera la forma de las imágenes (32x32x3).

🧩 Issue 4: Convertir las etiquetas a formato One-Hot

Descripción:
Transformar las etiquetas numéricas (0–9) a formato one-hot encoding.

Tareas:

Usar tensorflow.keras.utils.to_categorical.

Comprobar la forma resultante (10 columnas).

Criterios de aceptación:

Las etiquetas se transforman correctamente.

La cantidad de clases es 10.

🧩 Issue 5: Validar el dataset procesado

Descripción:
Verificar formas, rangos y tipos de datos antes de pasar al modelo.

Tareas:

Imprimir las formas de los conjuntos.

Confirmar los rangos de valores y formato de etiquetas.

Criterios de aceptación:

Los datos están listos para usarse en una CNN.

No se perdió información durante el preprocesamiento.

🧠 Milestone 2: Arquitectura del “Córtex Visual” (Construcción del Modelo)
🎯 Objetivo

Construir la arquitectura CNN en Keras con un extractor de características (capas convolucionales) y un clasificador (capas densas).

⚙️ Issue 1: Iniciar el modelo secuencial en Keras

Descripción:
Crear un modelo tf.keras.Sequential() vacío.

Tareas:

Importar las capas necesarias.

Inicializar el modelo secuencial.

Mostrar su resumen inicial.

Criterios de aceptación:

El modelo se crea sin errores.

El resumen muestra 0 parámetros entrenables.

⚙️ Issue 2: Construir el bloque convolucional 1

Descripción:
Primer bloque del extractor de características.

Tareas:

Añadir Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)).

Añadir MaxPooling2D(pool_size=(2,2)).

Criterios de aceptación:

El modelo muestra ambas capas correctamente.

El tamaño se reduce tras el pooling.

⚙️ Issue 3: Construir el bloque convolucional 2

Descripción:
Segundo bloque convolucional para detectar patrones más complejos.

Tareas:

Añadir Conv2D(64, (3,3), activation='relu').

Añadir MaxPooling2D(pool_size=(2,2)).

Criterios de aceptación:

El modelo tiene cuatro capas en total (2 Conv + 2 Pool).

El tamaño del tensor se reduce correctamente.

⚙️ Issue 4: Construir el clasificador

Descripción:
Capas densas encargadas de clasificar las características extraídas.

Tareas:

Añadir Flatten().

Añadir Dense(64, activation='relu').

Añadir Dense(10, activation='softmax').

Criterios de aceptación:

El modelo muestra todas las capas esperadas.

Es compilable sin errores.

⚙️ Issue 5: Compilar el modelo

Descripción:
Preparar el modelo para el entrenamiento.

Tareas:

Usar optimizador 'adam'.

Pérdida 'categorical_crossentropy'.

Métrica 'accuracy'.

Criterios de aceptación:

El modelo compila sin errores.

El resumen muestra todos los parámetros correctamente.

🧩 Issue 6: Análisis — Relación con la visión artificial tradicional

Descripción:
Relacionar el modelo con las etapas clásicas de visión por computadora.

Tareas:

Identificar qué capas corresponden a Input, Preprocessing, Feature Extraction y Classifier.

Explicar cómo las CNN automatizan la extracción de características.

Criterios de aceptación:

El análisis es coherente y completo.

Se explica correctamente la automatización del “Feature Extraction”.

🚀 Milestone 3: Entrenamiento y Evaluación
🎯 Objetivo

Entrenar el modelo, visualizar el proceso de aprendizaje y evaluar su rendimiento final.

🧩 Issue 1: Compilar el modelo

Descripción:
Configurar optimizador, pérdida y métricas antes del entrenamiento.

Tareas:

Compilar con 'adam', 'categorical_crossentropy' y 'accuracy'.

Mostrar resumen del modelo.

Criterios de aceptación:

Compila correctamente.

Se muestran los parámetros totales.

🧩 Issue 2: Entrenar el modelo

Descripción:
Entrenar la CNN usando .fit() y reservar parte del conjunto para validación.

Tareas:

Entrenar con epochs=10 (mínimo).

Usar validation_split=0.1.

Guardar el historial (history).

Criterios de aceptación:

Entrenamiento sin errores.

Se generan métricas de entrenamiento y validación.

🧩 Issue 3: Visualizar el aprendizaje

Descripción:
Graficar la evolución de precisión y pérdida en entrenamiento y validación.

Tareas:

Crear gráficos con matplotlib.

Mostrar accuracy y loss por época.

Añadir títulos, leyendas y etiquetas.

Criterios de aceptación:

Se muestran ambas gráficas.

Las curvas reflejan correctamente el progreso del modelo.

🧩 Issue 4: Evaluar el rendimiento final

Descripción:
Medir la precisión final del modelo con el conjunto de prueba.

Tareas:

Ejecutar model.evaluate(x_test, y_test).

Mostrar la precisión final en consola.

Criterios de aceptación:

Evaluación sin errores.

Precisión mostrada correctamente (ej. “Accuracy: 0.78”).

🧩 Issue 5: Análisis de rendimiento

Descripción:
Interpretar los resultados del modelo tras el entrenamiento y evaluación.

Tareas:

Analizar sobreajuste o subajuste.

Sugerir mejoras potenciales (más capas, regularización, data augmentation, etc.).

Criterios de aceptación:

El análisis es técnico, claro y con posibles líneas de mejora.
