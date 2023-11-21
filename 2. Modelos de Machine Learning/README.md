# <img src="https://github.com/shimadasoftware/machine-learning/assets/73977456/157a767f-2deb-43a7-8023-71506a9ef97a" alt="Italian Trulli" style="width:35px;height:35px;"> Modelos de Machine Learning

## Aprendizaje supervisado y no supervisado

Según la cantidad de supervisión humana que tienen los procesos de aprendizaje, los métodos de Machine Learning pueden clasificarse en:

### - Aprendizaje Supervisado
Necesita tener los datos tanto de la variable predictora X como de la variable a predecir Y. Se entrenan con datos etiquetados (datos de respuesta deseada).
Por ejemplo: para detectar si una TC es fraudulenta se hace el entrenamiento con TC tanto válidas como fraudulentas. Aplicable a problemas de regresión y clasificación.

El objetivo principal es definir un mapeo o asociación entre las muestras de datos de entrada x y sus correspondientes salidas y, basándose en múltiples instancias de datos de entrenamiento. Este conocimiento aprendido se puede utilizar en el futuro para predecir una salida y′ para cualquier nueva muestra de datos de entrada x′ que antes se desconocía o no se veía durante el proceso de entrenamiento del modelo. Estos métodos se denominan supervisados porque el modelo aprende sobre muestras de datos donde las respuestas o etiquetas de salida deseadas ya se conocen de antemano en la fase de entrenamiento.

El aprendizaje supervisado a menudo requiere un esfuerzo humano para construir el conjunto de capacitación, pero luego lo automatiza y, a menudo, acelera una tarea que de otro modo sería laboriosa o inviable.

Los métodos de aprendizaje supervisado son de dos clases, según el tipo de tareas de aprendizaje automático que pretenden resolver:

- Regresión: el objetivo es predecir un número real continuo. Ejemplo: predecir los ingresos anuales de una persona, conociendo su nivel de educación, su edad y el lugar donde vive.

- Clasificación: el objetivo es predecir una etiqueta de clase, que es una elección de una lista predefinida de posibilidades. Puede ser binaria (distinguir entre dos clases) o multiclase (clasificación entre más de dos clases). Ejemplo de binaria: responder pregunta sí o no – ¿Este es un correo spam?. Ejemplo de multiclase: la selección de especie de las flores de iris.

La clasificación se puede asociar con el proceso de asignar una "etiqueta de clase" a un artículo en particular. Algoritmos de clasificación:

- Regresión logística.
- Árbol de decisión.
- Bosque aleatorio.
- Bayes sencillo

<img src="https://github.com/shimadasoftware/machine-learning/assets/73977456/f4a245dc-8c60-4bc6-8746-fc1a6de4aca7" alt="Italian Trulli" style="width:600px;height:400px;">

### - Aprendizaje No Supervisado
No necesita los datos de la variable objetivo Y. Usan datos sin etiquetas y busca encontrar relaciones entre ellos. Aplicable a problemas de Agrupamiento o clustering.
