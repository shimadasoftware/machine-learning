# <img src="https://github.com/shimadasoftware/machine-learning/assets/73977456/157a767f-2deb-43a7-8023-71506a9ef97a" alt="Italian Trulli" style="width:35px;height:35px;"> Modelos de Machine Learning: Regresión logística

La clasificación se puede asociar con el proceso de asignar una "etiqueta de clase" a un artículo en particular. Uno de los algoritmos empleados para esta tarea es el de Regresión Logística.

## Regresión Logística

La regresión logística es un algoritmo de clasificación binaria que da la probabilidad de que algo sea verdadero o falso. En otras palabras, es un método estadístico para predecir clases binarias. Es uno de los algoritmos más simples y más usados para la clasificación de dos clases.

Ejemplos:
- Detección de cáncer.
- Determinar si un correo es un spam.
- Determinar si un cliente comprará un producto o no.
  
### En términos matemáticos: 

La regresión logística es un modelo estadístico utilizado para predecir la probabilidad de que una variable dependiente pertenezca a una categoría particular. Es especialmente útil cuando la variable dependiente es binaria, es decir, tiene dos categorías (por ejemplo, sí/no, 0/1).

La función logística (también conocida como función sigmoide) se utiliza en la regresión logística para transformar una combinación lineal de las variables independientes en un valor entre 0 y 1, que puede interpretarse como una probabilidad.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/img/regresi%C3%B3n%20log%C3%ADsitica.png)

Por ejemplo:
La siguiente gráfica varia entre valores del 0 al 1. Si el valor predicho se ubica entre 0.5 y 1, entonces se puede catalogar como si y el valor será la probabilidad de dicho evento. Por otra parte, si el valor se ubica entre 0 y menos que 0.5, se cataloga como no.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/img/regresi%C3%B3n%20log%C3%ADsitica%20gr%C3%A1fica.png)

## Ejemplo práctico 01

Se deben seguir tres pasos básicos:
  1. Preparar los datos
  2. Entrenar el modelo
  3. Realizar las predicciones.

Objetivo: Predecir si el cáncer es benigno o maligno.

### Dataset

Dataset de cáncer de mama de Wisconsin (diagnóstico).

#### Contexto:

Las características se calculan a partir de una imagen digitalizada de una aspiración con aguja fina (PAAF) de una masa mamaria. Describen características de los núcleos celulares presentes en la imagen.

#### Contenido
-  ID number:
  contador sencillo.

- Diagnosis (M = maligno, B = benigno):
  a) radius (media de las distancias desde el centro a los puntos del perímetro)
  b) texture (desviación estándar de los valores de escala de grises)
  c) perimeter
  d) area
  e) smoothness (variación local en la longitud del radio)
  f) compactness (perímetro^2 / área - 1,0)
  g) concavity (número de porciones cóncavas del contorno)
  h) concave points (number of concave portions of the contour)
  i) symmetry
  j) fractal dimension ("aproximación de la línea costera" - 1)

La media, el error estándar y el "peor" o mayor (media de los tres los valores más grandes) de estas características se calcularon para cada imagen, resultando en 30 características. Por ejemplo, el campo 3 es Radio medio, campo 13 es Radio SE, el campo 23 es Peor Radio.

Todos los valores de las características se recodifican con cuatro dígitos significativos.

Valores de atributos faltantes: ninguno

Distribución de clases: 357 benignos, 212 malignos

- **La media (mean):** es una medida estadística que representa el valor central de un conjunto de datos. También se conoce como el promedio aritmético. Para calcular la media, sumas todos los valores en un conjunto de datos y luego divides esa suma por la cantidad total de elementos en el conjunto.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/img/teor%C3%ADa%20media.png)

- **Desviación estándar:** es una medida de dispersión que indica cuánto se desvían los valores individuales de la media en un conjunto de datos.

<img src="https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/img/teor%C3%ADa%20desviaici%C3%B3n%20est%C3%A1ndar.png" alt="Italian Trulli" style="width:500px;height:500px;">

- **Error estándar (standar error):**

El error estándar es una medida de la variabilidad de una estadística muestral, como la media. Indica cuánto puede variar la media de una muestra con respecto a la media de la población. Se calcula dividiendo la desviación estándar de la muestra por la raíz cuadrada del tamaño de la muestra.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/img/teor%C3%ADa%20error%20desviaci%C3%B3n%20est%C3%A1ndar.png)

En resumen, la desviación estándar mide la dispersión de los valores en un conjunto de datos, mientras que el error estándar mide cuánto puede variar la media muestral en relación con la media poblacional. El error estándar se utiliza comúnmente en inferencia estadística para estimar la precisión de una estadística muestral.

- **Peor (worst):** es la media de los tres valores más grandes de las características que se calcularon para cada imagen.

El dataset se encuentra en https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

![image](https://github.com/shimadasoftware/machine-learning/assets/73977456/f2e41271-a2e0-44af-ad29-125db71d2fc9)

### Desarrollo:

Para ver el desarrollo planteado del reto ir al [notebook](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/Regresi%C3%B3n%20lineal/marathon/marathon.ipynb)
