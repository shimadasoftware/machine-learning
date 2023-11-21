# <img src="https://github.com/shimadasoftware/machine-learning/assets/73977456/157a767f-2deb-43a7-8023-71506a9ef97a" alt="Italian Trulli" style="width:35px;height:35px;"> Modelos de Machine Learning: Regresión lineal

Si se requiere hallar un valor continuo, entonces el problema es de **regresión**, de lo contrario se habla de clasificación. Ejemplo: los ingresos son valores continuos, mientras seleccionar una especie de flor es un valor discreto, dentro de un número limitado de posiblidades.

Generalización: ocurre cuando el modelo es capaz de hacer predicciones precisas sobre datos invisibles.

## Regresión Lineal

Es una técnica que se utiliza para poder predecir variables continuas dependientes, a partir de un conjunto de variables
independientes. Es de carácter paramétrico, debido a que las suposiciones se realizan a partir de un conjunto de datos previo conocido.

### En términos matemáticos: 

Se utiliza la ecuación de la recta para aproximar los datos.

```math
y = ax + b
```
Donde:
  - y: variable dependiente o variable a predecir.
  - x: variable independiente o variable predictora.
  - a: pendiente.
  - b: intersección con el eje y.

Con la regresión lineal se busca minimizar la distancia vertical entre los datos y la línea.
Uno de los métodos más usados es el conocido como Mínimos Cuadrados, donde se busca reducir el error entre la línea resultante y los puntos de los datos.

![image](https://github.com/shimadasoftware/machine-learning/assets/73977456/50ef57fd-2247-4876-9707-d2e25ed9e5e4)

### Método para solucionar los ejercicios

¿Cómo se soluciona un ejercicio con regresión lineal?

1. Se debe establecer un problema que:
  - Se conozca el contexto.
  - Se busque predecir algún dato o evento.

2. Se requieren datos sobre el problema que
  - Den resolución anterior al problema que se está resolviendo.
  - Contenga una buena cantidad de datos.
  - Sean fácil de conseguir o que ya se tengan.
  - Estén en formato de tabla (Excel o csv).

3. Se debe contar con un software para programar el modelo.

Ejemplo de ejercicios:
  - ¿Cuánto tiempo empleará un corredor en la próxima maratón?
  - ¿Qué menú ofrecer a un cliente nuevo?, según su edad, género y estilo.
  - ¿Cuánto me gastaré en el próximo mes?

## Ejemplo práctico 01

Se deben seguir tres pasos básicos:
  1. Preparar los datos
  2. Entrenar el modelo
  3. Realizar las predicciones.

Objetivo: Predecir cuánto tiempo gastará un corredor en correr una maratón.

### Dataset

#### Contexto:

Cada Maratonista tiene un objetivo de tiempo en mente, y este es el resultado de todo el entrenamiento realizado en meses de ejercicios. Las carreras largas, las zancadas, los kilómetros y el ejercicio físico, suman mejora al resultado. La predicción del tiempo de maratón es un arte, generalmente guiado por fisiólogos expertos que prescriben los ejercicios semanales y los hitos del maratón. El método "simple" El enfoque es mirar los datos después de la competencia, la tabla de clasificación.

#### Contenido
- id:
  contador sencillo.

- marathon:
  el nombre del maratón donde se extrajeron los datos. Utilizo los datos que salen de Strava "Comparación lado a lado" y los datos procedentes del resultado final del maratón.

- name:
  El nombre del atleta, todavía hay algunos problemas con UTF-8, lo solucionaré pronto.

- category:
  el sexo y grupo de edad de un corredor.

  - Atletas masculinos MAM menores de 40 años.
  - WAM Mujeres menores de 40 años.
  - M40 Atletas Masculinos entre 40 y 45 años.

-km4week
  Es el número total de kilómetros recorridos en las últimas 4 semanas antes del maratón, maratón incluido. Si por ejemplo el km4semana es 100, el deportista ha corrido 400 km en las cuatro semanas previas al maratón.

- sp4week
  Esta es la velocidad media del deportista en las últimas 4 semanas de entrenamiento. La media cuenta todos los kilómetros recorridos, incluidos los kilómetros lentos realizados antes y después del entrenamiento. Una sesión de carrera típica puede ser de 2 km de carrera lenta, luego de 12 a 14 km de carrera rápida y finalmente otros 2 km de carrera lenta. El promedio de la velocidad es este número, y con el tiempo este es uno de los números que hay que perfeccionar.

- cross training:
  Si el corredor también es ciclista, o triatleta, ¿cuenta? Utilice este parámetro para ver si el deportista también es elíptica en otras disciplinas.

- wall21:
  En decimales. El campo complicado. Para reconocer una buena actuación, como maratonista, tengo que correr la primera media maratón con el mismo intervalo de la segunda mitad. Si, por ejemplo, corro la primera media maratón en 1h30m, debo terminar la maratón en 3h (por hacer un buen trabajo). Si termino en 3h20m, empecé demasiado rápido y me golpeé "contra la pared". Mi historial de entrenamiento es, por tanto, menos válido, ya que no estaba estimando mi resultado.

- marathon time:
  En decimales. Este es el resultado final. Según mi historial de entrenamiento, debo predecir mi tiempo esperado en el maratón.

- category:
  Este es un campo auxiliar. Da alguna dirección, así que siéntete libre de usarlo o desecharlo. Se agrupa en:

  - Resultados en menos de 3h.
  - Resultados B entre 3h y 3h20m.
  - Resultados C entre 3h20m y 3h40m.
  - Resultados D entre las 3h40 y las 4h.

#### El objetivo de esta competencia:
Según mi historial de entrenamiento, debo predecir mi tiempo esperado en el maratón. ¿Qué otros datos relevantes podrían ayudarme a ser más preciso? Ritmo cardíaco, cadencia, entrenamiento de velocidad, ¿qué más? ¿Y cómo podría obtener esos datos?

El dataset se encuentra en https://www.kaggle.com/girardi69/marathon-time-predictions/version/2

![image](https://github.com/shimadasoftware/machine-learning/assets/73977456/742099b9-f052-4271-82ac-4d0840db3386)

### Desarrollo 

Para ver el desarrollo planteado del reto ir al [notebook](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/Regresi%C3%B3n%20lineal/marathon/marathon.ipynb)

## Ejemplo práctico 02

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

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/Regresi%C3%B3n%20lineal/img/teor%C3%ADa%20media.png)

- **Desviación estándar:** es una medida de dispersión que indica cuánto se desvían los valores individuales de la media en un conjunto de datos.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/Regresi%C3%B3n%20lineal/img/teor%C3%ADa%20desviaici%C3%B3n%20est%C3%A1ndar.png)

- **Error estándar (standar error):**

El error estándar es una medida de la variabilidad de una estadística muestral, como la media. Indica cuánto puede variar la media de una muestra con respecto a la media de la población. Se calcula dividiendo la desviación estándar de la muestra por la raíz cuadrada del tamaño de la muestra.

![image](https://github.com/shimadasoftware/machine-learning/blob/main/2.%20Modelos%20de%20Machine%20Learning/Regresi%C3%B3n%20lineal/img/teor%C3%ADa%20error%20desviaci%C3%B3n%20est%C3%A1ndar.png)

En resumen, la desviación estándar mide la dispersión de los valores en un conjunto de datos, mientras que el error estándar mide cuánto puede variar la media muestral en relación con la media poblacional. El error estándar se utiliza comúnmente en inferencia estadística para estimar la precisión de una estadística muestral.

- **Peor (worst):** es la media de los tres valores más grandes de las características que se calcularon para cada imagen.

### Desarrollo:

El dataset se encuentra en https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

![image](https://github.com/shimadasoftware/machine-learning/assets/73977456/f2e41271-a2e0-44af-ad29-125db71d2fc9)
