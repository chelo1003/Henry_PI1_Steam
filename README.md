# Proyecto Steam - Recomendación de Juegos

Este proyecto consiste en crear un MVP (Minimum Viable Product) consumible mediante una API REST o RESTful, que recomiende de manera personalizada videojuegos a cada usuario (usuario-item). Para el análisis de sentimiento se utiliza la libreria TextBlob y para el entrenamiento del modelo de recomendación se utiliza la librería surprice.

# Diagrama de Flujo del Proyecto

Como se puede apreciar en el diagrama siguiente, el proceso de trabajo comienza con Ingeniería de Datos, para pasar luego a la sección de Machine Learning, finalizando en un "Deploy" del modelo de recomendaciones.

![Diagrama de Trabajo](diagrama.png)

## Ingeniería de Datos (ETL)

### Ingesta de Datos (Extracción)

Los datos se encuentran en formato json "impuro", ya que estamos en presencia de json anidados y/o que no respetan la estructura tradicional de datos dentro de un formato json.
A pesar de lo expuesto, utilizando las librerías de pandas y json en Python, y realizando algunas iteraciones, se pueden extraer y desanidar los datos, transformarlos en un Dataframe y finalmente exportarlos a un archivo csv.
Los datasets se encuentran en el siguiente enlace: [Link a los Datos.](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj)

### Tratamiento de Datos (Transformación)

En esta etapa se realizan las transformaciones que se consideren indispensables para preparar y optimizar el rendimiento de los modelos de aprendizaje automático, y optimizar el rendimiento de la API.
Entre algunas de las transformaciones realizadas (se pueden ver todas las transformaciones en los archivos ipynb correspondientes), se pueden mencionar la eliminación de filas donde todos sus elementos son nulos, la revisión y corrección del tipo de datos, la eliminación de duplicados y la eliminación de columnas irrelevantes para el análisis. No se revisan en esta instancia, valores nulos, distribuciones ni outliers, que serán revisados en el EDA (Exploratory Data Analysis).

**Feature Engineering:** Una de las transformaciones particulares que se realiza, es el agregado de la columna *sentiment_analysis* aplicando análisis de sentimiento con NLP la librería TextBlobcon la siguiente escala respecto al review: valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna reemplaza la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. Si no hay reseña escrita, toma el valor de '1'.

### Disponibilización de Datos (Carga/Load)

Podemos abordar la disponibilización desde dos perspectivas. Por un lado, desde la óptica del uso de los datos para los modelos de aprendizaje, la carga está implícita en el proceso, ya que los archivos csv generados son utilizados para la etapa siguiente.
Una de las etapas relevantes del proyecto es la **disponibilización de datos a través de una API**. Este es el segundo enfoque y para conseguirlo se ofrecerán los datos de la empresa usando el framework FastAPI, y realizando un deploy mediante el servicio de render ([Link a la web de Render](https://henry-pi1-steam.onrender.com/docs)). En este caso, y debido a las limitaciones de tamaño para implementar el servicio, los Dataframes vuelven a recibir un proceso de transformación buscando disponibilizar solo la información necesaria para las consultas, y no todo el Dataset.
Los datos se encuentran disponibles desde el siguiente enlace:

[Link al deploy en Render](https://henry-pi1-steam.onrender.com/docs)

Las consultas propuestas son las siguientes:

  1) def PlayTimeGenre( genero : str ): Debe devolver año con mas horas jugadas para dicho género.
  Ejemplo de retorno: {'El año con más horas jugadas para el género X es el 2012'}
  
  2) def UserForGenre( genero : str ): Debe devolver el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año.
  Ejemplo de retorno: {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas:   23}]}
  
  3) def UsersRecommend( año : int ): Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. (reviews.recommend = True y comentarios positivos/neutrales)
  Ejemplo de retorno: [{'Puesto 1': 'X', 'Puesto 2': 'Y', 'Puesto 3':'Z'}]
  
  4) def UsersNotRecommend( año : int ): Devuelve el top 3 de juegos MENOS recomendados por usuarios para el año dado. (reviews.recommend = False y comentarios negativos)
  Ejemplo de retorno: [{'Puesto 1': 'X', 'Puesto 2': 'Y', 'Puesto 3':'Z'}]
  
  5) def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento. Ejemplo de retorno: [{'release_date': 2000, 'Negativo': 18, 'Neutro': 32, 'Positivo': 49}]

<br> 
<br> 
<br> 

# MACHINE LEARNING

## Análisis exploratorio de los datos: (Exploratory Data Analysis-EDA)

A grandes rasgos, se tiene un dataset con videojuegos de la platafoma Steam Games con descripción sobre estos como el género o etiquetas en general y el precio, otro dataframe con los usuarios registrados y las horas jugadas totales para cada juego y por último los reviews de estos usuarios hacia cada juego. 

Hay 32133  juegos y 58459 usuarios, de los cuales 25485 usuarios,  dejaron 58431 comentarios sobre los juegos. 

Luego de este analisis general se realiza un reconocimiento de las variables más importantes para la recomendación, se categorizan algunas de ellas y se muestran en gráficos. En este punto se debe aclarar que debido a ser este un MVP, se redujeron las variables a utilizar en el ML a lo mínimo necesario que requiere la libreria surprice en sus propuestas básicas, estas son: user_id, item_id y sentimente, que es la "puntuación" del usuario hacia determinado juego, que se obtiene del análisis de sentimiento y que tiene tres opciones. Puntuación negativa, nula o positiva.

## Modelo de aprendizaje automático:

Para armar un sistema de recomendación, se procede a entrenar el modelo de machine learning, para ello se utiliza la librería surprice, ingestando los datos como un dataframe y utlizando el método TrainTestSplit. Puede ver la documentación oficial aquí: https://surprise.readthedocs.io/en/stable/getting_started.html#train-test-split-and-the-fit-method

El modelo entrenado se guardo con la librería joblib, obteniendo un archivo .pkl que será consumido por la función de recomendación. 

Para probar el sistema de recomendación, se puede abrir el documento df_ML.csv provisto en Github y elegir un usuario o probar con los siguientes usuarios:

76561197970982479
evcentric
js41637
ApxLGhost
72947282842
wayfeng
pikawuu2
Sammeh00
Michaelele

## Video 
En el enlace puede ver un video de la API en Render y su funcionamieto

https://youtu.be/

## requirements.txt
En este archivo de texto se colocan las librías que son necesarias para que la API funcione correctamente. 

## .gitignore
Permite que Render ignore los archivos que se contienen aquí pero que puedan permanecer en el repositorio.  

