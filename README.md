# Proyecto Steam - Recomendación de Juegos

Este proyecto consiste en crear un MVP (Minimum Viable Product) que recomiende videojuegos por similitud, consumible mediante una API REST o RESTful, implementado con el famework FastAPI y deployado en Render. 
Para el análisis de sentimiento se utiliza la libreria TextBlob y para el entrenamiento del modelo de recomendación se utiliza el método de Similitud del Coseno de la librería Scikit Learn.
Los datos provienen de [STEAM](https://store.steampowered.com/), que es una plataforma de distribución digital de videojuegos, para Windows y Unix.

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

Los datos fueron recolectados entre los años 2010 y 2015. La presentación de los datos en "bruto" es a través de tres archivos json.

- "steam_games" con información sobre de los juegos, como género, desarrollador, precio, etc. En este dataset hay aproximadamente 32000 juegos.  
- "user_reviews" que contiene reseñas y recomendaciones de juegos por parte de los usuarios. En este dataset hay aproximadamente 25500 usuarios que dejaron unas 60000 reseñas.
- "user_items" que contiene el tiempo jugado por cada usuario a cada juego. En este dataset hay datos de tiempo jugado de unos 70000 usuarios a unos 11000 juegos diferentes.

Como criterio general para todos los datasets, en el EDA se revisaron los valores nulos y se decidió si rellenarlos o eliminarlos, se borraron registros duplicados, se hicieron estudios de distribución para variables categóricas, se revisaron distribuciones mediante boxplots para variables cuantitativas contínuas, etc. Los datasets se dejaron procesados para la etapa de Machine Learning, con la posibilidad de hacer cambios en esta última etapa para que puedan adaptarse y ser consumidos por los modelos, una vez seleccionados.

## Modelo de aprendizaje automático

Para la etapa de modelos de Machine Learning, se ofrecen dos alternativas, armar un sistema de recomendación de juegos para un usuario dado (user-item), o hacerlo a partir de un juego dado (item-item). La elección es libre, y en este caso se elige item-item. El modelo elegido para la recomendación es similitud de coseno de la librería Scikit Learn. [En este enlace se puede revisar la documentación.](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html#sklearn.metrics.pairwise.cosine_similarity)

A partir de los datasets obtenidos en el EDA, se adaptan las variables para que puedan ser consumidas por el modelo y se eliminan columnas innecesarias. En el cuaderno ML_item-item.ipynb se pueden revisar las transformaciones realizadas. Finalmente se realiza la unión de todos los Dataframes, y se elimina la columna de rótulos para pasar los datos numéricos al modelo. Se analiza la similitud de 2834 juegos entre sí.

A la matriz resultante se le agregan nuevamente los id de juegos como índice y los nombres de juegos como nombres de columnas, y se sube al repositorio para que sea levantada en el deploy.

La consulta propuesta para la sección de machine learning es la siguiente:

  6) def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

## Video 
En siguiente es un enlace a una carpeta de Google Drive donde está el video explicativo del proyecto. [Enlace a la carpeta.](https://drive.google.com/drive/folders/1RZIILeALXjE1Zc-VDHFdESgqYUjR1Hq2)
