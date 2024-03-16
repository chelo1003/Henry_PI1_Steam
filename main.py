from fastapi import FastAPI
import pandas as pd
import pyarrow.parquet as pq

# Importar los Datos
df_PlayTimeGenre = pd.read_csv("funciones/PlayTimeGenre.csv", low_memory=False)
df_UserForGenre_2 = pd.read_csv("funciones/UserForGenre_2.csv", low_memory=False)
df_UsersRecommend = pd.read_csv("funciones/UsersRecommend.csv", low_memory=False)
df_UsersNotRecommend = pd.read_csv("funciones/UsersNotRecommend.csv", low_memory=False)
df_sentiment_analysis = pd.read_csv("funciones/sentiment_analysis.csv", low_memory=False)
df_matriz_similitudes = pd.read_parquet("ML/df_matriz_similitudes.parquet")

app = FastAPI()


@app.get("/PlayTimeGenre/{genre}", name="Ingresar un género para obtener el año con más horas jugadas:")

def PlayTimeGenre(genre: str = None):

    # Devolver una advertencia si no se introduce un género
    if genre == "":
        return {"Si no sabe que buscar, no busque nada."}
    
    # Filtrar el Dataframe por el género recibido
    genero_elegido = df_PlayTimeGenre[df_PlayTimeGenre['genre'].str.contains(genre, case=False)]

    # Evitar errores si recibo un valor desconocido
    if genero_elegido.empty:
        return {"No hay información para el género": genre}
    
    # Obtener el año buscado
    anio = genero_elegido['release_date'].values[0]

    return {"El año con más horas jugadas para el género " + genre + " es el " + str(anio)}


@app.get("/UserForGenre/{genre}", name="Ingresar un género para obtener el usuario con más tiempo jugado para ese género, y el tiempo jugado para cada año:")

def UserForGenre(genre: str = None):

    # Devolver una advertencia si no se introduce un género
    if genre == "":
        return {"Si no sabe que buscar, no busque nada."}
    
    # Filtrar el Dataframe por el género recibido
    genero_elegido = df_UserForGenre_2[df_UserForGenre_2['genre'].str.contains(genre, case=False)]

    # Evitar errores si recibo un valor desconocido
    if genero_elegido.empty:
        return {"No hay información para el género": genre}

    # Obtener el usuario buscado
    usuario = genero_elegido['user_id'].values[0]

    # Llenar el diccionario de horas jugadas por año
    lista_horas_jugadas = [
    {"Año": anio, "minutos": tiempo} for anio, tiempo in zip(genero_elegido['posted'], genero_elegido['playtime_forever'])
    ]

    return {
        "Usuario": "El usuario con más horas jugadas para el género " + genre + " es " + usuario,
        "Horas jugadas": lista_horas_jugadas
    }


@app.get("/UsersRecommend/{posted}", name="Ingresar un año para saber cuales fueron los 3 juegos más recomendados ese año:")

def UsersRecommend(posted: int):

    # Verificar que se ingresa un entero
    if type(posted) != int:
        return ("Ingrese un número.")
    
    # Buscar las filas que coinciden con el año ingresado
    df_anio = df_UsersRecommend[df_UsersRecommend['posted'] == posted]

    # Verificar si hay datos para el año ingresado
    if df_anio.empty:
        return ("No hay datos para el año indicado.")

    # Construir la respuesta
    top_3_dict = {}
    for i, row in df_anio.iterrows():
        puesto = "Puesto " + str(i + 1 - df_anio.index[0])
        top_3_dict[puesto] = row['app_name']

    return top_3_dict


@app.get("/UsersNotRecommend/{posted}", name="Ingresar un año para saber cuales fueron los 3 juegos menos recomendados ese año:")
    
def UsersNotRecommend(posted: int):
    
    # Verificar que se ingresa un entero
    if type(posted) != int:
        return ("Ingrese un número.")
    
    # Buscar las filas que coinciden con el año ingresado
    df_anio = df_UsersNotRecommend[df_UsersNotRecommend['posted'] == posted]

    # Verificar si hay datos para el año ingresado
    if df_anio.empty:
        return ("No hay datos para el año indicado.")
    
    # Construir la respuesta
    top_3_dict = {}
    for i, row in df_anio.iterrows():
        puesto = "Puesto " + str(i + 1 - df_anio.index[0])
        top_3_dict[puesto] = row['app_name']

    return top_3_dict


@app.get("/Sentiment_Analysis/{release_year}", name="Ingresar un año para obtener el recuento re reseñas negativas, neutras y positivas de ese año (release_date):")

def Sentiment_Analysis(year: int):

    # Verificar que se ingresa un entero
    if type(year) != int:
        return ("Ingrese un número.")
    
    # Filtrar el DataFrame para obtener las filas que coinciden con el release_date dado
    df_anio = df_sentiment_analysis[df_sentiment_analysis['release_date'] == year]
 
    # Verificar si hay datos para el año ingresado
    if df_anio.empty:
        return ("No hay datos para el año indicado.")
       
    # Convertir las filas coincidentes a una lista de diccionarios
    filas_lista = df_anio.to_dict(orient='records')
    
    return filas_lista


@app.get("/recomendacion_juego/{item_id}", name="Ingresar el id de un juego para que recomiende los 5 juegos que más se le parecen. Ej: 10")

def recomendacion_juego(item_id: int):
    N = 5

    # Reviso si el id ingresado es un entero
    if type(item_id) != int:
        return ("Ingrese un id de juego.")  

    # Reviso si el id ingresado está en la lista de juegos
    if item_id not in df_matriz_similitudes.index:
        return ("No se encuentra un juego con el id ingresado.")
    
    # Supongamos que 'df_matriz_similitudes' es tu matriz de similitud y 'juego_referencia' es el id del juego que quieres comparar
    juego_referencia = item_id

    # Encuentra el índice del juego de referencia
    # indice_juego_referencia = df_matriz_similitudes.index.get_loc(juego_referencia)

    # Extrae la fila de similitud para el juego de referencia
    fila_similitud = df_matriz_similitudes.iloc[juego_referencia]

    # Ordena los valores de similitud en orden descendente
    juegos_similares = fila_similitud.sort_values(ascending=False)

    juego_seleccionado = juegos_similares[0:1]
    juegos_recomendados = juegos_similares[1:N+1]  # Excluye el juego de referencia
    

    return {
        "El juego ":juego_seleccionado.index[0], 
        "es similar a estos 5 juegos": juegos_recomendados.index.to_list()       
    }