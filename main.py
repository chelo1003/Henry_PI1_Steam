from fastapi import FastAPI
import pandas as pd

# Importar los Datos
df_PlayTimeGenre = pd.read_csv("funciones/PlayTimeGenre.csv", low_memory=False)
df_UserForGenre_1 = pd.read_csv("funciones/UserForGenre_1.csv", low_memory=False)
df_UserForGenre_2 = pd.read_csv("funciones/UserForGenre_2.csv", low_memory=False)
df_UsersRecommend = pd.read_csv("funciones/UsersRecommend.csv", low_memory=False)
df_UsersNotRecommend = pd.read_csv("funciones/UsersNotRecommend.csv", low_memory=False)
df_sentiment_analysis = pd.read_csv("funciones/sentiment_analysis.csv", low_memory=False)

app = FastAPI()


@app.get("/PlayTimeGenre/{genre}", name="Ingresar un género para obtener el año con más horas jugadas:")

def PlayTimeGenre(genre: str = None):
    
    genre = genre.capitalize()
    
    # Filtrar el Dataframe por el género recibido
    genero_elegido = df_PlayTimeGenre[df_PlayTimeGenre['genre'] == genre]

    # Evitar errores si recibo un valor desconocido
    if genero_elegido.empty:
        return {"No hay información para el género": genre}
    
    # Obtener el año buscado
    anio = genero_elegido['release_date'].values[0]

    return {"El año con más horas jugadas para el género " + genre + " es el " + str(anio)}


@app.get("/UserForGenre/{genre}", name="Ingresar un género para obtener el usuario con más tiempo jugado para ese género, y el tiempo jugado para cada año:")

def UserForGenre(genre: str = None):

    genre = genre.capitalize()
    
    # Filtrar el Dataframe por el género recibido
    genero_elegido = df_UserForGenre_1[df_UserForGenre_1['genre'] == genre]

    # Evitar errores si recibo un valor desconocido
    if genero_elegido.empty:
        return {"No hay información para el género": genre}

    # Obtener el usuario buscado
    usuario = genero_elegido['user_id'].values[0]

    # Filtrar el segundo Dataframe por el usuario Top
    usuario_top = df_UserForGenre_2[df_UserForGenre_2['user_id'] == usuario]

    # Llenar el diccionario de horas jugadas por año
    lista_horas_jugadas = [
    {"Año": anio, "minutos": tiempo} for anio, tiempo in zip(usuario_top['posted'], usuario_top['playtime_forever'])
    ]

    return {
        "Usuario": "El usuario con más horas jugadas para el género " + genre + " es " + usuario,
        "Horas jugadas": lista_horas_jugadas
    }


@app.get("/UsersRecommend/{posted}", name="Ingresar un año para saber cuales fueron los 3 juegos más recomendados ese año:")

def UsersRecommend(posted: int):

    # Filtrar el Dataframe por el año recibido
    df_anio = df_UsersRecommend[df_UsersRecommend['posted'] == posted]

    # Crear un diccionario con los 3 primeros puestos
    top_3_dict = {}
    for i, row in df_anio.iterrows():
        puesto = "Puesto " + str(i + 1 - df_anio.index[0])
        top_3_dict[puesto] = row['app_name']

    return top_3_dict


@app.get("/UsersNotRecommend/{posted}", name="Ingresar un año para saber cuales fueron los 3 juegos menos recomendados ese año:")
    
def UsersNotRecommend(posted: int):

    # Filtrar el Dataframe por el año recibido
    df_anio = df_UsersNotRecommend[df_UsersNotRecommend['posted'] == posted]

    # Crear un diccionario con los 3 primeros puestos
    top_3_dict = {}
    for i, row in df_anio.iterrows():
        puesto = "Puesto " + str(i + 1 - df_anio.index[0])
        top_3_dict[puesto] = row['app_name']

    return top_3_dict


@app.get("/Sentiment_Analysis/{release_year}", name="Ingresar un año para obtener el recuento re reseñas negativas, neutras y positivas de ese año (release_date):")

def Sentiment_Analysis(year: int):
    # Filtrar el Dataframe por el año recibido
    df_anio = df_sentiment_analysis[df_sentiment_analysis['release_date'] == year]
    
    # Crear un diccionario con los datos recibidos
    filas_lista = df_anio.to_dict(orient='records')
    
    return filas_lista