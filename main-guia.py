'''main.py'''

import pandas as pd
from fastapi import FastAPI
import uvicorn 
import joblib

app = FastAPI()


users_recommend = pd.read_csv('Funcion_3_jup.csv', low_memory=False)
users_not_recommend = pd.read_csv('Funcion_4_jup.csv', low_memory=False)
sentiment_analysis_5 = pd.read_csv('Funcion_5_jup.csv', low_memory=False)
df = pd.read_csv('df_ML.csv', low_memory=False)
modelo_svd = joblib.load('modelo_svd_I.pkl')






@app.get("/recomendacion_usuario/{user_id}", name='devuelve lista con 5 juegos recomendados para dicho usuario.')


def recomendacion_usuario(user_id: object):
    top=5

    # Obtener los juegos que el usuario no ha visto aún
    games_play = set(df[df['user_id'] == user_id]['item_id'])
    games_all = set(df['item_id'])
    games_unplay = list(games_all - games_play)

    # Obtener las recomendaciones
    predicted_ratings = [modelo_svd.predict(user_id, item_id).est for item_id in games_unplay]

    # Ordenar los juegos según su predicción de rating
    games_rating = list(zip(games_unplay, predicted_ratings))
    games_rating.sort(key=lambda x: x[1], reverse=True)

    # Obtener los títulos de los juegos recomendadas
    recommended_games = games_rating[:top]
    recommended_games = [df[df['item_id'] == item_id]['app_name'].iloc[0] for item_id, _ in recommended_games]

    return {
        "A usuarios que son similares a":user_id, 
        "también les gustó estos 5 juegos": recommended_games,
         
    }
