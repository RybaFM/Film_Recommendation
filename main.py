from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from fastapi.staticfiles import StaticFiles
import pandas as pd


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class FormData(BaseModel):
    age: int
    gender: int
    genres: list[int]

model = joblib.load("movie_model.pkl")
df_films = pd.read_csv('films.csv')

def make_recommendation(user_info, mod, df_films):
  film_id = mod.predict(np.array([user_info]))
  return (int(film_id[0]),
          df_films[df_films['movie_id'] == film_id[0]].iloc[0, 1])

@app.post("/recommend")
def recommend(data: FormData):
    input_features = [data.age, data.gender] + data.genres
    
    prediction = make_recommendation(input_features, model, df_films)
    
    return {"recommended_movie": f"({int(prediction[0])}) {prediction[1]}"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")
#python -m uvicorn main:app --reload
#cd "D:\Backup\My personal attemps in graphic design and coding\PythonLearning\Film_Recommendation"
