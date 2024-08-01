from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import pandas as pd
import pickle

# Spécifiez le chemin vers votre fichier .pkl
model_path = 'classification_model.pkl'

# Ouvrir le fichier en mode lecture binaire et charger le modèle
with open(model_path, 'rb') as file:
    model = pickle.load(file)



app = FastAPI(
    title="Prédiction de survie sur le Titanic",
    description="Application de prédiction de survie sur le Titanic 🚢 <br>Une version par API pour faciliter la réutilisation du modèle 🚀"
                + "<br><br><img src=\"https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg\" width=\"200\">"
   ,version="0.1",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "Support",
        "url": "http://example.com/support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "http://example.com/license/",
    }
)
class PredictionInput(BaseModel):
    sex: str = "female"
    age: float = 29.0
    fare: float = 16.5
    embarked: str = "S"

class PredictionOutput(BaseModel):
    prediction: str

@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """
    return {
        "Message": "API de prédiction de survie sur le Titanic",
        "Model_name": 'Titanic ML',
        "Model_version": "0.1",
    }

@app.post("/predict", tags=["Predict"], response_model=PredictionOutput)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Prédit la survie sur le Titanic en fonction des caractéristiques fournies.

    - **sex**: Sexe du passager (`'male'` ou `'female'`)
    - **age**: Âge du passager (en années)
    - **fare**: Tarif payé par le passager (en livres)
    - **embarked**: Port d'embarquement (`'C'`, `'Q'` ou `'S'`)

    Retourne une réponse indiquant si le passager est survécu ou décédé.
    """
    df = pd.DataFrame(
        {
            "Sex": [input_data.sex],
            "Age": [input_data.age],
            "Fare": [input_data.fare],
            "Embarked": [input_data.embarked],
        }
    )

    prediction = "Survived 🎉" if int(model.predict(df)) == 1 else "Dead ⚰️"

    return PredictionOutput(prediction=prediction)
