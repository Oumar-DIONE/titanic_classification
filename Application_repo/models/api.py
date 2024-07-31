import os
from fastapi import FastAPI
from joblib import load
import pandas as pd

# Obtenir le chemin absolu du répertoire courant (où api.py est situé)
current_dir = os.path.dirname(__file__)
# Obtenir le chemin du parent du répertoire courant
Application_repo = os.path.abspath(os.path.join(current_dir, '..'))
# Construire le chemin vers le sous-répertoire 'docs' du parent
docs_dir = os.path.abspath(os.path.join(Application_repo, 'models'))
# Chemin du modèle
model_local_path = os.path.join(docs_dir, "classification_model.pkl")

# Charger le modèle
model = load(model_local_path)

# Créer l'application FastAPI
app = FastAPI(
    title="Prédiction de survie sur le Titanic",
    description="Application de prédiction de survie sur le Titanic 🚢 <br>Une version par API pour faciliter la réutilisation du modèle 🚀" +
                "<br><br><img src=\"https://media.vogue.fr/photos/5faac06d39c5194ff9752ec9/1:1/w_2404,h_2404,c_limit/076_CHL_126884.jpg\" width=\"200\">",
    docs_url="/swagger",  # Chemin pour Swagger UI
    redoc_url="/redoc"    # Chemin pour ReDoc
)

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

@app.get("/predict", tags=["Predict"])
async def predict(
    sex: str = "female",
    age: float = 29.0,
    fare: float = 16.5,
    embarked: str = "S"
) -> str:
    """
    Predict survival based on provided features.
    """
    df = pd.DataFrame(
        {
            "Sex": [sex],
            "Age": [age],
            "Fare": [fare],
            "Embarked": [embarked],
        }
    )

    prediction = "Survived 🎉" if int(model.predict(df)) == 1 else "Dead ⚰️"
    return prediction
