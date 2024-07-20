import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from . import upload_dowload_from_Minio

def retrieve_data(config_file="config.yaml", filename_="titanic_data.csv"):
    config = upload_dowload_from_Minio.import_yaml_config()
    print("Configuration pour retrieve_data :", config)
    test_fraction = float(config.get("test_fraction"))
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du script actuel
    data_path_default_value = os.path.join(script_dir, '../../data/')  # Chemin relatif vers config.yaml
    
    data_path = config.get("data_path", data_path_default_value)  # Ajouter une valeur par défaut pour data_path
    print("test_fraction :", test_fraction)

    # lecture du fichier
    os.chdir(data_path)
    # import data from S3
    local_path_ = os.path.join(data_path, filename_)
    s3_file = "Titanic_Data/data.csv"
    upload_dowload_from_Minio.download_from_s3(remote_path=s3_file, local_path=local_path_)
    Trainingdata = pd.read_csv(filename_)
    Trainingdata.head()
    Trainingdata["Ticket"].str.split("/").str.len()
    Trainingdata["Name"].str.split(",").str.len()
    Trainingdata.isnull().sum()

    return Trainingdata

"""
# Un peu d'exploration et de feature engineering
TrainingData = retrieve_data()
# Statut socioéconomique
# layout matplotlib 1 ligne 2 colonnes taille 16*8
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
VAR1 = "fréquence des Pclass"
VAR2 = "survie des Pclass"
sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(VAR1)
sns.barplot(data=TrainingData, x="Pclass", y="Survived", ax=axes[1]).set_title(VAR2)

# Age
VAR3 = "Distribution de l'âge"
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(VAR3)
plt.show()
print("All is well done")

"""
