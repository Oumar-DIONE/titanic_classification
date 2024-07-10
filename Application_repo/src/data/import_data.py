import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Obtenir le chemin absolu du répertoire courant (où build_features.py est situé)
current_dir = os.path.dirname(__file__)   # .../src/data
# Obtenir le chemin du parent du répertoire courant
src = os.path.dirname(current_dir)  # ...Application_repo/src/
Application_repo = os.path.dirname(src)  # .../Application_repo/

# Construire le chemin vers les répertoire 'data' et 'configuration' 
data_dir = os.path.abspath(os.path.join(Application_repo, 'data'))
config_dir = os.path.abspath(os.path.join(Application_repo, 'configuration'))
# Ajouter 'data_dir' à 'sys.path' pour permettre l'importation des modules depuis 'src/data'
sys.path.insert(0, config_dir)
sys.path.insert(0, data_dir)
import import_config  


def retrieve_data(config_file="config.yaml"):
    config = import_config.import_yaml_config(config_file)
    test_fraction = float(config.get("test_fraction"))
    data_path = config["data_path"]
    print("test_fraction :", test_fraction)

# lecture du fichier
    os.chdir(data_path)
    Trainingdata = pd.read_csv("raw/data.csv")
    Trainingdata.head()
    Trainingdata["Ticket"].str.split("/").str.len()
    Trainingdata["Name"].str.split(",").str.len()
    Trainingdata.isnull().sum()
    return Trainingdata


# Un peu d'exploration et de feature engineering
TrainingData = retrieve_data()
# Statut socioéconomique
# layout matplotlib 1 ligne 2 colonnes taile 16*8
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
VAR1 = "fréquence des Pclass"
VAR2 = "survie des Pclass"
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(VAR1)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title(VAR2)


# Age
VAR3 = "Distribution de l'âge"
sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(VAR3)
plt.show()
print("All is well done")
