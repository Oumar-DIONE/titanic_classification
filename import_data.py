import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def import_yaml_config(config_path):
    config_ = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            try:
                config_ = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Erreur lors de la lecture du fichier YAML: {exc}")
    return config_
    
    
def retrieve_data(config_file="config.yaml"):
    config = import_yaml_config(config_file)
    TEST_FRACTION = float(config.get("test_fraction"))
    data_path = config["data_path"]
    print(TEST_FRACTION)

# lecture du fichier
    os.chdir(data_path)
    Trainingdata = pd.read_csv("data.csv")
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
