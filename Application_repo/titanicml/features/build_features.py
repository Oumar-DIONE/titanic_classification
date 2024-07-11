import sys
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# Importer le module depuis src/data
# Obtenir le chemin absolu du répertoire courant (où build_features.py est situé)
current_dir = os.path.dirname(__file__)   # .../src/features
# Obtenir le chemin du parent du répertoire courant
src = os.path.abspath(os.path.join(current_dir, '..'))  # .../src/
Application_repo = os.path.dirname(src)  # .../Application_repo/
# Construire le chemin vers le repertoire 'configuration' 
# Construire le chemin vers le sous-répertoire 'data' du parent
data_dir = os.path.abspath(os.path.join(src, 'data')) # .../src/data
# Ajouter 'data_dir' à 'sys.path' pour permettre l'importation des modules depuis 'src/data'
sys.path.insert(0, data_dir)


import import_data


numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


def split_data(x, y, test_size_, train_path="train.csv", test_path="test.csv", config_file="config.yaml"):
    config = import_data.import_config.import_yaml_config(config_file)
    data_path = config["data_path"]
    os.chdir(data_path)
    train_path = "processed/" + train_path
    test_path = "processed/" + test_path
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_)
    pd.concat([x_train, y_train]).to_csv(train_path)
    pd.concat([x_test, y_test]).to_csv(test_path)
    print("split data well done")
    return x_train, y_train, x_test, y_test


MAX_DEPTH = 15
MAX_FEATURES = 10


# Random Forest
def fit_rmfr(x_train, y_train, n_treesa, numeric_features=["Age", "Fare"], categorical_features=["Embarked", "Sex"], max_depth=MAX_DEPTH, max_feature=MAX_FEATURES):
    # Encoder les données imputées ou transformées.
    preprocessor = ColumnTransformer(transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ])
    pipe_ = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=n_treesa)),
    ])
# Ici demandons d'avoir 20 arbres
    pipe_.fit(x_train, y_train)
    print("max_depth", max_depth)
    print("max_depth", max_feature)

    return pipe_ 