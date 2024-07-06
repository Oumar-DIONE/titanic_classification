import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import argparse


os.chdir("/home/onyxia/work/titanic_classification")
TrainingData = pd.read_csv("data.csv")

TrainingData.head()


TrainingData["Ticket"].str.split("/").str.len()

TrainingData["Name"].str.split(",").str.len()

parser = argparse.ArgumentParser(description="les arguments optionnels en CMD")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbre")
args = parser.parse_args()

n_trees = args.n_trees
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
print("le nombre d'arbre vaut :", args.n_trees)
TrainingData.isnull().sum()


# Un peu d'exploration et de feature engineering

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

# Encoder les données imputées ou transformées.

numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

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


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie
# pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([X_train, y_train]).to_csv("train.csv")
pd.concat([X_test, y_test]).to_csv("test.csv")

JETONAPI = "$trotskitueleski1917"


# Random Forest


# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)

# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# #(10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")
print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))
