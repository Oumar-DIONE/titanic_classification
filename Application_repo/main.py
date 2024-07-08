import os
import sys
import argparse
# Importer le module depuis src/data
# Obtenir le chemin absolu du répertoire courant (où build_features.py est situé)
current_dir = os.path.dirname(__file__)   # .../src/features
# Obtenir le chemin du parent du répertoire courant
src = os.path.abspath(os.path.join(current_dir, 'src'))  # .../src/
# Construire le chemin vers le sous-répertoire 'data' du parent
data_dir = os.path.abspath(os.path.join(src, 'data'))
features_dir = os.path.abspath(os.path.join(src, 'features'))
models_dir = os.path.abspath(os.path.join(src, 'models'))
# Ajouter 'data_dir' à 'sys.path' pour permettre l'importation des modules depuis 'src/data'
sys.path.insert(0, data_dir)
sys.path.insert(0, features_dir)
sys.path.insert(0, models_dir)
import import_data
import build_features
import train_evaluate
parser = argparse.ArgumentParser(description="les arguments optionnels en CMD")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbre")
args = parser.parse_args()
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
print("le nombre d'arbre vaut :", args.n_trees)


TrainingData = import_data.retrieve_data()
# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")
X_train, y_train, X_test, y_test = build_features.split_data(X, y, test_size_=0.1)
model = build_features.fit_rmfr(X_train, y_train, n_treesa=20)
train_evaluate.assess_rdmf(X_train, y_train, X_test, y_test, model)
