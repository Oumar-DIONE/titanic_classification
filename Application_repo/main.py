import argparse
# Importation des objets définis dans __all__
from titanicml import retrieve_data, split_data, fit_rmfr, assess_rdmf


parser = argparse.ArgumentParser(description="les arguments optionnels en CMD")
parser.add_argument("--n_trees", type=int, default=20, help="Nombre d'arbre")
args = parser.parse_args()
MAX_DEPTH = None
MAX_FEATURES = "sqrt"
print("le nombre d'arbre vaut :", args.n_trees)


TrainingData = retrieve_data()
# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")
X_train, y_train, X_test, y_test = split_data(X, y, test_size_=0.1)
model = fit_rmfr(X_train, y_train, n_treesa=20)
assess_rdmf(X_train, y_train, X_test, y_test, model)
