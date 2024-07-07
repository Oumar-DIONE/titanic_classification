import argparse
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
