from sklearn.metrics import confusion_matrix
# On _split_ notre _dataset_ d'apprentisage pour faire de la validation croisée une partie pour apprendre une partie
# pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
# calculons le score sur le dataset d'apprentissage et sur le dataset de test
# #(10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction


def assess_rdmf(x_train, y_train, x_test, y_test, model):
    rdmf_score_test = model.score(x_test, y_test)
    rdmf_score_tr = model.score(x_train, y_train)
    print(f"{rdmf_score_tr:.1%} de bonnes réponses sur les données de Train pour validation")
    print(f"{rdmf_score_test:.1%} de bonnes réponses sur les données de test pour validation")
    print("matrice de confusion")
    print(confusion_matrix(y_test, model.predict(x_test)))
