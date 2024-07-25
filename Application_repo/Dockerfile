# Utiliser une image de base officielle Ubuntu 22.04
FROM ubuntu:22.04

# Définir le répertoire de travail dans le conteneur
WORKDIR ${HOME}/titanic_classification/Application_repo

# Installer Python et pip
RUN apt-get -y update && \
    apt-get install -y python3-pip

# Copier le fichier requirements.txt dans le répertoire de travail du conteneur
COPY requirements.txt .

# Installer les dépendances du projet
RUN pip install -r requirements.txt

# Copier le script principal dans le répertoire de travail
COPY docs/main.py ./docs/

# Copier le répertoire src dans le répertoire de travail
COPY titanicml ./titanicml

# Définir la commande par défaut pour exécuter votre application
CMD ["python3", "docs/main.py"]
