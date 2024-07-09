import os
import yaml


def import_yaml_config(config_path="config.yaml"):
    os.chdir("/home/onyxia/work/titanic_classification/Application_repo/configuration/")
    config_ = {}
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            try:
                config_ = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Erreur lors de la lecture du fichier YAML: {exc}")
    return config_