import os
import sys
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Importer le module import_config depuis Application_repo/configuration
# Obtenir le chemin absolu du répertoire courant (où build_features.py est situé)
current_dir = os.path.dirname(__file__)   # .../Application_repo/test
# Obtenir le chemin du parent du répertoire courant
Application_repo = os.path.abspath(os.path.join(current_dir, '..'))  # .../src/
# Construire le chemin vers le repertoire 'configuration' 
config_dir = os.path.abspath(os.path.join(Application_repo, 'configuration'))
# Ajouter 'data_dir' à 'sys.path' pour permettre l'importation des modules depuis 'src/data'
sys.path.insert(0, config_dir)

# Supposons que import_config est un module local qui contient la fonction import_yaml_config
import import_config


# La fonction à tester
def split_data(x, y, test_size_, train_path="train.csv", test_path="test.csv", config_file="config.yaml"):
    config = import_config.import_yaml_config(config_file)
    data_path = config["data_path"]
    os.chdir(data_path)
    train_path = "processed/" + train_path
    test_path = "processed/" + test_path
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_)
    pd.concat([x_train, y_train]).to_csv(train_path)
    pd.concat([x_test, y_test]).to_csv(test_path)
    print("split data well done")
    return x_train, y_train, x_test, y_test

class TestSplitData(unittest.TestCase):

    @patch('import_config.import_yaml_config')
    @patch('os.chdir')
    @patch('pandas.DataFrame.to_csv')
    def test_split_data(self, mock_to_csv, mock_chdir, mock_import_yaml_config):
        # Mock the config
        mock_import_yaml_config.return_value = {"data_path": "/mocked/path"}

        # Create dummy data
        x = pd.DataFrame(np.random.rand(100, 10))
        y = pd.DataFrame(np.random.rand(100, 1))
        
        # Call the function
        x_train, y_train, x_test, y_test = split_data(x, y, 0.2)
        
        # Check the split sizes
        self.assertEqual(len(x_train), 80)
        self.assertEqual(len(x_test), 20)
        self.assertEqual(len(y_train), 80)
        self.assertEqual(len(y_test), 20)
        
        # Ensure to_csv was called twice
        self.assertEqual(mock_to_csv.call_count, 2)

        # Check the filenames passed to to_csv
        mock_to_csv.assert_any_call('processed/train.csv')
        mock_to_csv.assert_any_call('processed/test.csv')

    @patch('import_config.import_yaml_config')
    @patch('os.chdir')
    @patch('pandas.DataFrame.to_csv')
    def test_split_data_file_creation(self, mock_to_csv, mock_chdir, mock_import_yaml_config):
        # Mock the config
        mock_import_yaml_config.return_value = {"data_path": "/mocked/path"}

        # Create dummy data
        x = pd.DataFrame(np.random.rand(100, 10))
        y = pd.DataFrame(np.random.rand(100, 1))
        
        # Mock the to_csv method to check the content written to the files
        mock_to_csv.side_effect = lambda path: self.assertIn(path, ['processed/train.csv', 'processed/test.csv'])
        
        # Call the function
        split_data(x, y, 0.2)
        
        # Ensure to_csv was called twice
        self.assertEqual(mock_to_csv.call_count, 2)



if __name__ == '__main__':
    unittest.main()


