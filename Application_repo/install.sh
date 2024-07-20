# !/bin/bash
# Install Python
apt-get -y update
apt-get install -y python3-pip python3-venv
#create empty virtual environnemnt
python3 -m venv titanic_env
source titanic_env/bin/activate
#install projec dependencis
pip install -r requirements.txt
