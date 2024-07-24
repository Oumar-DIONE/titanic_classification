FROM ubuntu:22.04
WORKDIR ${HOME}/titanic_classification/Application_repo
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python3", "main.py"]