import os
import boto3
import import_config


def initialize_minio(endpoint_url_, aws_access_key_id_, aws_secret_access_key_, aws_session_token_):
    s3 = boto3.client("s3", endpoint_url=endpoint_url_, aws_access_key_id=aws_access_key_id_, aws_secret_access_key=aws_secret_access_key_, aws_session_token=aws_session_token_)
    # Tester la connexion
    try:
        response = s3.list_buckets()
        print("Buckets:")
        for bucket in response['Buckets']:
            print(f"  {bucket['Name']}")
    except Exception as e:
        print(f"Error: {e}")
    return s3


config = import_config.import_yaml_config()
endpoint_url = config["endpoint_url"]
aws_access_key_id = config["aws_access_key_id"]
aws_secret_access_key = config["aws_secret_access_key"]
aws_session_token = config["aws_session_token"]

my_s3 = initialize_minio(endpoint_url, aws_access_key_id, aws_secret_access_key, aws_session_token)


def list_objects(client_minio=my_s3, bucket_name="odione"):
    # Lister les objets dans le bucket
    try:
        response = client_minio.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            print("Objects in bucket:")
            for obj in response['Contents']:
                print(f"  {obj['Key']}")
        else:
            print("Bucket is empty or does not exist.")
    except Exception as e:
        print(f"Error: {e}")


def dowlnload_from_s3(remote_path, local_path, s3=my_s3, bucket_name="odione"):
    # Télécharger un fichier
# Vérifier si le fichier local existe, sinon créer le répertoire
    if not os.path.exists(os.path.dirname(local_path)):
        try:
            os.makedirs(os.path.dirname(local_path))
            print(f"Created directory {os.path.dirname(local_path)}")
        except OSError as exc:
            print(f"Error creating directory: {exc}")

# Télécharger le fichier
    try:
        s3.download_file(bucket_name, remote_path, local_path)
        print("Download successful.")
    except Exception as e:
        print(f"Error: {e}")


def updload_towards_s3(remote_path, local_path, s3=my_s3, bucket_name="odione"):
    # Vérifier si le fichier local existe, sinon le créer
    if not os.path.exists(local_path):
        try:
            with open(local_path, 'w') as f:
                pass  # Créer un fichier vide
            print(f"Created file {f}")
        except OSError as exc:
            print(f"Error creating file: {exc}")

# Uploader le fichier
    try:
        s3.upload_file(local_path, bucket_name, remote_path)
        print("Upload successful.")
    except Exception as e:
        print(f"Error: {e}")