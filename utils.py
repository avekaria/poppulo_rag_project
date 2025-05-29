import os
import logging
from dotenv import load_dotenv
import boto3
import json

load_dotenv()

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
s3_input_folder = os.getenv('S3_INPUT_DATA_FOLDER')
aws_region = os.getenv('AWS_REGION')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')


if not all([s3_bucket, s3_input_folder, aws_region, aws_access_key, aws_secret_key]):
    raise EnvironmentError("AWS environment variables not set")


# Configure logging for every module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers = [
        logging.FileHandler("app.log"),  # Writes the logs to a file
        logging.StreamHandler()          # Shows logs in console
    ]
)
logger = logging.getLogger(__name__)


def upload_pdfs_to_s3(local_folder, s3_bucket, s3_input_folder):
    """ local_folder=/path/to/local/pdf_files
    s3_bucket and s3_input_folder are defined as environment variables """

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
    s3_key_list = []

    for filename in os.listdir(local_folder):
        if filename.lower().endswith('.pdf'):
            local_path = os.path.join(local_folder, filename)
            s3_key = f"{s3_input_folder.rstrip('/')}/{filename}"

            try:
                s3.upload_file(local_path,s3_bucket, s3_key)
                logger.info(f"Successfully uploaded {filename} to s3://{s3_bucket}/{s3_key}")
                s3_key_list.append(s3_key)
            except FileNotFoundError:
                logger.error(f"File not found: {local_path}")
            except Exception as e:
                logger.error(f"Failed to upload {filename}: {e}")

    return s3_key_list


def save_file(json_body, json_filepath):
    with open(json_filepath, 'w') as fw:
        json.dump(json_body, fw, indent=4, ensure_ascii=True)
    return json_filepath


def load_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


# s3_files_key = upload_pdfs_to_s3(local_folder= 'data', s3_bucket=s3_bucket, s3_input_folder=s3_input_folder)
# print(s3_files_key)