import os
import logging
from dotenv import load_dotenv
import boto3
import json

load_dotenv()

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


def upload_pdfs_to_s3(local_filepaths, s3_bucket, s3_input_folder, aws_access_key, aws_secret_key, aws_region):

    s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
    s3_key_list = []

    for filepath in local_filepaths:
        filename = os.path.basename(filepath)
        s3_key = f"{s3_input_folder.rstrip('/')}/{filename}"
        try:
            s3.upload_file(filepath,s3_bucket, s3_key)
            logger.info(f"Successfully uploaded {filename} to s3://{s3_bucket}/{s3_key}")
            s3_key_list.append(s3_key)
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
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