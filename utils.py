import os
import logging
from dotenv import load_dotenv
import boto3

load_dotenv()

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
s3_input_folder = os.getenv('S3_INPUT_DATA_FOLDER')
aws_region = os.getenv('AWS_REGION')

if not all([s3_bucket, s3_input_folder, aws_region]):
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

    s3 = boto3.client('s3')

    for filename in os.listdir(local_folder):
        if filename.lower().endswith('.pdf'):
            local_path = os.path.join(local_folder, filename)
            s3_key = f"{s3_input_folder.rstrip('/')}/{filename}"

            try:
                s3.upload_file(local_path,s3_bucket, s3_key)
                logger.info(f"Successfully uploaded {filename} to s3://{s3_bucket}/{s3_key}")
            except FileNotFoundError:
                logger.error(f"File not found: {local_path}")
            except Exception as e:
                logger.error(f"Failed to upload {filename}: {e}")

#upload_pdfs_to_s3(local_folder= 'data', s3_bucket=s3_bucket, s3_input_folder=s3_input_folder)