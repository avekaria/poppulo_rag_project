from typing import List, Dict
from dotenv import load_dotenv
import json
import logging
import boto3
import os
import glob
import time
import spacy

load_dotenv()

from utils import save_file, load_file

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

# Validate environment variables
required_vars = {
    'S3_BUCKET_NAME': s3_bucket,
    'AWS_REGION': aws_region,
    'AWS_ACCESS_KEY_ID': aws_access_key,
    'AWS_SECRET_ACCESS_KEY': aws_secret_key
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

# Configure logging for every module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Writes the logs to a file
        logging.StreamHandler()  # Shows logs in console
    ]
)
logger = logging.getLogger(__name__)

# Initialize Textract and Spacy clients
textract_client = boto3.client(service_name='textract', aws_access_key_id=aws_access_key,aws_secret_access_key=aws_secret_key,
                               region_name=aws_region)
nlp = spacy.load("en_core_web_sm")

def job_complete(job_id):
    global textract_client
    try:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response.get('JobStatus')
        logger.info(f"Textract job {job_id} returned status: {status}")

        return status in ['SUCCEEDED', 'FAILED']

    except Exception as e:
        logger.error(f"Error while checking Textract job status for JobId {job_id}: {e}")
        return False

def parse_pdf(s3_filepath):
    global s3_bucket, textract_client

    job_id = textract_client.start_document_text_detection(
        DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': s3_filepath}})["JobId"]
    logger.info(f"Started Textract job {job_id}")

    while not job_complete(job_id):
        time.sleep(5)
    response = textract_client.get_document_text_detection(JobId=job_id)
    next_token_idx = 1
    next_token = response.get('NextToken', None)
    output_files = []

    while response['JobStatus'] == 'SUCCEEDED' and next_token is not None:
        output_files.append(save_file(response, s3_filepath.split(os.sep)[-1].replace(".pdf", f"-{next_token_idx}.json")))
        response = textract_client.get_document_text_detection(JobId=job_id, NextToken=next_token)
        next_token = response.get('NextToken', None)
        next_token_idx += 1
    output_files.append(save_file(response, s3_filepath.split(os.sep)[-1].replace(".pdf", f"-{next_token_idx}.json")))
    return output_files


def convert_to_pages(files, s3_filepath):
    page_content = {}
    current_page = 0
    current_page_data = ""

    for file in files:
        content = load_file(file)
        blocks = content['Blocks']
        for block in blocks:
            if block['BlockType'] == 'LINE':
                if current_page != block['Page']:
                    if len(current_page_data) > 0:
                        page_content[s3_filepath.split(os.sep)[-1] + "::" + str(current_page)] = current_page_data.strip()
                    current_page_data = ""
                    current_page = block['Page']
                current_page_data += block['Text'] + " "
    if len(current_page_data) > 0:
        page_content[s3_filepath.split(os.sep)[-1] + "::" + str(current_page)] = current_page_data.strip()
    return page_content
    # save_file(page_content, s3_filepath.split(os.sep)[-1].replace(".pdf", f".json"))


def convert_to_chunks(page_data, word_threshold=5):
    global nlp
    chunk_data = []

    for key in page_data.keys():
        doc = nlp(page_data[key])
        sentences = [sent.text.strip() for sent in doc.sents]
        for item in sentences:
            if len(item.split(" ")) > word_threshold:
                chunk_data.append({"filename": key.split("::")[0], "page": key.split("::")[-1], "text": item})
    #save_file(chunk_data, s3_key.split(os.sep)[-1].replace(".pdf", f".json"))
    return chunk_data

# files = parse_pdf(FILEPATH)
# files = sorted(glob.glob(FILEPATH.split(os.sep)[-1].replace(".pdf", f"-*.json")))
# content = convert_to_pages(files, FILEPATH)
# convert_to_chunks(content)