# ingestion/textract_parser.py
import os
from dotenv import load_dotenv
import logging
import time
import boto3
import json
from typing import Dict, List

load_dotenv()

s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
if not s3_bucket or not aws_region:
    raise EnvironmentError("S3_BUCKET_NAME or AWS_REGION is not set in the .env file")


# Configure logging for every module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Function to process the PDF files
def process_pdf_files():
    # Get the directory of the script and data folder
    # ingestion_dir = os.getcwd()
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
    pdf_folder = os.path.join(base_dir, "data")  # Path to data folder with PDF files

    # Verify that the data folder exists with PDF files
    if not os.path.exists(pdf_folder):
        raise FileNotFoundError(f"PDF folder not found at {pdf_folder}")

    # Find all PDFs
    pdf_files = [
        os.path.join(pdf_folder, filename)
        for filename in os.listdir(pdf_folder)
        if filename.lower().endswith('.pdf')
    ]

    if not pdf_files:
        print("No PDF files found in directory")
        return

    # Process each file
    for pdf_path in pdf_files:
        print(f"\nProcessing {os.path.basename(pdf_path)}...")
        try:
            result = extract_text_with_textract(pdf_path, s3_bucket)
            print(f"Success (Job ID: {result.get('job_id', 'N/A')})")
        except Exception as e:
            print(f"Failed to process {pdf_path}: {str(e)}")


# Function to extract text and metadata from PDF file using AWS Textract
def extract_text_with_textract(pdf_path: str, s3_bucket: str) -> Dict:

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    # Setting clients for Textract and S3
    textract = boto3.client('textract', region_name=aws_region)
    s3 = boto3.client('s3')

    try:
        # Uploads input PDF files to 'pdf-uploads' folder in S3 bucket
        s3_key = f"pdf-uploads/{os.path.basename(pdf_path)}"
        s3.upload_file(pdf_path, s3_bucket, s3_key)
        logger.info(f"Uploaded {pdf_path} to s3://{s3_bucket}/{s3_key}")

        # Start Textract job to extract data from PDF file present on S3
        response = textract.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': s3_bucket, 'Name': s3_key}}
        )
        # Stores the job ID for tracking purpose
        job_id = response['JobId']
        logger.info(f"Started Textract job {job_id}")

        # Wait for job to complete and raises exception if job fails
        max_attempts = 60
        attempt = 0
        while attempt < max_attempts:
            status = textract.get_document_text_detection(JobId=job_id)
            if status['JobStatus'] == 'SUCCEEDED':
                break
            if status['JobStatus'] == 'FAILED':
                logger.error(f"Textract job failed: {status.get('StatusMessage')}")
                raise Exception("Textract job failed")
            time.sleep(5)
            attempt += 1
        else:
            raise TimeoutError("Textract job polling exceeded timeout")

        # Retrieve results
        blocks = []
        next_token = None
        while True:
            params = {'JobId': job_id}
            if next_token:
                params['NextToken'] = next_token

            response = textract.get_document_text_detection(**params)
            blocks.extend(response['Blocks'])
            next_token = response.get('NextToken')
            if not next_token:
                break

        # Process blocks into structured data
        result = {
            'text': '',
            'pages': {},
            'blocks': [],
            'job_id': job_id
        }

        # Stores Page number and layout geometry
        for block in blocks:
            if block['BlockType'] == 'PAGE':
                result['pages'][block['Id']] = {
                    'number': block.get('Page', 1),
                    'geometry': block['Geometry']
                }
            # Concatenates text and stores individual lines with text content, page number, geometry and confidence score
            elif block['BlockType'] == 'LINE':
                result['text'] += block['Text'] + '\n'
                result['blocks'].append({
                    'text': block['Text'],
                    'page': block.get('Page', 1),
                    'geometry': block['Geometry'],
                    'confidence': block['Confidence']
                })

        # Saves the extracted result to S3 folder 'textract-output'
        output_key = f"textract-output/{s3_key.replace('pdf-uploads/', '').replace('.pdf', '.json')}"
        s3.put_object(
            Bucket=s3_bucket,
            Key=output_key,
            Body=json.dumps(result)
        )
        logger.info(f"Saved extracted PDF results to s3://{s3_bucket}/{output_key}")

        return result

    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    process_pdf_files()
