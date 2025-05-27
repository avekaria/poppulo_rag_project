# ingestion/extraction_chunking_embedding.py
import os
from dotenv import load_dotenv
import logging
import spacy
import pinecone
import time
import boto3
import json
from typing import Dict, List


load_dotenv()

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index = os.getenv('PINECONE_INDEX_NAME')
s3_prefix = "textract-output/"

# Validate environment variables
required_vars = {
    'S3_BUCKET_NAME': s3_bucket,
    'AWS_REGION': aws_region,
    'PINECONE_API_KEY': pinecone_api_key,
    'PINECONE_ENVIRONMENT': pinecone_env,
    'PINECONE_INDEX_NAME': pinecone_index,
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")

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


# Load SpaCy model that includes sentence segmentation capabilities
nlp = spacy.load("en_core_web_sm")

# Initializing Pinecone to store chunking data in Vector database
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)


# Function to process the input PDF files for extraction
def process_pdf_files():
    # Get the directory where input data is stored
    base_dir = os.path.dirname(os.path.dirname(__file__))
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


# Function to return a list of JSON files present in "textract-output" folder on S3
def list_textract_outputs(s3_bucket: str, s3_prefix: str) -> List[str]:
    s3 = boto3.client('s3', region_name=aws_region)
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
    return [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.json')]

# Function to transform the Textract output in required format
def pagify_json(data):
    current_page = 0
    current_page_data = ""
    page_content = {}
    for block in data['blocks']:
        if current_page != block['page']:
            if len(current_page_data) > 0:
                page_content[current_page] = current_page_data.strip()
                current_page_data = ""
                current_page = block['page']
        current_page_data += block['text'] + " "
    return page_content

def parse_file(json_file):
    # Define the path to the data folder
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
    json_folder = os.path.join(base_dir, "json_data")  # Path to data folder with JSON files

    # Check that the folder exists
    if not os.path.isdir(json_folder):
        logger.error(f"JSON files not found at: {json_folder}")
        return

    # Get all .json files in the folder
    json_files = [f for f in os.listdir(json_data) if f.lower().endswith('.json')]

    if not json_files:
        logger.info("No .json files found in the json_data folder.")
        return

    # Process each JSON file
    for file_name in json_files:
        json_file = os.path.join(json_data, file_name)
        logger.info(f"Reading file: {json_file}")

        with open(json_file, 'r') as f:
            data = json.load(f)
        print(pagify_json(data))

parse_file('1706.03762v7.json')



# Sentence level chunking using SpaCy
def sentence_chunking(text):
    doc = nlp(text)
    chunks = []

    for i, sent in enumerate(doc.sents):
        chunks.append({
            'id': f"sent_{i}",
            'text': sent.text.strip()
        })

    return chunks

# Fetching Textract output (JSON file) for chunking
def process_text_files():
    # Define the path to the data folder
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
    data_folder = os.path.join(base_dir, "data")  # Path to data folder with PDF files

    # Check that the folder exists
    if not os.path.isdir(data_folder):
        logger.error(f"Data folder not found at: {data_folder}")
        return

    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.json')]

    if not txt_files:
        logger.info("No .json files found in the data folder.")
        return

    # Process each text file
    for file_name in txt_files:
        file_path = os.path.join(data_folder, file_name)
        logger.info(f"Reading file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = sentence_chunking(text)

        logger.info(f"Chunked {len(chunks)} sentences from {file_name}")

        # Print first few sentences
        for chunk in chunks[:5]:
            print(chunk)


if __name__ == "__main__":
    process_pdf_files()
