# chunking.py
from typing import List, Dict
import spacy
import json
import logging
import boto3
import os

load_dotenv()

s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
if not s3_bucket or not aws_region:
    raise EnvironmentError("S3_BUCKET_NAME or AWS_REGION is not set in the .env file")
s3_key = f"textract-output/{s3_key.replace('pdf-uploads/', '').replace('.pdf', '.json')}"


# Configure logging for every module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load English language model that includes sentence segmentation capabilities
nlp = spacy.load("en_core_web_sm")

# Load a Textract JSON output from S3
def load_textract_output_from_s3(s3_bucket: str, s3_key: str) -> Dict:
    try:
        s3 = boto3.client('s3', region_name=aws_region)
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        content = response['Body'].read()
        return json.loads(content)
    except Exception as e:
        logger.error(f"S3 load failed: {e}")
        raise

# Function to convert Textract output into sentence-level chunks
def sentence_chunking(textract_output: Dict) -> List[Dict]:
    #Processes the Textract output and processes it through spaCy model
    try:
        doc = nlp(textract_output['text'])
    except Exception as e:
        logger.error(f"NLP processing failed: {e}")
        return []

    chunks = []         # Will store all sentence chunks
    current_page = 1    # Default page number if not found
    sentence_id = 0     # Counter for unique sentence IDs

    # Iterate through spaCy-identified sentences
    for sent_id, sent in enumerate(doc.sents):
        sentence_text = sent.text.strip()
        best_block = None

        # Find page number for each sentence in the Textract output
        for block in textract_output['blocks']:
            if block['BlockType'] == 'LINE' and sentence_text in block['text']:
                best_block = block
                break

        # Default metadata if no match is found
        metadata = {
            'page': best_block.get('page', 1) if best_block else 1,
            'confidence': best_block.get('confidence') if best_block else None,
            'geometry': best_block.get('geometry') if best_block else None
        }

        chunks.append({
            'id': f"sent_{sentence_id}",        # Unique identifier
            'text': sent.text,                  # The sentence text
            'page': current_page,               # Source page number
            'document': textract_output.get('document_name', 'unknown'),    # Name of the document where the text belongs
            'metadata': {
                'confidence': metadata['confidence'],   # Confidence score for each block of text
                'geometry': metadata['geometry']        # Geometry info for each block of text
            }
        })
        sentence_id += 1

    return chunks