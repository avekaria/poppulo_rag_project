# chunking.py
from typing import List, Dict
from dotenv import load_dotenv
import spacy
import json
import logging
import boto3
import os
import pinecone

load_dotenv()

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index = os.getenv('PINECONE_INDEX_NAME')
embedding_model = os.getenv('EMBEDDING_MODEL_NAME')
s3_key = f"textract-output/"

# Validate environment variables
required_vars = {
    'S3_BUCKET_NAME': s3_bucket,
    'AWS_REGION': aws_region,
    'PINECONE_API_KEY': pinecone_api_key,
    'PINECONE_ENVIRONMENT': pinecone_env,
    'PINECONE_INDEX_NAME': pinecone_index,
    'EMBEDDING_MODEL_NAME': embedding_model,
}
missing_vars = [name for name, value in required_vars.items() if not value]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")


# Configure logging for every module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load English language model that includes sentence segmentation capabilities
nlp = spacy.load("en_core_web_sm")

# Initializing Pinecone to store chunking data in Vector database
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(pinecone_index)

#process_text_with_spacy_and_pinecone(result, os.path.basename(pdf_path))

# Load a Textract JSON output from S3
def load_textract_output_from_s3(s3_bucket: str, s3_key: str) -> Dict:
    try:
        s3 = boto3.client('s3', region_name=aws_region)
        response = s3.get_object(Bucket=s3_bucket, Key=s3_key)
        data = response['Body'].read()
        logger.info(f"Loaded Textract output from s3://{s3_bucket}/{s3_key}")
        return json.loads(data)
    except Exception as e:
        logger.error(f"Failed to load Textract output from S3: {e}")
        raise


# Function to convert Textract output into sentence-level chunks
def sentence_chunking(textract_output: Dict) -> List[Dict]:
    try:
        doc = nlp(textract_output.get('text', ''))
    except Exception as e:
        logger.error(f"NLP processing failed: {e}")
        return []

    chunks = []         # Stores all sentence chunks
    document_name = os.path.basename(textract_output.get('job_id', 'unknown'))

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




# Function to insert chunked data to Pinecone vector database
def insert_to_pinecone(chunks: List[Dict], document_name: str):
    vectors = []
    for chunk in chunks:
        # Generate embedding (using dummy embedding for example - replace with real model)
        embedding = [0.1] * 384  # Replace with actual embedding model

        vectors.append({
            'id': chunk['id'],
            'values': embedding,
            'metadata': {
                'text': chunk['text'],
                'page': chunk['metadata']['page'],
                'document': document_name,
                'confidence': chunk['metadata']['confidence'],
                'geometry': json.dumps(chunk['metadata']['geometry']),
                'job_id': chunk['metadata']['job_id']
            }
        })

    try:
        index.upsert(vectors=vectors)
        logger.info(f"Inserted {len(vectors)} chunks to Pinecone")
    except Exception as e:
        logger.error(f"Failed to insert to Pinecone: {e}")
        raise
