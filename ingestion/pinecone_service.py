from typing import List, Dict
from dotenv import load_dotenv
import json
import logging
import boto3
import os
from pinecone import Pinecone

load_dotenv()

# Fetching environment variables
# s3_bucket = os.getenv('S3_BUCKET_NAME')
# aws_region = os.getenv('AWS_REGION')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index = os.getenv('PINECONE_INDEX_NAME')
pinecone_namespace = os.getenv('PINECONE_NAMESPACE')

# Validate environment variables
required_vars = {
    # 'S3_BUCKET_NAME': s3_bucket,
    # 'AWS_REGION': aws_region,
    'PINECONE_API_KEY': pinecone_api_key,
    'PINECONE_ENVIRONMENT': pinecone_env,
    'PINECONE_INDEX_NAME': pinecone_index,
    'PINECONE_NAMESPACE': pinecone_namespace
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


def upload_to_pinecone(pinecone_vectors):
    """
    Uploads embedding vectors with metadata to Pinecone index.
    Args:
        pinecone_vectors (List[Dict]): Output from embed_batch(). Each item must contain 'id', 'values' and 'metadata'.
    """

    global pinecone_api_key, pinecone_env, pinecone_index, pinecone_namespace
    try:
        # Initializing Pinecone
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)

        # Connect to Pinecone index
        index_name = pc.Index(pinecone_index)
        logger.info(f"Successfully connected to index: {index_name}")

        # Convert items into (id, values, metadata) tuples for upsert
        formatted_vectors = [(item['id'], item['values'], item['metadata']) for item in pinecone_vectors]

        # Upload in batches
        batch_size = 100
        for i in range(0, len(formatted_vectors), batch_size):
            batch = formatted_vectors[i:i + batch_size]
            index_name.upsert(vectors=batch, namespace=pinecone_namespace)

        logger.info(f"Uploaded {len(pinecone_vectors)} vectors to Pinecone index '{index_name}' in namespace '{pinecone_namespace}'")

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone client: {str(e)}")
        raise


# import uuid
# vc = [0.0] * 1023 + [0.1]
# embedding_test = {'id': str(uuid.uuid4()), 'values': vc, 'metadata': {'filename': 'abc.txt', 'page': '9', 'text': 'abcdef'}}
# upload_to_pinecone([embedding_test])


