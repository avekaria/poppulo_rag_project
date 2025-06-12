import logging
import os
import argparse

from ingestion.textract import parse_pdf, convert_to_pages, convert_to_chunks
from retrieval.bedrock import embed_batch
from ingestion.pinecone_service import upload_to_pinecone
from retrieval.semantic_search import semantic_search_and_generate

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
    handlers=[
        logging.FileHandler("app.log"),  # Writes the logs to a file
        logging.StreamHandler()  # Shows logs in console
    ]
)
logger = logging.getLogger(__name__)


def run_ingestion_pipeline(s3_filepath):
    logger.info("Starting ingestion pipeline")

    for s3_key in s3_filepath:
        logger.info(f"Starting ingestion for {s3_key}")
        files = sorted(parse_pdf(s3_key))
        #files = sorted(glob.glob(s3_key.split(os.sep)[-1].replace(".pdf", f"-*.json")))
        content = convert_to_pages(files, s3_key)
        chunks = convert_to_chunks(content)
        pinecone_vectors = embed_batch(chunks)
        upload_to_pinecone(pinecone_vectors)

    logger.info("Ingestion completed")

def run_llm_pipeline(user_query):
    logger.info("Starting LLM response pipeline")

    response = semantic_search_and_generate(user_query=user_query)

    logger.info("Generated LLM response")
    return response
