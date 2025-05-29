import logging
import glob
import os
import argparse

from ingestion.textract import parse_pdf, convert_to_pages, convert_to_chunks
from retrieval.bedrock import embed_batch
from ingestion.pinecone_service import upload_to_pinecone
from retrieval.semantic_search import semantic_search_and_generate


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


def ingest(s3_filepath):
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

def complete():
    logger.info("Starting LLM response pipeline")

    user_query = input("Enter your question: ")
    response = semantic_search_and_generate(user_query=user_query)

    logger.info("Generated LLM response")
    print("\n Answer:\n", response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RAG Pipeline: Ingest and Query PDFs with LLM.")
    parser.add_argument("action", choices=["ingest", "complete"])
    parser.add_argument("--files", nargs="*", help="List of S3 PDF file keys")

    args = parser.parse_args()

    if args.action == "ingest":
        if args.files:
            ingest(args.files)
        else:
            logger.error("Please provide --files argument with S3 keys")
    elif args.action == "complete":
        complete()