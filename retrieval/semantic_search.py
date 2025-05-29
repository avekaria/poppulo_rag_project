import os
from pinecone import Pinecone
import logging
import boto3

from retrieval.bedrock import embed_single, llm_call

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
embedding_model = os.getenv('EMBEDDING_MODEL_NAME')
bedrock_llm_model = os.getenv('BEDROCK_LLM_MODEL')
bedrock_region = os.getenv('BEDROCK_MODEL_REGION')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
pinecone_index = os.getenv('PINECONE_INDEX_NAME')
pinecone_namespace = os.getenv('PINECONE_NAMESPACE')


# Validate environment variables
required_vars = {
    'S3_BUCKET_NAME': s3_bucket,
    'AWS_REGION': aws_region,
    'AWS_ACCESS_KEY_ID': aws_access_key,
    'AWS_SECRET_ACCESS_KEY': aws_secret_key,
    'EMBEDDING_MODEL_NAME': embedding_model,
    'BEDROCK_LLM_MODEL': bedrock_llm_model,
    'BEDROCK_MODEL_REGION': bedrock_region,
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


# Initialize Bedrock client
bedrock_client = boto3.client(service_name='bedrock-runtime', aws_access_key_id=aws_access_key,
                              aws_secret_access_key=aws_secret_key, region_name=bedrock_region)


def semantic_search_and_generate(user_query, top_k=10):
    global bedrock_client, embedding_model, pinecone_index, pinecone_namespace

    logger.info(f"Starting semantic search for query: {user_query}")

    # Embed the user query
    try:
        query_embedding = embed_single(user_query)
        logger.info("Query successfully embedded.")
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    # Initialize Pinecone and access the index
    try:
        pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
        index = pc.Index(pinecone_index)
        logger.info(f"Successfully connected to index: {pinecone_index}")
    except Exception as e:
        logger.error(f"Failed to initialize or connect to Pinecone: {e}")
        raise

    # Query Pinecone to retrieve top-k chunks
    try:
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=pinecone_namespace
        )
        logger.info(f"Retrieved top {top_k} similar chunks from Pinecone.")
        for idx, m in enumerate(search_results['matches']):
            logger.info(f"Match {idx + 1}: Page {m['metadata']['page']} from {m['metadata']['filename']} - {m['metadata']['text']}")
    except Exception as e:
        logger.error(f"Failed to query Pinecone: {e}")
        raise

    # Build context from top-k chunks
    try:
        top_chunks = [match['metadata']['text'] for match in search_results['matches']]
        context = "\n\n---\n\n".join(top_chunks)
        logger.info("Generated context from top chunks.")
    except Exception as e:
        logger.error(f"Failed to build context from search results: {e}")
        raise

    # Generate LLM response using top-k chunks as context and user query
    try:
        response = llm_call(context=context, query=user_query)
        logger.info("Generated LLM response successfully.")
        return response
    except Exception as e:
        logger.error("Failed to generate LLM response: %s", e)
        raise

