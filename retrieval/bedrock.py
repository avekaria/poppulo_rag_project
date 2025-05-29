import boto3
import os
import json
import uuid
from dotenv import load_dotenv
import logging

load_dotenv()

#from utils import load_file

# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
aws_region = os.getenv('AWS_REGION')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
embedding_model = os.getenv('EMBEDDING_MODEL_NAME')
bedrock_llm_model = os.getenv('BEDROCK_LLM_MODEL')
bedrock_region = os.getenv('BEDROCK_MODEL_REGION')

# Validate environment variables
required_vars = {
    'S3_BUCKET_NAME': s3_bucket,
    'AWS_REGION': aws_region,
    'AWS_ACCESS_KEY_ID': aws_access_key,
    'AWS_SECRET_ACCESS_KEY': aws_secret_key,
    'EMBEDDING_MODEL_NAME': embedding_model,
    'BEDROCK_LLM_MODEL': bedrock_llm_model,
    'BEDROCK_MODEL_REGION': bedrock_region
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
system_message = "You are a helpful assistant who uses the provided context to answer questions"

def embed_single(sentence):
    global bedrock_client, embedding_model
    body = {"inputText": sentence, "dimensions": 1024, "normalize": True}
    response = bedrock_client.invoke_model(modelId=embedding_model, body=json.dumps(body))
    content = json.loads(response['body'].read())
    return content['embedding']

def embed_batch(sentence_list):
    pinecone_list = []
    for item in sentence_list:
        metadata = {}
        for mk in item.keys():
            metadata[mk] = item[mk]
        pc_item = {"id": str(uuid.uuid4()), "values": embed_single(item['text']), "metadata": metadata}
        pinecone_list.append(pc_item)
    return pinecone_list


def llm_call(context, query):
    global system_message, bedrock_client, bedrock_llm_model
    prompt = {"role": "user", "content": [{"text": f"{context}\nQUERY:{query}"}]}
    system = [{"text": system_message}]
    config = {"maxTokens": 500, "temperature": 0, "topP": 0.1}

    response = bedrock_client.converse(modelId=bedrock_llm_model, messages=[prompt], system=system, inferenceConfig=config)
    output = response["output"]["message"]["content"][0]["text"]
    return output


# sentences = ["The sky is blue", "Who are you?"]
# sentences = [{"filename": "ABC.txt", "text": "How are you?"}, {"filename": "BCD.txt", "text": "The sky is blue?"}]
# print(embed_batch(sentences))
#response = llm_call("John is a fisherman. He fishes in the Nile. He has been fishing for 23 years. He has been fishing since age 5.", "When was John born?")
#print(response)

# embed_batch(load_file('1706.03762_page2.json'))