import gradio as gr
import logging
import os
from utils import upload_pdfs_to_s3
from app import run_ingestion_pipeline, run_llm_pipeline


# Fetching environment variables
s3_bucket = os.getenv('S3_BUCKET_NAME')
s3_input_folder = os.getenv('S3_INPUT_DATA_FOLDER')
aws_region = os.getenv('AWS_REGION')
aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

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

def upload_and_ingest(files):
    file_paths = [file.name for file in files]
    logger.info("Starting PDF file upload to S3")
    s3_keys = upload_pdfs_to_s3(
        local_filepaths=file_paths,
        s3_bucket=s3_bucket,
        s3_input_folder=s3_input_folder,
        aws_access_key=aws_access_key,
        aws_secret_key=aws_secret_key,
        aws_region=aws_region
    )
    run_ingestion_pipeline(s3_keys)
    return "Uploaded and Ingested files:\n" + "\n".join(s3_keys)

def ask_question(user_query):
    response = run_llm_pipeline(user_query)
    return response


with gr.Blocks() as demo:
    gr.HTML("""
    <style>
        .title {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 1rem;
            color: #6366f1; 
        }
        .subtitle, .file_title {
            font-size: 1.25rem;
            font-weight: 300;
            color: #374151; 
            margin: 0 0 4px 0 !important;
            padding: 0;
        }
        .custom-button {
            background-color: #477e5e !important; 
            color: white !important;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: background-color 0.3s ease;
            height: 40px !important;
        }
        .custom-button:hover {
            background-color: #000000 !important;
            border: 1px solid #444;
            color: white;
        }
        .custom-box,  {
            background-color: #272729 !important;  
        }
    </style>
    <div class='title'>PDF RAG Application</div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML("<div class='subtitle'>Upload and Process Files</div>")
            file_input = gr.File(label="Choose PDF file(s)", file_types=[".pdf"], file_count="multiple", height="130px", elem_classes="custom-box")
            upload_btn = gr.Button("Upload and Process", elem_classes="custom-button")
            output_box = gr.Textbox(label="Upload Status", lines=2, interactive=False, elem_classes="custom-box")
        with gr.Column(scale=1):
            gr.HTML("<div class='subtitle'>Ask a Question</div>")
            user_query = gr.Textbox(label="Enter the query", placeholder="Ask a question about the uploaded PDFs...", lines=10, elem_classes="custom-box")
            submit_btn = gr.Button("Submit", elem_classes="custom-button")

    gr.HTML("<div class='subtitle'>LLM Response</div>")
    llm_response = gr.Textbox(label="LLM Response", lines=20, interactive=False, elem_classes="custom-box")

    upload_btn.click(fn=upload_and_ingest, inputs=[file_input], outputs=[output_box])
    submit_btn.click(fn=ask_question, inputs=[user_query], outputs=[llm_response])

demo.launch(server_name="0.0.0.0", server_port=7860)

