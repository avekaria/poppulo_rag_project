import os
import logging
import spacy
import json

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(pagify_json(data))


# --- Main logic to process all files ---
def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "../textract_json_output"))
    output_dir = os.path.abspath(os.path.join(script_dir, "../chunks_input"))
    output_file = os.path.join(output_dir, "combined_output.json")

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    combined_results = {}

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_path = os.path.join(input_dir, filename)
            try:
                result = parse_file(json_path)
                combined_results[filename] = result  # Use filename as key
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    # Write combined results
    with open(output_file, "w", encoding="utf-8") as out_file:
        json.dump(combined_results, out_file, indent=2)

    print(f"✅ Processed {len(combined_results)} files.")
    print(f"📄 Output saved to: {output_file}")




def sentence_chunking(text):
    """
    Split text into sentences using SpaCy
    """
    doc = nlp(text)
    chunks = []

    for i, sent in enumerate(doc.sents):
        chunks.append({
            'id': f"sent_{i}",
            'text': sent.text.strip()
        })

    return chunks

def process_text_files():
    # Define the path to the data folder
    base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
    data_folder = os.path.join(base_dir, "data")  # Path to data folder with PDF files

    # Check that the folder exists
    if not os.path.isdir(data_folder):
        logger.error(f"Data folder not found at: {data_folder}")
        return

    # Get all .txt files in the folder
    txt_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.txt')]

    if not txt_files:
        logger.info("No .txt files found in the data folder.")
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
    process_text_files()
