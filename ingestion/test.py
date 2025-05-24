import os
from traceback import print_tb

current_folder = os.getcwd() #rag_pdf_app
# path = os.path.dirname(current_folder) # Document
# path =  os.path.dirname(os.path.dirname(__file__)) # Document

ingestion_dir = os.getcwd()
base_dir = os.path.dirname(os.path.dirname(__file__))  # Goes up one level
pdf_folder = os.path.join(base_dir, "data")  # Path to data folder with PDF files

print(ingestion_dir)
print(base_dir)
print(pdf_folder)
print(os.path.basename(pdf_folder))