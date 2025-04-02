import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
import pandas as pd

def get_embeddings(embedding_model_id): # returns the embedding model
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
pdf_load_path = "./Agri_chatbot/pdfs"
pdf_save_path = "./Agri_chatbot/VS/VS_pdfs"
pdf_metadata = "./Agri_chatbot/pdfs/metadata.csv"

# Convert PDF files to embeddings in vector store
def VS_pdfs(embedding_model_id, pdf_load_path, pdf_save_path):
    text_chunks_all = list()
    embeddings = get_embeddings(embedding_model_id)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    # Load metadata csv
    metadata = pd.read_csv(pdf_metadata)

    for index, row in metadata.iterrows():
        print(row['pdf'])
        print("\n")
        loader = PDFMinerLoader(os.path.join(pdf_load_path,row['pdf']))
        document = loader.load()[0]
        doc_metadata = row
        # Convert doc_metadata to dictionary
        doc_metadata = doc_metadata.to_dict()
        # Append metadata to document metadata
        document.metadata.update(doc_metadata)
        print(document.metadata)
        text_chunks = text_splitter.split_documents([document])
        text_chunks_all.extend(text_chunks)
    vectorstore = FAISS.from_documents(text_chunks_all, embeddings)
    vectorstore.save_local(pdf_save_path)
    
save_vs_pdf = VS_pdfs(embedding_model_id, pdf_load_path, pdf_save_path)