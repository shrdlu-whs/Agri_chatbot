import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader

def get_embeddings(embedding_model_id): # returns the embedding model
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"

def VS_pdfs(embedding_model_id):
    text_chunks_all = list()
    embeedings = get_embeddings(embedding_model_id)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for pdf in os.listdir("./pdfs"):
        loader = PDFMinerLoader(os.path.join("./pdfs",pdf))
        documents = loader.load()
        text_chunks=text_splitter.split_documents(documents)
        text_chunks_all.extend(text_chunks)
    vectorstore=FAISS.from_documents(text_chunks_all, embeedings)
    vectorstore.save_local("./VS/VS_pdfs")
    
save_vs_pdf = VS_pdfs(embedding_model_id)