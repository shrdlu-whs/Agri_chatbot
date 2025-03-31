import os
import pandas as pd
import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore

def get_embeddings(embedding_model_id):
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
tables_load_path = "./Agri_chatbot/tables"
tables_save_path = "./Agri_chatbot/VS/VS_tables"

def add_documents_to_vector_store(batch, vector_store):
    vector_store.add_documents(documents=batch)

# Convert tables in CSV ans XSLX format to embeddings in vector store
def VS_tables(embedding_model_id, tables_load_path, tables_save_path):
    text_chunks_all = list()
    embeddings = get_embeddings(embedding_model_id)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    dim = len(embeddings.embed_query("dummy text"))
    index = faiss.IndexHNSWFlat(dim, 32)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    
    # Process CSV and Excel files in the "tables" directory
    for file in os.listdir(tables_load_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(tables_load_path, file), encoding='utf-8', encoding_errors="replace")
        elif file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(tables_load_path, file))
        
        docs = []
        for index, row in df.iterrows():
            output_str = ""
            for col in df.columns:
                output_str += f"{col}: {row[col]},\n"
            docs.append(output_str)
        
        # Create documents and split into chunks
        documents = Document(page_content=str(docs), metadata={"source": file})
        text_chunks = text_splitter.split_documents([documents])
        text_chunks_all.extend(text_chunks)
    batch_size = 2000
    total_documents = len(text_chunks_all)

    for i in range(0, total_documents, batch_size):
        batch = text_chunks_all[i:i + batch_size]
        add_documents_to_vector_store(batch, vector_store)
    vector_store.save_local(tables_save_path)

save_vs_tab = VS_tables(embedding_model_id, tables_load_path, tables_save_path)