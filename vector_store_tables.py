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

#embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
embedding_model_id = "intfloat/e5-base-v2"
tables_load_path = "./Agri_chatbot/tables"
tables_save_path = "./Agri_chatbot/VS/VS_tables"
tables_metadata = "./Agri_chatbot/tables/metadata.csv"

def preprocess_financial_allocation_cap(df):
    df = df.drop(columns=['Member State Code','CSP Version','CSP Version Status'])
    return df

def create_documents(df, metadata, rows_per_document: int = 1):
    documents = []
    for start_idx in range(0, len(df), rows_per_document):
        content=""
        # Get the batch of rows for the current document
        batch = df.iloc[start_idx:start_idx + rows_per_document]
        # Add rows for the current batch
        content += "\n".join([", ".join([f"{row[col]}" for col in df.columns]) for _, row in batch.iterrows()]) + "\n"
        # Create a document with the content
        documents.append(Document(page_content=content, metadata=metadata))

    return documents

# Include table header in each chunk
def include_headers_in_chunks(chunks, header, embedding_header="passage"):
    updated_chunks = []
    for chunk in chunks:
        # Prepend the header to the chunk
        updated_chunk = header + "\n" + chunk.page_content
        # Prepend embedding header
        chunk.page_content = f"{embedding_header}: {chunk.page_content}"

        # Add the updated chunk to the list
        updated_chunks.append(Document(page_content=updated_chunk, metadata=chunk.metadata))
    return updated_chunks

def add_documents_to_vector_store(batch, vector_store):
    vector_store.add_documents(documents=batch)

# Convert tables in CSV ans XSLX format to embeddings in vector store
def vector_store_tables(embedding_model_id, tables_load_path, tables_save_path):
    text_chunks_all = list()
    embeddings = get_embeddings(embedding_model_id)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    dim = len(embeddings.embed_query("dummy text"))
    index = faiss.IndexHNSWFlat(dim, 32)
    
    # Process CSV and Excel files in the metadata csv
    # Load metadata csv
    metadata = pd.read_csv(tables_metadata)

    for index, row in metadata.iterrows():
        print(row['file'])
        print("\n")
        file = row['file']
        metadata = row.to_dict()
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(tables_load_path, file), encoding='utf-8', encoding_errors="replace")
        elif file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(tables_load_path, file))
        
        if file.startswith("Financial allocation to CAP"):
            df = preprocess_financial_allocation_cap(df)
        documents = create_documents(df, metadata, rows_per_document=5)
        # Split the documents into chunks
        header = ", ".join([f"{col}" for col in df.columns]) 
        text_chunks = text_splitter.split_documents(documents)
        # Include table headers and embedding prefix in chunk
        text_chunks_with_header = include_headers_in_chunks(text_chunks, header)
        text_chunks_all.extend(text_chunks_with_header)

    vectorstore = FAISS.from_documents(text_chunks_all, embeddings)
    vectorstore.save_local(tables_save_path)

vector_store_tables(embedding_model_id, tables_load_path, tables_save_path)