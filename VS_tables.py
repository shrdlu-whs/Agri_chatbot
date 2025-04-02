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

def process_pesticide_effects(df):
    # Clean data
    df = df.drop(columns=["id","X","CodingFinished","Screener","Extracter","Extraction.Suitable","ExtractionComments","Comments","GeneralComments","Country","SystemComments","ExperimentObservation","SpatialExtent.km2.","CaseDescription","Case_ID"])
    na_columns = df.columns[df.isna().all()].tolist()
    df = df.drop(columns=na_columns)
    single_value_columns = [col for col in df.columns if df[col].isin([False, 'false','FALSE',True,'true','TRUE','yes','no']).all()]
    df = df.drop(columns=single_value_columns)
    print("Columns with only NA values are dropped:", na_columns)
    print("Columns with only false or true values are dropped:", single_value_columns)
    # Drop columns with a single unique value
    single_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    df = df.drop(columns=single_value_columns)
    print("Columns with a single unique value are dropped:", single_value_columns)
    # Group rows by unique paper ID and create a document for each group
    grouped = df.groupby("ID")
    docs = []
    for paper_id, group in grouped:
        metadata = {
            "id": paper_id,
            "author": group["Author"].iloc[0] if "Author" in group.columns else None,
            "title": group["NameOfPDF"].iloc[0] if "NameOfPDF" in group.columns else None,
            "doi": group["DOI"].iloc[0] if "DOI" in group.columns else None
        }
        # Drop metadata columns from the group
        group = group.drop(columns=["ID", "Author", "NameOfPDF", "DOI"], errors="ignore")
        # Combine all rows to a single output string
        content = ""
        for index, row in group.iterrows():
            output_str = ""
            for col in group.columns:
                output_str += f"{col}: {row[col]}, "
            output_str += "\n"
            content = ''.join([content,output_str])
        # Create a document with the content and metadata
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def add_documents_to_vector_store(batch, vector_store):
    vector_store.add_documents(documents=batch)

# Convert tables in CSV ans XSLX format to embeddings in vector store
def vector_store_tables(embedding_model_id, tables_load_path, tables_save_path):
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
        
        if file.startswith("PesticidesEffectsClean"):
            docs = process_pesticide_effects(df)
        else:
            # Create documents from file
            content = ""
            for index, row in df.iterrows():
                output_str = ""
                for col in df.columns:
                    output_str += f"{col}: {row[col]}, "
                output_str += "\n"
                content = ''.join([content,output_str])
            
            documents = Document(page_content=content, metadata={"author": "Atharva Ingle","title": "Crop Recommendation Dataset", "url":"https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"})
        # Split document into chunks
        text_chunks = text_splitter.split_documents([documents])
        text_chunks_all.extend(text_chunks)
    batch_size = 2000
    total_documents = len(text_chunks_all)

    for i in range(0, total_documents, batch_size):
        batch = text_chunks_all[i:i + batch_size]
        add_documents_to_vector_store(batch, vector_store)
    vector_store.save_local(tables_save_path)

vector_store_tables(embedding_model_id, tables_load_path, tables_save_path)