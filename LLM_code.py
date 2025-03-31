import os
from typing import List, TypedDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import START, StateGraph
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv(override=True)
groq_api_key = os.getenv("GROQ_API_KEY")

# Define the state structure
class State(TypedDict):
    question: str
    context_papers: List[Document]
    context_tables: List[Document]
    response_papers: str
    response_tables: str
    final_answer: str

# Initialize embedding model
embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
def get_embeddings(embedding_model_id):  # Returns the embedding model
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

# Initialize LLM
#model_id = "llama3-8b-8192"
#model_provider="groq"
model_id="smollm2:135m"
model_provider="ollama"
def load_llm():
    llm = init_chat_model(model_id, model_provider=model_provider,temperature=0, groq_api_key=groq_api_key)
    return llm
llm = load_llm()

# Load vector stores
def load_vector_store(path: str):
    return FAISS.load_local(path, get_embeddings(embedding_model_id), allow_dangerous_deserialization=True)

vector_store_pdf = load_vector_store("./Agri_chatbot/VS/VS_pdfs")
vector_store_tables = load_vector_store("./Agri_chatbot/VS/VS_tables")

# Load prompt template
template_path = "./Agri_chatbot/Prompt_for_fusion_1.txt"
def load_prompt_template():
    with open(template_path, "r") as f:
        template_content = f.read()
    return PromptTemplate(
        input_variables = ["response_papers", "response_tables"],
        template = template_content
    )

prompt = load_prompt_template()

# Define application steps
def retrieve_papers(state: State):
    retrieved_docs = vector_store_pdf.similarity_search(state["question"], k=5)
    return {"context_papers": retrieved_docs}

def retrieve_tables(state: State):
    retrieved_docs = vector_store_tables.similarity_search(state["question"], k=5)
    return {"context_tables": retrieved_docs}

def generate_response_papers(state: State):
    # Combine content into a single string
    context_content = "\n\n".join(doc.page_content for doc in state["context_papers"])
    
    # Format the input using BaseMessages
    messages = [
        HumanMessage(content=f"Question: {state['question']}"),
        HumanMessage(content=f"Context: {context_content}")
    ]
    
    response = llm.invoke(messages)  # Send formatted list of messages
    return {"response_papers": response.content}

def generate_response_tables(state: State):
    # Combine content into a single string
    context_content = "\n\n".join(doc.page_content for doc in state["context_tables"])
    # Format the input using BaseMessages
    messages = [
        HumanMessage(content=f"Question: {state['question']}"),
        HumanMessage(content=f"Context: {context_content}")
    ]
    
    response = llm.invoke(messages)  # Send formatted list of messages
    return {"response_tables": response.content}

def generate_response_generic(state: State):

    # Format the input using BaseMessages
    messages = [
        HumanMessage(content=f"Question: {state['question']}")
    ]
    
    response = llm.invoke(messages)  # Send formatted list of messages
    return {"response_generic": response.content}


def generate_final_response(state: State):
    final_prompt = prompt.format(
        response_papers = state["response_papers"],
        response_tables = state["response_tables"],
    )
    response = llm.invoke(final_prompt)
    return {"final_answer": response.content}

# Build the StateGraph
graph_builder = StateGraph(State).add_sequence([
    retrieve_papers,
    retrieve_tables,
    generate_response_papers,
    generate_response_tables,
    generate_final_response
])
graph_builder.add_edge(START, "retrieve_papers")
graph = graph_builder.compile()

# Example usage
def fused_response(query: str):
    # Initialize state
    state = {
        "question": query,
        "context_papers": [],
        "ref_papers": [],
        "context_tables": [],
        "ref_tables": [],
        "response_papers": "",
        "response_tables": "",
        "final_answer": ""
    }
    
    # Execute graph
    final_state = graph.invoke(state)

    # Extract metadata from papers to cite title, author and date when available
    references_papers = set({
        f"{doc.metadata.get('author', 'Unknown Author')} - {doc.metadata.get('title', 'Untitled')} ({doc.metadata.get('date', 'Unknown Date')})"
        for doc in final_state["context_papers"]
    })

    # Extract metadata from tables when available
    references_tables = set([
        doc.metadata.get("source", "Unknown Source") for doc in final_state["context_tables"]
    ])

    # Retrieve the final answer
    return final_state["final_answer"], references_papers, references_tables

