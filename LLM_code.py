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
    references_papers: dict[str, List[str]]  # Stores references as keys with multiple segments
    context_tables: List[Document]
    references_tables: dict[str, List[str]]  # Same for tables
    response_papers: str
    response_tables: str
    final_answer: str

# Initialize embedding model
embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
def get_embeddings(embedding_model_id):  # Returns the embedding model
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

# Initialize LLM
model_id = "llama3-8b-8192"
model_provider="groq"
#model_id="tinyllama"
#model_provider="ollama"
def load_llm():
    llm = init_chat_model(model_id, model_provider=model_provider,temperature=0, groq_api_key=groq_api_key)
    return llm
llm = load_llm()

# Load vector stores
def load_vector_store(path: str):
    return FAISS.load_local(path, get_embeddings(embedding_model_id), allow_dangerous_deserialization=True)

vector_store_pdf = load_vector_store("./Agri_chatbot/VS/VS_pdfs")
vector_store_tables = load_vector_store("./Agri_chatbot/VS/VS_tables")

# Load prompt templates
template_path_fusion = "./Agri_chatbot/prompts/fusion_prompt.txt"
input_variables_fusion = ["question","response_papers", "response_tables"]
template_path_rag = "./Agri_chatbot/prompts/rag_prompt.txt"
input_variables_rag = ["question", "context"]
def load_prompt_template(template_path, input_variables):
    with open(template_path, "r") as f:
        template_content = f.read()
    return PromptTemplate(
        input_variables = input_variables,
        template = template_content
    )

fusion_prompt_template = load_prompt_template(template_path_fusion, input_variables_fusion)
rag_prompt_template = load_prompt_template(template_path_rag, input_variables_rag)

def retrieve_papers(state: State):
    # Retrieve documents along with relevance scores
    retrieved_docs = vector_store_pdf.similarity_search_with_relevance_scores(
        state["question"], k=3
    )

    threshold = 0.8
    # Filter documents based on the relevance scores
    retrieved_docs = [
        doc for doc, score in retrieved_docs if score >= threshold
    ]

    references_papers = {}

    for doc in retrieved_docs:
        url = doc.metadata.get('url', 'https://')
        ref_key = f"{doc.metadata.get('author', 'Unknown Author')}: {doc.metadata.get('title', 'Untitled')} ({doc.metadata.get('year', 'Unknown Year')}). [{'Link'}]({url})"

        if ref_key not in references_papers:
            references_papers[ref_key] = []
        references_papers[ref_key].append(doc.page_content)

    return {"context_papers": retrieved_docs, "references_papers": references_papers}

def retrieve_tables(state: State):
        # Retrieve documents along with relevance scores
    retrieved_docs = vector_store_tables.similarity_search_with_relevance_scores(
        state["question"], k=3
    )

    threshold = 0.8
    # Filter documents based on the relevance scores
    retrieved_docs = [
        doc for doc, score in retrieved_docs if score >= threshold
    ]
    references_tables = {}

    for doc in retrieved_docs:
        url = doc.metadata.get('url', 'https://')
        ref_key = f"{doc.metadata.get('author', 'Unknown Author')}: {doc.metadata.get('title', 'Untitled')} ({doc.metadata.get('year', 'Unknown Year')}). [{'Link'}]({url})"
        if ref_key not in references_tables:
            references_tables[ref_key] = []
        references_tables[ref_key].append(doc.page_content)

    return {"context_tables": retrieved_docs, "references_tables": references_tables}

def generate_response_papers(state: State):
    # Combine content into a single string
    context_content = "\n\n".join(doc.page_content for doc in state["context_papers"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"],
        context=context_content,
    )

    # Format the input using BaseMessages
    messages = [
        HumanMessage(content=rag_prompt),
    ]
    
    response = llm.invoke(messages)  # Send formatted list of messages
    return {"response_papers": response.content}

def generate_response_tables(state: State):
    # Combine content into a single string
    context_content = "\n\n".join(doc.page_content for doc in state["context_tables"])
    rag_prompt = rag_prompt_template.format(
        question=state["question"],
        context=context_content,
    )

    # Format the input using BaseMessages
    messages = [
        HumanMessage(content=rag_prompt),
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


from fuzzywuzzy import fuzz

def find_best_match(segment, response_text):
    similarity_score = fuzz.partial_ratio(segment, response_text)
    return similarity_score > 50  # Adjust threshold as needed

def generate_final_response(state: State):
    final_prompt = fusion_prompt_template.format(
        question=state["question"],
        response_papers=state["response_papers"],
        response_tables=state["response_tables"],
    )
    response = llm.invoke(final_prompt)
    references_list = set()

    # Apply fuzzy matching for inline citations
    for ref, segments in state["references_papers"].items():
        for segment in segments:
            if find_best_match(segment, response.content):
                response.content = response.content.replace(segment, f"{segment} [{ref}]")
                references_list.add(ref)

    for ref, segments in state["references_tables"].items():
        for segment in segments:
            if find_best_match(segment, response.content):
                response.content = response.content.replace(segment, f"{segment} [{ref}]")
                references_list.add(ref)

    # Append a structured reference section at the end
    reference_section = "\n\nReferences:\n" + "\n".join(f"- {ref}" for ref in references_list)
    response.content += reference_section

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

def fused_response(query: str):
    # Initialize state
    state = {
        "question": query,
        "context_papers": [],
        "references_papers": [],
        "context_tables": [],
        "references_tables": [],
        "response_papers": "",
        "response_tables": "",
        "final_answer": ""
    }
    
    # Execute graph
    final_state = graph.invoke(state)

    # Retrieve the final answer
    return final_state["final_answer"]

