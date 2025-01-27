import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory

load_dotenv(override = True)
groq_api_key = os.getenv('groq_api_key')

def read_txt(txt_path):  # Returns the content in the text files
    with open(txt_path, 'r') as f:
        content = f.read()
    return content

def load_llm(model_id):  # Returns the defined llm
    llm = ChatGroq(model_name=model_id, temperature=0, groq_api_key=groq_api_key)
    return llm

def get_embeddings(embedding_model_id):  # Returns the embedding model
    return HuggingFaceEmbeddings(model_name=embedding_model_id)

embedding_model_id = "sentence-transformers/all-MiniLM-L12-v2"
model_id = "llama-3.3-70b-versatile"
llm = load_llm(model_id)

# Create Conversational RAG chain
def conversation_QAchain(path):
    vectorstore = FAISS.load_local(path, get_embeddings(embedding_model_id), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    message_history = ChatMessageHistory()
    window_memory = ConversationBufferWindowMemory(k=4,
                                                   memory_key="chat_history",
                                                   output_key="answer",
                                                   chat_memory=message_history,
                                                   return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=retriever,
                                                  chain_type="stuff",
                                                  memory=window_memory,
                                                  return_source_documents=True,
                                                  get_chat_history=lambda h: h)
    return chain

# Create Conversational LLM chain
def conversational_llm():
    chain_llm = ConversationChain(
        llm=llm,
        verbose=False,
        memory=ConversationBufferWindowMemory(k=4)
    )
    return chain_llm

# Generate query responses from all three sources
# RAG with papers in vector store
# RAG with tables in vector store
# General LLM
def Responses(query):
    chain_pdf = conversation_QAchain("./VS/VS_pdfs")
    chain_tables = conversation_QAchain("./VS/VS_tables")
    chain_llm = conversational_llm()

    # Synchronously invoke the chains for responses
    res_pdf = chain_pdf.invoke(query)
    res_tables = chain_tables.invoke(query)
    gen_responses = chain_llm.invoke(query)
    
    return res_pdf, res_tables, gen_responses

# Create prompt template
def prompt():
    template = read_txt("Prompt_for_fusion_1.txt")
    prompt_template = PromptTemplate(input_variables=["Answer_from_scientic_papers", "Answer_from_scientific_data_tables", "Generic_Answer_from_LLM"], template=template)
    return prompt_template

# Fuse query responses from all three LLMs into one response
def fused_response(query):
    res_pdf, res_tables, gen_responses = Responses(query)
    prompt_template = prompt()
    final_prompt = prompt_template.format(Answer_from_scientic_papers=res_pdf["answer"], Answer_from_scientific_data_tables=res_tables["answer"], Generic_Answer_from_LLM=gen_responses["response"])
    return llm.invoke(final_prompt)

