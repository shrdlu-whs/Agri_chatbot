import streamlit as st
import os
from LLM_code import fused_response
from langchain.schema import HumanMessage, AIMessage

# Load the user guide on prompt creation
def load_prompt_guide(file_path):
    with open(file_path, 'r', encoding="utf8") as f:
        return f.read()

prompt_guide_path = os.path.abspath("Agri_chatbot/How_To_Create_Effective_Prompts.md")
prompt_guide = load_prompt_guide(prompt_guide_path)

# Configure the Streamlit app
st.set_page_config(
    page_title="Agri Chatbot",
    page_icon="🤖",
    layout="wide"
)

# App title
st.title("Agri Chatbot")

# Display the prompt creation guide in the sidebar
st.sidebar.markdown(prompt_guide)

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello there, I am Agri Chatbot. How can I help you today?")
    ]

# Display all chat messages in the session state
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Handle user input
user_prompt = st.chat_input("Type your message here...")

if user_prompt:
    # Add user message as a HumanMessage to the session state
    user_message = HumanMessage(content=user_prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(user_message.content)

    # Generate a response from the assistant
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Process the user input using fused_response
                response, references_papers, references_tables = fused_response(user_message.content)
                ai_message = AIMessage(content=response)
                st.write(ai_message.content)
                st.write("References:\n")
                for ref in references_papers:
                    st.markdown(f"- {ref}")
                for ref in references_tables:
                    st.markdown(f"- {ref}")

                # Add assistant's response as an AIMessage to the session state
                st.session_state.messages.append(ai_message)

            except Exception as e:
                st.error(f"An error occurred while processing your query: {e}")