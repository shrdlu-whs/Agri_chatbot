import streamlit as st
from LLM_code import fused_response

# Load user guide on prompt creation
txt_path = "How_To_Create_Effective_Prompts.md"
with open(txt_path, 'r', encoding="utf8") as f:
        prompt_guide = f.read()

st.set_page_config(
    page_title="Agri Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Agri Chatbot")

st.sidebar.markdown(
        prompt_guide
 )

# Check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, I am Agri Chatbot. How can I help you today?"}
    ]

# Display all messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get the user input
user_prompt = st.chat_input()

# Process user input and generate a response
if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    # Make sure the assistant responds only when the last message isn't from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                # Call the synchronous function to get the response
                ai_response = fused_response(user_prompt)  # Direct call without await
                ai_response = ai_response.content
                st.write(ai_response)
        
        new_ai_message = {"role": "assistant", "content": ai_response}
        st.session_state.messages.append(new_ai_message)

