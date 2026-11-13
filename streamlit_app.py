
import streamlit as st
from chatbot import get_bot_response

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Application title and Clear Chat button
header_col, button_col = st.columns([4, 1])
with header_col:
    st.title("🤖 AI Chatbot")
    st.caption("Your friendly AI-powered conversation assistant")
with button_col:
    st.write("")
    if st.button("Clear Chat", help="Clear conversation history"):
        st.session_state.messages = []
        st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Type your message here...")

if user_input is not None:
    cleaned_input = user_input.strip()

    if not cleaned_input:
        # Handle empty or whitespace-only inputs safely
        st.warning("Please enter a non-empty message.")
    else:
        # Save and display user message
        st.session_state.messages.append({
            "role": "user",
            "content": cleaned_input
        })

        with st.chat_message("user"):
            st.markdown(cleaned_input)

        # Generate, save, and display chatbot response immediately
        bot_response = get_bot_response(cleaned_input)

        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_response
        })

        with st.chat_message("assistant"):
            st.markdown(bot_response)
