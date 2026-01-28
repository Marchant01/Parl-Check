import os
import streamlit as st
from chatbot import Chatbot
from document_handler import DocumentHandler

@st.cache_resource
def get_vector_store():
    handler = DocumentHandler()
    return handler.build_vector_store()

@st.cache_resource
def get_chat_bot(google_api_key):
    bot = Chatbot(api_key=google_api_key)
    return bot

def chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Welcome To Parl-Check!")
    chat_history()
    get_vector_store()

    google_api_key = st.sidebar.text_input("Google API Key", type="password")
    if not google_api_key:
        st.warning("Please enter your google API key!")
        return

    bot = get_chat_bot(google_api_key)

    for turn in st.session_state.messages:
        st.chat_message("user").write(turn["question"])
        st.chat_message("assistant").write(turn["answer"])
        
    prompt = st.chat_input("Ställ en fråga:")
    if prompt:
        with st.spinner("Hämtar svar..."):
            response = bot.ask(prompt)
        st.session_state.messages.append(
            {"question": prompt, "answer": response["answer"]}
        )
        st.rerun()

    if st.sidebar.button("Clear chat"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()
