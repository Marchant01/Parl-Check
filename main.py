import os
import streamlit as st
from chatbot import Chatbot

@st.cache_resource
def get_chat_bot():
    bot = Chatbot()
    return bot

def chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Välkommen till Parl-Check!")
    chat_history()
    
    bot = get_chat_bot()

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
