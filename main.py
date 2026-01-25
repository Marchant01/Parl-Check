import os
from chatbot import Chatbot
from dotenv import load_dotenv
from document_handler import DocumentHandler


def main():
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    handler = DocumentHandler()
    handler.build_vector_store()
    print("Done!")
    bot = Chatbot(api_key=google_api_key)
    result = bot.ask(
        "Hur har SD röstat under 2024 - 2025"
    )
    print(result)

if __name__ == "__main__":
    main()
