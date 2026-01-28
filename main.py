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
        # "Vad nämndes under anförande HA091?"
        "Kan du ge mig information om anföranden relaterat till klimatet och miljön?"
    )
    print(result)

if __name__ == "__main__":
    main()
