import os
from chatbot import Chatbot
from fastapi import FastAPI
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

from api_caller import fetch_document_html

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https//example.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
API_BASE = os.getenv("API_BASE")

app = FastAPI()

bot = Chatbot()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post(API_BASE + "/chat")
async def post_question(prompt: str):
    # Put up prompt protection here
    if prompt:
        response = bot.ask(prompt)
        return {"response": response}

@app.get("/anforande/{anforande_id}")
async def post_anforande(anforande_id: str):
    if anforande_id:
        response = fetch_document_html(anforande_id).page_content
        return {"response": response}