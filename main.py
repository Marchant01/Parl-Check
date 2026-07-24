import os
from chatbot import Chatbot
from fastapi import FastAPI
from dotenv import load_dotenv

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
async def read_anforande(anforande_id):
    return {"anforande_id": anforande_id}