import os
from chatbot import Chatbot
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

from api_caller import fetch_document_html

from pydantic import BaseModel

API_PREFIX = "/api"

class ChatRequest(BaseModel):
    prompt: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://parlcheck.se"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = Chatbot()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post(API_PREFIX + "/chat")
async def post_question(request: ChatRequest):
    # Put up prompt protection here
    if request.prompt:
        response = bot.ask(request.prompt)
        return {"response": response}

@app.get(API_PREFIX + "/anforande/{anforande_id}")
async def post_anforande(anforande_id: str):
    if anforande_id:
        response = fetch_document_html(anforande_id).page_content
        return {"response": response}