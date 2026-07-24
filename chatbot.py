import os
import re
import torch
import json as pyjson

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text


class Chatbot:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        load_dotenv()
        self.api_key = os.getenv("LLM_API_KEY")
        self.database_url = os.getenv("DATABASE_URL")
        
        #Local embedding model that will run either run with cuda(GPU) or PCU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": 128,
                "normalize_embeddings": True
            },
        )
        
        
        self.engine = create_engine(
            self.database_url,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_pre_ping=True
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        #LLM model through API key
        self.model = init_chat_model("google_genai:gemini-2.5-flash-lite", api_key=self.api_key)
                
        #Prompt for LLM
        self.prompt = ChatPromptTemplate.from_template(
            "Du är en assistent som svarar på frågor om svenska riksdagsanföranden.\n"
            "Svara ENDAST utifrån informationen i KONTEXT nedan. Om svaret inte "
            "framgår av kontexten, säg tydligt att du inte kan hitta svaret i underlaget.\n"
            "Hänvisa till talare när det är relevant, t.ex. '[2] Magdalena Andersson (S) sa...'.\n\n"
            "KONTEXT:\n"
            "{context}\n\n"
            "FRÅGA: {question}\n\n"
            "SVAR:"
        )
        
        self.pipeline = (
            {
                "context": RunnableLambda(self.retrieve_anforande),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.model
        )

    def retrieve_anforande(self, question: str, k: int = 20) -> str:
        """Embed question, pull the k closest anforande  rows, format for the LLM."""
        query_embedding = self.embeddings.embed_query(question)

        with self.SessionLocal() as sess:
            rows = sess.execute(
                text("""
                    SELECT talare, parti, anforandetext 
                    FROM anforande 
                    ORDER BY embedding <-> CAST(:embedding AS vector) 
                    LIMIT :k
                """),
                {"embedding": str(query_embedding), "k":k},
            ).fetchall()

        return self._format_anforand(rows)

    @staticmethod
    def _format_anforand(rows) -> str:
        """Turn DB rows into clear numbered blocks for the LLM"""
        if not rows:
            return "Ingen relevant information hittades i databasen."
        
        blocks = []
        for i, row in enumerate(rows, 1):
            blocks.append(
                f"[{i}] Talare: {row.talare} ({row.parti}\n)"
                f"{row.anforandetext}"
            )
            return "\n\n---\n\n".join(blocks)

    def ask(self, question) -> dict:
        """ Main qeuery """
        response = self.pipeline.invoke(question)
        return {"answer": response.content}
        
