import os
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

from api_caller import get_debate_text

class Chatbot:
    def __init__(self, api_key, persist_directory="./chroma_langchain_db"):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "batch_size": 256,
                "normalize_embeddings": True
            },
            show_progress=True
        )
        
        self.model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        
        self.vector_store = Chroma(
            collection_name='documents',
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        self.prompt = ChatPromptTemplate.from_template(
            "Du svarar endast utifrån data.\n\nDATA:\n{data}\n\nFRÅGA:\n{question}\n\nSVAR:"
        )

        def extract_anforande_meta(docs):
            metas = []
            for doc in docs:
                m = doc.metadata or {}
                if m.get("type") in {"anforande", "anföring"}:
                    metas.append(m)
            return metas
        
        def call_anforande_api(metas):
            results = []
            for m in metas:
                doc_id = m.get("dok_id") or m.get("dokument_id")
                nr = m.get("anforande_nummer")
                if doc_id and nr:
                    results.append(get_debate_text(doc_id, nr))
            return results

        self.pipeline = (
            {
                "data": self.retriever
                | RunnableLambda(extract_anforande_meta)
                | RunnableLambda(call_anforande_api),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.model
        )

    def ask(self, question):
        """ Main qeuery """
        response = self.pipeline.invoke(question)
        return {"answer": response.content}

        