import os
import re
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document

from api_caller import get_document

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
            #show_progress=True
        )
        
        self.model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        
        self.vector_store = Chroma(
            collection_name='documents',
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})

        bs_transformer = BeautifulSoupTransformer()

        def fetch_document_html(doc_id: str) -> Document:
            html = get_document(doc_id)
            return Document(
                page_content=html,
                metadata={"doc_id": doc_id, "source": f"https://data.riksdagen.se/dokument/{doc_id}/html"}
            )

        def html_to_text(doc: Document) -> Document:
            cleaned = bs_transformer.transform_documents([doc], tags_to_extract=["p", "h1", "h2", "h3", "li"])

            return cleaned[0]

        def extract_name_and_parties(cleaned_doc: Document) -> list[dict]:
            text = cleaned_doc.page_content

            m = re.search(r"Följande ledamöter har deltagit i beslutet:(.*)", text)
            if not m:
                return []

            tail = m.group(1)

            pairs = re.findall(r"([^,]+?)\s*\(([^)]+)\)", tail)
            return [{"namn": name.strip(), "parti": party.strip()} for name, party in pairs]


        self.prompt = ChatPromptTemplate.from_template(
            "Du svarar endast utifrån data.\n\n"
            "DOK_ID: {dok_id}\n"
            "KAMMARAKTIVITET: {kammaraktivitet}\n"
            "AVSNITTSRUBRIK: {avsnittsrubrik}\n"
            "TALARE/PARTIER: {speakers}\n\n"
            "TEXT:\n{data}\n\n"
            "FRÅGA:\n{question}\n\n"
            "SVAR:"
        )

        def pick_hit(docs):
            if not docs:
                return {}
            m = docs[0].metadata or {}
            #print("retriever hit:", m)
            return {
                "dok_id": m.get("dok_id"),
                "kammaraktivitet": m.get("kammaraktivitet"),
                "avsnittsrubrik": m.get("avsnittsrubrik"),
                "talare": m.get("talare"),
                "parti": m.get("parti")
            }

        def build_context(hit: dict):
            dok_id = hit.get("dok_id")

            # Lyft alltid upp metadata så prompten alltid har nycklarna
            base = {
                "dok_id": dok_id,
                "kammaraktivitet": hit.get("kammaraktivitet"),
                "avsnittsrubrik": hit.get("avsnittsrubrik"),
                "talare": hit.get("talare"),
                "parti": hit.get("parti"),
            }

            if not dok_id:
                return {**base, "data": "", "speakers": []}
            
            #print("fetching dok_id:", dok_id)
            raw_doc = fetch_document_html(dok_id)
            clean_doc = html_to_text(raw_doc)

            speakers = extract_name_and_parties(clean_doc)
            #print(clean_doc.page_content)
            return {
                **base, 
                "data": clean_doc.page_content, 
                "speakers": speakers}

        self.pipeline = (
            {
                "hit": self.retriever | RunnableLambda(pick_hit),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: {**x, **build_context(x["hit"])})
            | self.prompt
            | self.model
        )

    def ask(self, question):
        """ Main qeuery """
        response = self.pipeline.invoke(question)
        return {"answer": response.content}

        
