import os
import re
import torch
import json as pyjson

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from api_caller import get_document, get_voting

class Chatbot:
    def __init__(self, api_key, persist_directory="./chroma_langchain_db"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["GOOGLE_API_KEY"] = api_key
        
        #Local embedding model that will run either run with cuda(GPU) or PCU
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": 64,
                "normalize_embeddings": True
            },
            #show_progress=True
        )
        
        #LLM model through API key
        self.model = init_chat_model("google_genai:gemini-2.5-flash-lite")
        
        self.vector_store = Chroma(
            collection_name='documents',
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        #Retriever that will search in the vector store
        self.retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 60})

        #Second retriever that will search in fetched documents from the gov. API
        self.text_store = Chroma(
            collection_name='document_text',
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

        #Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )

        
        bs_transformer = BeautifulSoupTransformer()

        def fetch_document_html(dok_id: str) -> Document:
            html = get_document(dok_id)
            return Document(
                page_content=html,
                metadata={"dok_id": dok_id, "source": f"https://data.riksdagen.se/dokument/{dok_id}/html"}
            )
        
        def fetch_voting_document(votering_id: str) -> Document:
            voting_doc = get_voting(votering_id)
            
            if isinstance(voting_doc, str):
                voting_doc = pyjson.loads(voting_doc)

            dok_id = voting_doc.get("dok_id")

            return Document(
                page_content="",
                metadata={
                    "type": "votering",
                    "votering_id": votering_id,
                    "dok_id": dok_id,
                    "source": f"https://data.riksdagen.se/votering/{votering_id}/json",
                },
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


        def pick_hit(docs):
            if not docs:
                return {}
            m = docs[0].metadata or {}
            # print("retriever hit:", m)
            return {
                "dok_id": m.get("dok_id"),
                "votering_id": m.get("votering_id"),
                "kammaraktivitet": m.get("kammaraktivitet"),
                "avsnittsrubrik": m.get("avsnittsrubrik"),
                "talare": m.get("talare"),
                "parti": m.get("parti")
            }

        def index_doc_if_missing(dok_id: str, cleaned_text: str):
            try:
                existing = self.text_store._collection.count(where={"dok_id": dok_id})
                if existing and existing > 0:
                    return
            except Exception:
                pass

            base_doc = Document(page_content=cleaned_text, metadata={"dok_id": dok_id})
            chunks = self.text_splitter.split_documents([base_doc])
            for c in chunks:
                c.metadata["dok_id"] = dok_id
                # print(c)
            
            self.text_store.add_documents(chunks)

        def build_context(hit: dict, question: str):
            dok_id = hit.get("dok_id")
            votering_id = hit.get("votering_id")

            # Lyft alltid upp metadata så prompten alltid har nycklarna
            base = {
                "dok_id": dok_id,
                "kammaraktivitet": hit.get("kammaraktivitet"),
                "avsnittsrubrik": hit.get("avsnittsrubrik"),
                "talare": hit.get("talare"),
                "parti": hit.get("parti"),
            }

            if not dok_id and votering_id:
                voting_doc = fetch_voting_document(votering_id)
                print(dok_id)
                dok_id = voting_doc.metadata.get("dok_id")
                base["dok_id"] = dok_id

            if not dok_id:
                return {**base, "data": "", "speakers": []}
            
            # print("fetching dok_id:", dok_id)
            raw_doc = fetch_document_html(dok_id)
            clean_doc = html_to_text(raw_doc)

            speakers = extract_name_and_parties(clean_doc)
            # print(clean_doc.page_content)

            index_doc_if_missing(dok_id, clean_doc.page_content)

            chunks = self.text_store.similarity_search(
                query=question,
                k=5,
                filter={"dok_id": dok_id},
            )

            data = "\n\n---\n\n".join(d.page_content for d in chunks)

            # Capping the data for gemini
            data = data[:12000]

            return {**base, "data": clean_doc.page_content, "speakers": speakers}

        #Prompt for LLM
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

        self.pipeline = (
            {
                "hit": self.retriever | RunnableLambda(pick_hit),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(lambda x: {**x, **build_context(x["hit"], x["question"])})
            | self.prompt
            | self.model
        )

    def ask(self, question):
        """ Main qeuery """
        response = self.pipeline.invoke(question)
        return {"answer": response.content}
        
