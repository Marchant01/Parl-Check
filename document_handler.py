import pandas as pd
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

class DocumentHandler:
    def __init__(self, persist_directory="./chroma_langchain_db"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 256, "normalize_embeddings": True},
            show_progress=True
        )
        self.persist_directory = persist_directory
        self.collection_name = "documents"

    def _has_persisted_db(self, vector_store):
        return vector_store._collection.count() > 0

    def build_vector_store(self):
        """Loads CSV files and adds them to a vector store."""
        vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        if self._has_persisted_db(vector_store):
            print("Loading existing Chroma DB")
            return vector_store

        print("Building Chroma DB from scratch")

        anforande_columns = [
            'dok_id', 
            'dok_rm', 
            'dok_nummer', 
            'dok_datum', 
            'avsnittsrubrik', 
            'kammaraktivitet', 
            'anforande_nummer', 
            'talare', 
            'parti', 
            'intressent_id', 
            'rel_dok_id', 
            'replik'
        ]

        votering_columns = [
            'rm',
            'beteckning',
            'votering_id',
            'punkt',
            'namn',
            'intressent_id',
            'parti',
            'valkrets',
            'rost',
            'avser',
            'banknummer',
            'kon',
            'fodd',
            'systemdatum'
        ]

        csv_encoding = "utf-8"

        def load_csv(filename, columns):
            print(f"Loading Document: {filename}")
            return pd.read_csv(
                Path("documents") / filename,
                header=0,
                names=columns,
                dtype={"intressent_id": "string"},
                encoding=csv_encoding,
                chunksize=50000,
            )

        anforande_files = [
            "anforande-202223.csv",
            "anforande-202324.csv",
            "anforande-202425.csv",
        ]
        votering_files = [
            "votering-202223.csv",
            "votering-202324.csv",
            "votering-202425.csv",
            "votering-202526.csv",
        ]

        doc_batch_size = 5000

        def add_in_batches(docs):
            for i in range(0, len(docs), doc_batch_size):
                vector_store.add_documents(docs[i : i + doc_batch_size])

        for filename in votering_files:
            df = load_csv(filename, votering_columns)
            for chunk in df:
                docs = []
                for row in chunk.itertuples(index=False):
                    text = "".join([
                        f"Votering: {row.rm}\n"
                        f"Beteckning: {row.beteckning}\n",
                        f"Votering ID: {row.votering_id}\n",
                        f"Namn: {row.namn}\n",
                        f"Intressent ID: {row.intressent_id}\n",
                        f"Parti: {row.parti}\n",
                        f"Valkrets: {row.valkrets}\n",
                        f"Röst: {row.rost}\n"
                        f"Datum: {row.systemdatum}",
                    ])

                    metadata = {
                        'type': 'votering',
                        'votering_id': str(row.votering_id),
                        'rm': str(row.rm),
                        'beteckning': str(row.beteckning),
                        'namn': str(row.namn),
                        'parti': str(row.parti),
                        'valkrets': str(row.valkrets),
                        'rost': str(row.rost),
                        'datum': str(row.systemdatum),
                    }

                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))

                add_in_batches(docs)

        for filename in anforande_files:
            df = load_csv(filename, anforande_columns)
            for chunk in df:
                docs = []
                for row in chunk.itertuples(index=False):
                    text = "".join([
                        f"Dokument ID: {row.dok_id}\n",
                        f"Avsnittsrubrik: {row.avsnittsrubrik}\n",
                        f"Dokument nummer: {row.dok_nummer}\n",
                        f"Anförande nummer: {row.anforande_nummer}\n",
                        f"Kammaraktivitet {row.kammaraktivitet}\n",
                        f"Datum: {row.dok_datum}\n",
                        f"Talare: {row.talare}\n",
                        f"Parti: {row.parti}\n",
                        f"Intressent ID: {row.intressent_id}\n",
                        f"Rel Dok ID: {row.rel_dok_id}"
                    ])

                    metadata = {
                        'type': 'anföring',
                        'dok_id': str(row.dok_id),
                        'avsnittsrubrik': str(row.avsnittsrubrik),
                        'dok_nummer': str(row.dok_nummer),
                        'anforande_nummer': str(row.anforande_nummer),
                        'kammaraktivitet': str(row.kammaraktivitet),
                        'datum': str(row.dok_datum),
                        'talare': str(row.talare),
                        'parti': str(row.parti),
                        'intressent_id': str(row.intressent_id),
                        'rel_dok_id': str(row.rel_dok_id)
                    }

                    docs.append(Document(
                        page_content=text,
                        metadata=metadata
                    ))
                
                add_in_batches(docs)


        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     add_start_index=True,
        # )
        # all_splits = text_splitter.split_documents(documents)
        # print(len(all_splits))

        # vector_store = Chroma.from_documents(
        #     documents=all_splits,
        #     collection_name=self.collection_name,
        #     embedding=self.embeddings,
        #     persist_directory=self.persist_directory,
        # )

        return vector_store
