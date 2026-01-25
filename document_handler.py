import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

class DocumentHandler:
    def __init__(self, persist_directory="./chroma_langchain_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "batch_size": 256,
                "normalize_embeddings": True
            },
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
            return pd.read_csv(
                Path("documents") / filename,
                header=0,
                names=columns,
                dtype={"intressent_id": "string"},
                encoding=csv_encoding,
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

        documents = []

        for filename in votering_files:
            df = load_csv(filename, votering_columns)
            for _, row in df.iterrows():
                text = "".join([
                    f"Votering: {row.get('rm')} - {row.get('beteckning')}\n",
                    f"Votering ID: {row.get('votering_id')}\n",
                    f"Namn: {row.get('namn')}\n",
                    f"Intressent ID: {row.get('intressent_id')}\n",
                    f"Parti: {row.get('parti')}\n",
                    f"Valkrets: {row.get('valkrets')}\n",
                    f"Röst: {row.get('rost')}\n"
                    f"Datum: {row.get('systemdatum')}",
                ])
                metadata = {
                    'type': 'votering',
                    'votering_id': str(row.get('votering_id')),
                    'rm': str(row.get('rm')),
                    'beteckning': str(row.get('beteckning')),
                    'namn': str(row.get('namn')),
                    'parti': str(row.get('parti')),
                    'valkrets': str(row.get('valkrets')),
                    'rost': str(row.get('rost')),
                    'datum': str(row.get('systemdatum')),
                }
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

        for filename in anforande_files:
            df = load_csv(filename, anforande_columns)
            for _, row in df.iterrows():
                text = "".join([
                    f"Dokument ID: {row.get('dok_id')}\n",
                    f"Avsnittsrubrik: {row.get('avsnittsrubrik')}\n",
                    f"Dokument nummer: {row.get('dok_nummer')}\n",
                    f"Anförande nummer: {row.get('anforande_nummer')}\n",
                    f"Kammaraktivitet {row.get('kammaraktivitet')}\n",
                    f"Datum: {row.get('dok_datum')}\n",
                    f"Talare: {row.get('talare')}\n",
                    f"Parti: {row.get('parti')}\n",
                    f"Intressent ID: {row.get('intressent_id')}\n",
                    f"Rel Dok ID: {row.get('rel_dok_id')}"
                ])

                metadata = {
                    'type': 'anföring',
                    'dok_id': str(row.get('dok_id')),
                    'avsnittsrubrik': str(row.get('avsnittsrubrik')),
                    'dok_nummer': str(row.get('dok_nummer')),
                    'anforande_nummer': str(row.get('anforande_nummer')),
                    'kammaraktivitet': str(row.get('kammaraktivitet')),
                    'datum': str(row.get('dok_datum')),
                    'talare': str(row.get('talare')),
                    'parti': str(row.get('parti')),
                    'intressent_id': str(row.get('intressent_id')),
                    'rel_dok_id': str(row.get('rel_dok_id'))
                }

                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(documents)

        vector_store = Chroma.from_documents(
            documents=all_splits,
            collection_name=self.collection_name,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

        return vector_store
