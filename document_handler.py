import pandas as pd
import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

class DocumentHandler:
    def __init__(self, api_key, persist_directory="./chroma_langchain_db"):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={
                "batch_size": 256,
                "normalize_embeddings": True
            },
            show_progress=True
        )
        self.persist_directory=persist_directory

    def build_vector_store(self):
        """ Loads CSV files and adds them to a vector store """

        vector_store = Chroma(
            collection_name='documents',
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        
        count = vector_store._collection.count()
        print(f"Chroma collection contains {count} documents")

        if count > 0:
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
            'kön',
            'fodd',
            'system_datum'
        ]

        anforande_202223 = pd.read_csv(
            'documents/anforande-202223.csv',
            header=0,
            names=anforande_columns,
            dtype={'intressent_id': 'string'}
        )

        anforande_202324 = pd.read_csv(
            'documents/anforande-202324.csv',
            header=0,
            names=anforande_columns,
            dtype={'intressent_id': 'string'}
        )

        anforande_202425 = pd.read_csv(
            'documents/anforande-202425.csv',
            header=0,
            names=anforande_columns,
            dtype={'intressent_id': 'string'}
        )


        votering_202223 = pd.read_csv(
            'documents/votering-202223.csv',
            header=0,
            names=votering_columns,
            dtype={'intressent_id': 'string'}
        )

        votering_202324 = pd.read_csv(
            'documents/votering-202324.csv',
            header=0,
            names=votering_columns,
            dtype={'intressent_id': 'string'}
        )

        votering_202425 = pd.read_csv(
            'documents/votering-202425.csv',
            header=0,
            names=votering_columns,
            dtype={'intressent_id': 'string'}
        )

        votering_202526 = pd.read_csv(
            'documents/votering-202526.csv',
            header=0,
            names=votering_columns,
            dtype={'intressent_id': 'string'}
        )

        anforande_dataset = [
            {"df": anforande_202223, 'source': 'anforande_202223'},
            {"df": anforande_202324, 'source': 'anforande_202324'},
            {"df": anforande_202425, 'source': 'anforande_202425'}
        ]

        votering_dataset = [
            {"df": votering_202223, 'source': 'votering_202223'},
            {"df": votering_202324, 'source': 'votering_202324'},
            {"df": votering_202425, 'source': 'votering_202425'},
            {"df": votering_202526, 'source': 'votering_202526'}
        ]

        documents = []

        for dataset in votering_dataset:
            df = dataset['df']

            for _, row in df.iterrows():
                text = "".join([
                    f"Votering: {row.get('rm')} - {row.get('beteckning')}\n",
                    f"Datum: {row.get('system_datum')}\n",
                    f"Votering ID: {row.get('votering_id')}\n",
                    f"Namn: {row.get('namn')}\n"
                    f"Intressent ID: {row.get('intressent_id')}\n",
                    f"Parti: {row.get('parti')}\n",
                    f"Valkrets: {row.get('valkrets')}\n"
                    f"\nRöst: {row.get('rost')}"
                ])
                metadata = {
                    'type': 'votering',
                    'votering_id': str(row.get('votering_id')),
                    'rm': str(row.get('rm')),
                    'beteckning': str(row.get('beteckning')),
                    'datum': str(row.get('system_datum')),
                    'rost': str(row.get('rost'))
                }
                documents.append(Document(
                    page_content=text,
                    metadata=metadata
                ))

        for dataset in anforande_dataset:
            df = dataset['df']

            for _, row in df.iterrows():
                text = "".join([
                    f"Dokument ID: {row.get('dok_id')}\n",
                    f"Dokument nummer: {row.get('dok_nummer')}\n",
                    f"Anförande nummer: {row.get('anforande_nummer')}\n",
                    f"Rubrik: {row.get('avsnittsrubrik')}\n",
                    f"Datum: {row.get('dok_datum')}\n",
                    f"Kammaraktivitet {row.get('kammaraktivitet')}\n"
                    f"Talare: {row.get('talare')}\n"
                    f"Parti: {row.get('parti')}\n",
                    f"Intressent ID: {row.get('intressent_id')}\n",
                    f"Rel Dok ID: {row.get('rel_dok_id')}"
                ])

                metadata = {
                    'type': 'anföring',
                    'dokument_id': str(row.get('dok_id')),
                    'dok_nummer': str(row.get('dok_nummer')),
                    'anforande_nummer': str(row.get('anforande_nummer')),
                    'avsnittsrubrik': str(row.get('avsnittsrubrik')),
                    'datum': str(row.get('dok_datum')),
                    'kammaraktivitet': str(row.get('kammaraktivitet')),
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
            collection_name='documents',
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )

        return vector_store