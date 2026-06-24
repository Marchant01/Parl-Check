import pandas as pd
import torch
import os
from pathlib import Path
import re

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    text
)
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector
from psycopg2.extras import Json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CONNECTION_STRING = os.getenv("DATABASE_URL")

PERSON_DATA_PATH = "documents/personer/personer"
DOCUMENTS_PATH = Path('documents')

BATCH_SIZE = 256
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def load_sql_paths(directory: str) -> list[Path]:
    documents_path = Path(DOCUMENTS_PATH) / directory
    return list(documents_path.glob("*.sql"))

def seed_tables_sql(sess: Session, path_to_directory: str):
    try:
        with sess.begin():
            for path in load_sql_paths(path_to_directory):
                print(f"Processing: {path.name}")
                raw_sql = path.read_text(encoding="utf-8")
                inserts = re.findall(r'INSERT INTO .+?^(?=INSERT|\Z)', raw_sql, re.DOTALL | re.MULTILINE)

                for insert in inserts:
                    sess.execute(text(insert).execution_options(no_parameters=True))

    except Exception as e:
        print(f"Error seeding {path_to_directory} tables: {e}")

def seed_persons(sess: Session, engine, path_to_csv: str):
    try:
        with sess.begin():
            sess.execute(text("DROP TABLE IF EXISTS person CASCADE"))
            print("Dropped old ´person´ table (cascade).")

    except Exception as e:
        print(f"Couldn't drop `person` table: {e}")

    columns = {
        "Förnamn": "fornamn",
        "Efternamn": "efternamn",
        "Parti": "parti",
        "Id": "intressent_id",
        "Kön": "kon",
        "Född": "fodd",
        "Valkrets": "valktrets",
    }

    person_df = pd.read_csv(path_to_csv, header=0, usecols=columns)
    person_df_copy = person_df.rename(columns=columns).drop_duplicates(subset=["intressent_id"]).copy()

    try:
        print("Seeding person table")
        person_df_copy.to_sql(
            name="person",
            con=engine,
            if_exists="replace",
            index=False
        )
    
    except Exception as e:
        print(f"Error seeding the person table: {e}")


def seed_embeddings(sess: Session, embed_columns: list[str], id_column: str, table: str, embed_column: str = "embedding"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": BATCH_SIZE, "normalize_embeddings": True},
        show_progress=True
    )
    col_names = [id_column] + embed_columns
    cols = ", ".join(f'"{col}"' for col in embed_columns)

    try:
        result = sess.execute(text(f"SELECT {cols} FROM {table} WHERE embedding IS NULL"))
        rows = result.fetchall()
    
        if not rows:
            print(f"No rows in {table}")
            return
        
        print(f"Generating embeddings for {len(rows)} rows in '{table}'...")
        
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]

            texts = [
                " ". join(str(row[col_names.index(col)]) for col in columns if row[col_names.index(col)] is not None)
                for row in batch
            ]

            vectors = embedding_model.embed_documents(texts)

            for row, vector in zip(batch, vectors):
                sess.execute(
                    text(f'UPDATE {table} SET {embed_column} = :vec WHERE "{id_column}" = :id'),
                    {"vex": str(vector), "id": row[col_names.index(id_column)]}
                )
            
            sess.commit()
            print(f"  Committed batch {i // BATCH_SIZE + 1} ({min(i + BATCH_SIZE, len(rows))}/{len(rows)})")

        print(f"Done seeding embeddings for '{table}'.")

    except Exception as e:
        print(f"Error generating embeddings for '{table}': {e}")