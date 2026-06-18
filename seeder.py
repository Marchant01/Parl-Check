import pandas as pd
from pathlib import Path

from sqlalchemy import (
    create_engine,
    text
)
from sqlalchemy.orm import sessionmaker
from pgvector.sqlalchemy import Vector

PERSON_DATA_PATH = "documents/personer/personer"
DOCUMENTS_PATH = Path('documents')

def load_sql_paths(directory: str) -> list[Path]:
    documents_path = Path(DOCUMENTS_PATH) / directory
    return list(documents_path.glob("*.sql"))

def seed_tables_sql(db: sessionmaker, path_to_directory: str):
    paths = load_sql_paths(path_to_directory)
    
    try:
        with db.begin():
            for path in paths:
                print(f"Inserting: {path}")
                sql = path.read_text(encoding="utf-8")
                db.execute(text(sql))
    except Exception as e:
        print(f"Error seeding {path_to_directory} tables: {e}")

def seed_persons(db: sessionmaker, path_to_csv: str):
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
            con=db,
            if_exists="replace",
            index=False
        )
    
    except Exception as e:
        print(f"Error seeding the person table: {e}")
