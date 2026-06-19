import pandas as pd
from pathlib import Path
import re

from sqlalchemy import (
    create_engine,
    text
)
from sqlalchemy.orm import Session
from pgvector.sqlalchemy import Vector

PERSON_DATA_PATH = "documents/personer/personer"
DOCUMENTS_PATH = Path('documents')

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
