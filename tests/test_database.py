import pytest
import psycopg2
import uuid7
from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy.exc import OperationalError
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    text
)
# from database import Dokument

# import os

# from dotenv import load_dotenv

# load_dotenv()

# DB_URL = os.getenv('')

DATABASE_URL = "postgresql+psycopg2://gov_check_user:gov_check_pw@postgres-test/gov_check_db_test"

@pytest.fixture(scope="session")
def engine():
    engine = create_engine(DATABASE_URL)
    try:
        yield engine
    finally:
        engine.dispose()

@pytest.fixture(scope="function")
def db_session(engine):
    with engine.connect() as connection:
        with connection.begin() as transaction:
            session = Session(bind=connection)
            yield session
            session.close()
            transaction.rollback()

def test_connection(engine):
    with engine.connect() as conn:
        assert conn.scalar(text("SELECT 1"))

# def seed_test_database(SessionLocal: session):
#     result = (
#         Dokument.insert("").returning(dokument.c.)
#     )
#     pass
