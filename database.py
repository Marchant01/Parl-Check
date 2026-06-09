from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    func
)

from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column

from pgvector.sqlalchemy import Vector

from uuid import uuidv7, uuid

DATABASE_URL = "postgresql+psycopg://govcheck_user@localhost:5432/gov_check_db"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def utc_now():
    return datetime.now(timezone.utc)

Base = declarative_base()

class Anforande(Base):
    __tablename__ = 'anforande'
    dok_hangar_id: Mapped[str] = mapped_column(String, primary_key=True)
    dok_id: Mapped[str] = mapped_column(String(50))
    dok_titel: Mapped[str] = mapped_column(String(255))
    dok_rm: Mapped[str] = mapped_column(String(20))
    dok_nummer: Mapped = mapped_column(Integer)
    dok_datum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    avsnittsrubrik: Mapped[str] = mapped_column(String(255))
    underrubrik: Mapped[str] = mapped_column(String(255))
    kammaraktivitet: Mapped[str] = mapped_column(String(250))
    anforande_id: Mapped[str] = mapped_column(String(50))
    anforande_numer: Mapped[int] = mapped_column(Integer)
    talare: Mapped[str] = mapped_column(String(250))
    parti: Mapped[str] = mapped_column(String(50))
    anforandetext: Mapped[str] = mapped_column(Text)
    intressent_id: Mapped[str] = mapped_column(String(50)) # Could be a fk to intressent_id in person
    rel_dok_id: Mapped[str] = mapped_column(String(50))
    replik: Mapped[str] = mapped_column(String(1))
    systemdatum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(768)) # This must exactly match the same dimensional embeddings as the embedding model!

class Person(Base):
    __tablename__ = 'person'
    intressent_id: Mapped[str] = mapped_column(String(20))
    född_år: Mapped[int] = mapped_column(Integer)
    kön: Mapped[str] = mapped_column(String(6))
    efternamn: Mapped[str] = mapped_column(String(50))
    tilltalsnamn: Mapped[str] = mapped_column(String(50))
    sorteringsnamn: Mapped[str] = mapped_column(String(80))
    iort: Mapped[str] = mapped_column(String(40))
    parti: Mapped[str] = mapped_column(String(40))
    valkrets: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(100))
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

class Dokument(Base):
    __tablename__ = 'dokument'
    dok_id: Mapped[str] = mapped_column(String, primary_key=True)
    innehåll: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))
    
    # id: Mapped[uuid.UUID] = mapped_column(server_default=func-uuidv7(monotonic=True))