from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
    func,
    Uuid
)

from seeder import seed_tables_sql, seed_persons

from sqlalchemy.orm import sessionmaker, declarative_base, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

import uuid

pythonDATABASE_URL = "postgresql+psycopg2://gov_check_user:gov_check_pw@localhost:5432/gov_check_db"
PATH_TO_CSV = "documents/personer"

engine = create_engine(
    DATABASE_URL, 
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def utc_now():
    return datetime.now(timezone.utc)

Base = declarative_base()

class Anforande(Base):
    __tablename__ = "anforande"
    dok_hangar_id: Mapped[str] = mapped_column(String)
    dok_id: Mapped[str] = mapped_column(String(50))
    dok_titel: Mapped[str] = mapped_column(String(255))
    dok_rm: Mapped[str] = mapped_column(String(20))
    dok_nummer: Mapped[int] = mapped_column(Integer)
    dok_datum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    avsnittsrubrik: Mapped[str] = mapped_column(String(255))
    underrubrik: Mapped[str] = mapped_column(String(255))
    kammaraktivitet: Mapped[str] = mapped_column(String(250))
    anforande_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    anforande_numer: Mapped[int] = mapped_column(Integer)
    talare: Mapped[str] = mapped_column(String(250))
    parti: Mapped[str] = mapped_column(String(50))
    anforandetext: Mapped[str] = mapped_column(Text)
    
    intressent_id: Mapped[int] = mapped_column(Integer, ForeignKey("person.intressent_id")) # Could be a fk to intressent_id in person
    person: Mapped["Person"] = relationship("Person", back_populates="anforanden")
    
    rel_dok_id: Mapped[str] = mapped_column(String(50))
    replik: Mapped[str] = mapped_column(String(1))
    systemdatum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(768)) # This must exactly match the same dimensional embeddings as the embedding model!

# This table might remain unnused since the document contents are present in the Anforande table
class Dokument(Base):
    __tablename__ = "dokument"
    dok_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    innehåll: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

class Person(Base):
    __tablename__ = "person"
    fornamn: Mapped[str] = mapped_column(String(80))
    efternamn: Mapped[str] = mapped_column(String(80))
    parti: Mapped[str] = mapped_column(String(40))
    intressent_id: Mapped[int] = mapped_column(Integer, primary_key=True, nullable=False)
    kon: Mapped[str] = mapped_column(String(6))
    fodd: Mapped[int] = mapped_column(Integer)
    valkrets: Mapped[str] = mapped_column(String(50))

    anforanden: Mapped[list["Anforande"]] = relationship("Anforande", back_populates="person")
    votering: Mapped[list["Votering"]] = relationship("Votering", back_populates="person")
    
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

class Votering(Base):
    __tablename__ = "votering"
    rm: Mapped[str] = mapped_column(String(8))
    beteckning: Mapped[str] = mapped_column(String(6))
    hangar_id: Mapped[int] = mapped_column(Integer)
    votering_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    punkt: Mapped[int] = mapped_column(Integer)
    namn: Mapped[str] = mapped_column(String(250))

    intressent_id: Mapped[int] = mapped_column(Integer, ForeignKey("person.intressent_id"))
    person: Mapped["Person"] = relationship("Person", back_populates="votering")

    parti: Mapped[str] = mapped_column(String(4))
    valkrets: Mapped[str] = mapped_column(String(50))
    valkretsnummer: Mapped[int] = mapped_column(Integer)
    iort: Mapped[str] = mapped_column(String)
    rost: Mapped[str] = mapped_column(String(20))
    avser: Mapped[str] = mapped_column(String(10))
    votering: Mapped[str] = mapped_column(String(20))
    banknummer: Mapped[int] = mapped_column(Integer)
    fornamn: Mapped[str] = mapped_column(String(80))
    efternamn: Mapped[str] = mapped_column(String(80))
    kon: Mapped[str] = mapped_column(String(6))
    fodd: Mapped[int] = mapped_column(Integer)
    datum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(768))


def seed_all():
    print("Trying to create database...")
    try:
        Base.metadata.create_all(engine)
        print("Tables created...")

        seed_tables_sql(SessionLocal, "anforanden")
        seed_tables_sql(SessionLocal, "voteringar")

        seed_persons(SessionLocal, PATH_TO_CSV)

        print("Database created and seeded! ")

    except Exception as e:
        print(f"Error seeding{e}")

if __name__ == "__main__":
    seed_all()