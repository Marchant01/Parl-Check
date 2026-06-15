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

DATABASE_URL = "postgresql+psycopg2://gov_check_user@gov_check_pw:5432/gov_check_db"

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
    dok_hangar_id: Mapped[str] = mapped_column(String)
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
    
    intressent_id: Mapped[int] = mapped_column(Integer(50), ForeignKey("person.intressent_id")) # Could be a fk to intressent_id in person
    person: Mapped["Person"] = relationship("Person", back_populates="anforanden")
    
    rel_dok_id: Mapped[str] = mapped_column(String(50))
    replik: Mapped[str] = mapped_column(String(1))
    systemdatum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(768)) # This must exactly match the same dimensional embeddings as the embedding model!

# This table might remain unnused since the document contents are present in the Anforande table
class Dokument(Base):
    __tablename__ = 'dokument'
    dok_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    innehåll: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

class Person(Base):
    __tablename__ = 'person'
    förnamn: Mapped[str] = mapped_column(String(80))
    efternamn: Mapped[str] = mapped_column(String(80))
    parti: Mapped[str] = mapped_column(String(40))
    intressent_id: Mapped[int] = mapped_column(Integer(50), primary_key=True)
    kön: Mapped[str] = mapped_column(String(6))
    född: Mapped[int] = mapped_column(Integer)
    valkrets: Mapped[str] = mapped_column(String(50))
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

    anforanden: Mapped[List["Anforande"]] = relationship("Anforande", back_populates="person")
    votering: Mapped[List["Votering"]] = relationship("Votering", back_populates="person")

class Votering(Base):
    __tablename__ = 'votering'
    rm: Mapped[str] = mapped_column(String(8))
    beteckning: Mapped[str] = mapped_column(String(6))
    hangar_id: Mapped[int] = mapped_column(Integer)
    votering_id: Mapped[str] = mapped_column(String)
    punkt: Mapped[int] = mapped_column(Integer(3))
    namn: Mapped[str] = mapped_column(String(250))

    intressent_id: Mapped[str] = mapped_column(Integer(50), ForeignKey("person.intressent_id"))
    person: Mapped["Person"] = relationship("Person", back_populates="voteringar")

    parti: Mapped[str] = mapped_column(String(4))
    valkrets: Mapped[str] = mapped_column(String(50))
    valkretsnummer: Mapped[int] = mapped_column(Integer(10))
    iort: Mapped[str] = mapped_column(String)
    rost: Mapped[str] = mapped_column(String(20))
    avser: Mapped[str] = mapped_column(String(10))
    votering: Mapped[str] = mapped_column(String(20))
    banknummer: Mapped[int] = mapped_column(Integer(10))
    fornamn: Mapped[str] = mapped_column(String(80))
    efternamn: Mapped[str] = mapped_column(String(80))
    kon: Mapped[str] = mapped_column(String(6))
    fodd: Mapped[int] = mapped_column(Integer(4))
    datum: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding: Mapped[list[float]] = mapped_column(Vector(768))

    # id: Mapped[uuid.UUID] = mapped_column(server_default=func-uuidv7(monotonic=True))