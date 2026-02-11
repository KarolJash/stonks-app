from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os
from dotenv import load_dotenv

load_dotenv()

class Base(DeclarativeBase):
    pass

engine = create_engine(os.environ.get("DATABASE_URL"))
SessionLocal = sessionmaker(bind=engine)