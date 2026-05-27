import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# For local development we use SQLite. In production, this can be changed to a PostgreSQL URL.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")

# Fix for Render: Render provides `postgres://` but SQLAlchemy 2.0+ requires `postgresql://`
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL, 
    # check_same_thread=False is needed only for SQLite in FastAPI
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False) # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True) # Stored as JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the tables if they don't exist
# We moved this to lazy-loading in get_db to absolutely guarantee it never blocks uvicorn startup!

_tables_created = False

def get_db():
    global _tables_created
    if not _tables_created:
        Base.metadata.create_all(bind=engine)
        _tables_created = True

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
