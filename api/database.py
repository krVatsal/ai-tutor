from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mira_tutor.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Database Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_path = Column(String)
    content_type = Column(String)
    text_content = Column(Text)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    user_id = Column(String, index=True)  # Clerk user ID

class Persona(Base):
    __tablename__ = "personas"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    document_name = Column(String)
    persona_id = Column(String, unique=True, index=True)  # Tavus persona ID
    persona_name = Column(String)
    system_prompt = Column(Text)
    context = Column(Text)
    created_date = Column(DateTime, default=datetime.utcnow)
    active = Column(Boolean, default=True)

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    persona_id = Column(String, index=True)
    conversation_id = Column(String, unique=True, index=True)  # Tavus conversation ID
    conversation_url = Column(String)
    document_name = Column(String)
    created_date = Column(DateTime, default=datetime.utcnow)
    ended_date = Column(DateTime, nullable=True)
    active = Column(Boolean, default=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_name = Column(String, index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    conversation_id = Column(String, nullable=True)
    user_id = Column(String, index=True)  # Clerk user ID

class VectorStore(Base):
    __tablename__ = "vector_stores"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    vector_store_path = Column(String)
    created_date = Column(DateTime, default=datetime.utcnow)

# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()