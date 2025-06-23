from sqlalchemy import create_engine, Column, String, Text, DateTime, Boolean, Integer, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mira_tutor.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# User Profile Model - stores Clerk user information
class UserProfile(Base):
    __tablename__ = "user_profiles"
    
    id = Column(String, primary_key=True, index=True)  # Clerk user ID
    email = Column(String, unique=True, index=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    profile_image_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

# Document Model - stores uploaded documents
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)  # Store original name separately
    file_path = Column(String)
    file_size = Column(Integer)  # File size in bytes
    content_type = Column(String)
    text_content = Column(Text)  # Extracted text content
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("UserProfile", back_populates="documents")
    personas = relationship("Persona", back_populates="document", cascade="all, delete-orphan")
    vector_stores = relationship("VectorStore", back_populates="document", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="document")

# Persona Model - stores Tavus personas
class Persona(Base):
    __tablename__ = "personas"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), index=True)
    persona_id = Column(String, unique=True, index=True)  # Tavus persona ID
    persona_name = Column(String)
    system_prompt = Column(Text)
    context = Column(Text)
    tavus_replica_id = Column(String)  # Associated Tavus replica
    created_date = Column(DateTime, default=datetime.utcnow)
    updated_date = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    active = Column(Boolean, default=True)
    
    # Relationships
    document = relationship("Document", back_populates="personas")
    conversations = relationship("Conversation", back_populates="persona", cascade="all, delete-orphan")

# Conversation Model - stores Tavus conversations
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), index=True)
    persona_id = Column(String, ForeignKey("personas.id"), index=True)
    conversation_id = Column(String, unique=True, index=True)  # Tavus conversation ID
    conversation_url = Column(String)
    conversation_name = Column(String)
    status = Column(String, default="active")  # active, ended, expired
    created_date = Column(DateTime, default=datetime.utcnow)
    started_date = Column(DateTime, nullable=True)
    ended_date = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Relationships
    user = relationship("UserProfile", back_populates="conversations")
    persona = relationship("Persona", back_populates="conversations")
    speech_generations = relationship("SpeechGeneration", back_populates="conversation", cascade="all, delete-orphan")

# Chat Message Model - stores all chat interactions
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), index=True)
    document_id = Column(String, ForeignKey("documents.id"), index=True, nullable=True)
    conversation_id = Column(String, nullable=True)  # Tavus conversation ID if from video chat
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    message_type = Column(String, default="text")  # text, voice, video
    timestamp = Column(DateTime, default=datetime.utcnow)
    tokens_used = Column(Integer, nullable=True)  # For cost tracking
    response_time_ms = Column(Integer, nullable=True)  # Performance tracking
    
    # Relationships
    user = relationship("UserProfile", back_populates="chat_messages")
    document = relationship("Document", back_populates="chat_messages")

# Vector Store Model - stores vector database information
class VectorStore(Base):
    __tablename__ = "vector_stores"
    
    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, ForeignKey("documents.id"), index=True)
    vector_store_path = Column(String)
    embedding_model = Column(String, default="gemini-embedding-exp-03-07")
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    total_chunks = Column(Integer, nullable=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="vector_stores")

# Speech Generation Model - tracks video/speech generation
class SpeechGeneration(Base):
    __tablename__ = "speech_generations"
    
    id = Column(String, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    speech_id = Column(String, unique=True, index=True)  # Tavus speech ID
    input_text = Column(Text)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    video_url = Column(String, nullable=True)
    audio_url = Column(String, nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    created_date = Column(DateTime, default=datetime.utcnow)
    completed_date = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="speech_generations")

# User Session Model - tracks user activity
class UserSession(Base):
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), index=True)
    session_start = Column(DateTime, default=datetime.utcnow)
    session_end = Column(DateTime, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    documents_accessed = Column(Integer, default=0)
    messages_sent = Column(Integer, default=0)
    video_sessions = Column(Integer, default=0)

# Usage Analytics Model - for tracking usage patterns
class UsageAnalytics(Base):
    __tablename__ = "usage_analytics"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("user_profiles.id"), index=True)
    date = Column(DateTime, default=datetime.utcnow)
    action_type = Column(String)  # upload, chat, video_start, video_end, etc.
    document_id = Column(String, nullable=True)
    metadata = Column(Text, nullable=True)  # JSON string for additional data

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

# Helper function to create or update user profile from Clerk data
def create_or_update_user_profile(db, user_data):
    """Create or update user profile from Clerk user data"""
    user_id = user_data.get("sub")
    email = user_data.get("email")
    first_name = user_data.get("given_name")
    last_name = user_data.get("family_name")
    profile_image_url = user_data.get("picture")
    
    # Check if user exists
    user_profile = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    
    if user_profile:
        # Update existing user
        user_profile.email = email
        user_profile.first_name = first_name
        user_profile.last_name = last_name
        user_profile.profile_image_url = profile_image_url
        user_profile.updated_at = datetime.utcnow()
        user_profile.last_login = datetime.utcnow()
    else:
        # Create new user
        user_profile = UserProfile(
            id=user_id,
            email=email,
            first_name=first_name,
            last_name=last_name,
            profile_image_url=profile_image_url,
            last_login=datetime.utcnow()
        )
        db.add(user_profile)
    
    db.commit()
    return user_profile

# Helper function to log user activity
def log_user_activity(db, user_id, action_type, document_id=None, metadata=None):
    """Log user activity for analytics"""
    activity = UsageAnalytics(
        user_id=user_id,
        action_type=action_type,
        document_id=document_id,
        metadata=metadata
    )
    db.add(activity)
    db.commit()