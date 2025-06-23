from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any
import os
import shutil
import uuid
import httpx
import json
import pickle
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import jwt
from jwt import PyJWKClient

# Import database models and functions
from database import (
    create_tables, get_db, Document, Persona, Conversation, 
    ChatMessage, VectorStore, UserProfile, SpeechGeneration,
    UserSession, UsageAnalytics, engine, create_or_update_user_profile,
    log_user_activity
)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Clerk configuration
CLERK_PEM_PUBLIC_KEY = os.getenv("CLERK_PEM_PUBLIC_KEY")
CLERK_JWKS_URL = "https://api.clerk.com/v1/jwks"

# JWT verification for Clerk
def verify_clerk_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        token = authorization.split(" ")[1] if authorization.startswith("Bearer ") else authorization
        
        # Get JWKS from Clerk
        jwks_client = PyJWKClient(CLERK_JWKS_URL)
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={"verify_aud": False}  # Clerk tokens don't always have aud
        )
        
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {str(e)}")

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    query: str
    document_name: Optional[str] = None

class SummarizeRequest(BaseModel):
    document_name: str

# Create directories for storing uploads and vector DB
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Initialize database
create_tables()

# Initialize Gemini embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# In-memory cache for vector stores (loaded from DB on startup)
vector_stores_cache = {}

# Legacy compatibility - keeping for backward compatibility
documents = {}
vector_stores = {}

# Tavus API configuration
TAVUS_API_URL = os.getenv("TAVUS_API_URL", "https://tavusapi.com/v2")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
TAVUS_REPLICA_ID = os.getenv("TAVUS_REPLICA_ID")
TAVUS_VOICE_ID = os.getenv("TAVUS_VOICE_ID")

# Pydantic models
class PersonaRequest(BaseModel):
    document_text: str
    document_name: str

class ConversationRequest(BaseModel):
    persona_id: str
    document_name: str

class SpeechRequest(BaseModel):
    conversation_id: str
    text: str

def load_vector_stores_from_db(db: Session):
    """Load vector stores from database on startup"""
    if not embeddings:
        print("Skipping vector store loading - embeddings not available")
        return
        
    vector_stores = db.query(VectorStore).all()
    for vs in vector_stores:
        try:
            if os.path.exists(vs.vector_store_path):
                vector_store = FAISS.load_local(
                    vs.vector_store_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                # Get document info
                doc = db.query(Document).filter(Document.id == vs.document_id).first()
                if doc:
                    # Use user-specific cache key
                    cache_key = f"{doc.user_id}_{doc.filename}"
                    vector_stores_cache[cache_key] = vector_store
                    print(f"Loaded vector store for: {doc.filename} (User: {doc.user_id})")
        except Exception as e:
            print(f"Failed to load vector store {vs.id}: {e}")

def process_document(file_path: str, document_id: str, user_id: str, db: Session):
    """Process PDF document and create vector store"""
    # Load PDF document
    loader = PyPDFLoader(file_path)
    doc_list = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(doc_list)
    
    # Create vector store
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save vector store to disk
    vector_store_path = f"vector_db/{document_id}"
    vector_store.save_local(vector_store_path)
    
    # Save vector store info to database    
    db_vector_store = VectorStore(
        id=str(uuid.uuid4()),
        document_id=document_id,
        vector_store_path=vector_store_path,
        total_chunks=len(splits)
    )
    db.add(db_vector_store)
    db.commit()
    
    # Log activity
    log_user_activity(db, user_id, "document_processed", document_id)
    
    return vector_store, doc_list, vector_store_path

def extract_text_from_documents(documents_list):
    """Extract text content from document list"""
    text_content = ""
    for doc in documents_list:
        text_content += doc.page_content + "\n\n"
    return text_content.strip()

async def create_tavus_persona(document_text: str, document_name: str, document_id: str, user_id: str, db: Session) -> Dict[str, Any]:
    """Create a Tavus persona with document context"""
    
    if not TAVUS_API_KEY:
        raise HTTPException(status_code=500, detail="Tavus API key not configured")
    
    # Truncate document text if too long (Tavus has limits)
    max_context_length = 8000
    if len(document_text) > max_context_length:
        document_text = document_text[:max_context_length] + "..."
    
    persona_data = {
        "persona_name": f"Mira - AI Tutor for {document_name}",
        "system_prompt": f"""You are Mira, an expert AI tutor specializing in helping students understand documents. 

You have access to the following document content:
{document_text}

Your role is to:
- Help students understand the document content
- Answer questions about the material  
- Provide explanations and clarifications
- Break down complex concepts into simpler terms
- Engage in educational discussions about the document
- Use the lookup_doc tool when you need to find specific information

Always be helpful, patient, and encouraging in your responses. Keep your answers concise but informative.""",
        "pipeline_mode": "full",
        "context": document_text,
        "default_replica_id": TAVUS_REPLICA_ID,
        "layers": {
            "llm": {
                "model": "tavus-llama-3-8b-instruct",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_doc",
                            "description": "Search and retrieve specific information from the document",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query to find relevant document content"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    }
                ]
            },
            "tts": {
                "tts_engine": "cartesia",
                "voice_id": TAVUS_VOICE_ID or "default"
            },
            "perception": {
                "perception_model": "raven-0"
            }
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TAVUS_API_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/personas",
                json=persona_data,
                headers=headers
            )
            
            if response.status_code != 200:
                error_detail = f"Tavus API error: {response.status_code} - {response.text}"
                print(error_detail)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )
            
            persona_response = response.json()
            
            # Save persona to database
            db_persona = Persona(
                id=str(uuid.uuid4()),
                document_id=document_id,
                persona_id=persona_response.get("persona_id"),
                persona_name=persona_data["persona_name"],
                system_prompt=persona_data["system_prompt"],
                context=document_text,
                tavus_replica_id=TAVUS_REPLICA_ID
            )
            db.add(db_persona)
            db.commit()
            
            # Log activity
            log_user_activity(db, user_id, "persona_created", document_id)
            
            return persona_response
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Tavus API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Tavus API request failed: {str(e)}")

async def create_tavus_conversation(persona_id: str, document_name: str, user_id: str, db: Session) -> Dict[str, Any]:
    """Create a Tavus conversation with persona"""
    
    if not TAVUS_API_KEY:
        raise HTTPException(status_code=500, detail="Tavus API key not configured")
    
    conversation_data = {
        "replica_id": TAVUS_REPLICA_ID,
        "persona_id": persona_id,
        "conversation_name": f"Document Discussion: {document_name}",
        "conversational_context": f"The user has uploaded a document titled '{document_name}' and wants to discuss it. Help them understand the content, answer questions, and provide educational guidance about the material.",
        "callback_url": f"{os.getenv('APP_URL', 'http://localhost:3000')}/api/tavusWebhook",
        "properties": {
            "max_call_duration": 300,  # 5 minutes
            "participant_left_timeout": 60,
            "language": "english",
            "enable_recording": False
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TAVUS_API_KEY
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/conversations",
                json=conversation_data,
                headers=headers
            )
            
            if response.status_code != 200:
                error_detail = f"Tavus API error: {response.status_code} - {response.text}"
                print(error_detail)
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )
            
            conversation_response = response.json()
            
            # Save conversation to database
            db_conversation = Conversation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                persona_id=persona_id,
                conversation_id=conversation_response.get("conversation_id"),
                conversation_url=conversation_response.get("conversation_url"),
                conversation_name=conversation_data["conversation_name"]
            )
            db.add(db_conversation)
            db.commit()
            
            # Log activity
            log_user_activity(db, user_id, "video_conversation_created")
            
            return conversation_response
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Tavus API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Tavus API request failed: {str(e)}")

# Load existing vector stores on startup
def startup_event():
    """Load vector stores from database on startup"""
    try:
        from database import SessionLocal
        db = SessionLocal()
        try:
            load_vector_stores_from_db(db)
            print("Vector stores loaded from database on startup")
        finally:
            db.close()
    except Exception as e:
        print(f"Error loading vector stores on startup: {e}")

# Call startup function
startup_event()

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Upload and process PDF document"""
    try:
        user_id = user_data.get("sub")  # Clerk user ID
        
        # Create or update user profile
        user_profile = create_or_update_user_profile(db, user_data)
        
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Validate file size (10MB limit)
        file_size = 0
        content = await file.read()
        file_size = len(content)
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        
        # Reset file pointer
        await file.seek(0)
        
        # Check if document already exists for this user
        existing_doc = db.query(Document).filter(
            Document.filename == file.filename,
            Document.user_id == user_id
        ).first()
        
        if existing_doc:
            # Load existing vector store
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == existing_doc.id).first()
            if vector_store_record and os.path.exists(vector_store_record.vector_store_path) and embeddings:
                try:
                    cache_key = f"{user_id}_{file.filename}"
                    vector_stores_cache[cache_key] = FAISS.load_local(
                        vector_store_record.vector_store_path, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"Error loading existing vector store: {e}")
            
            # Get existing persona
            existing_persona = db.query(Persona).filter(Persona.document_id == existing_doc.id).first()
            
            # Log activity
            log_user_activity(db, user_id, "document_reloaded", existing_doc.id)
            
            return {
                "success": True,
                "document_id": existing_doc.id,
                "filename": file.filename,
                "persona_id": existing_persona.persona_id if existing_persona else None,
                "message": "Document already exists and loaded successfully"
            }
        
        # Generate unique ID and save file
        document_id = str(uuid.uuid4())
        file_path = f"uploads/{document_id}.pdf"
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Save document to database with user_id
        db_document = Document(
            id=document_id,
            user_id=user_id,  # Associate with user
            filename=file.filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            content_type=file.content_type,
            processing_status="processing",
            processed=False
        )
        db.add(db_document)
        db.commit()
        
        # Log activity
        log_user_activity(db, user_id, "document_uploaded", document_id)
        
        # Process document and create vector store
        vector_store, doc_list, vector_db_path = process_document(file_path, document_id, user_id, db)
        
        # Extract text content
        document_text = extract_text_from_documents(doc_list)
        
        # Update document with text content and mark as processed
        db_document.text_content = document_text
        db_document.processed = True
        db_document.processing_status = "completed"
        db.commit()
        
        # Add to cache with user-specific key
        cache_key = f"{user_id}_{file.filename}"
        vector_stores_cache[cache_key] = vector_store
        
        # Store document info and vector store (legacy compatibility)
        documents[file.filename] = {
            "id": document_id,
            "path": file_path,
            "vector_db_path": vector_db_path,
            "type": file.content_type,
            "text": document_text,
            "user_id": user_id
        }
        vector_stores[file.filename] = vector_store
        
        # Create Tavus persona with document context
        persona_id = None
        try:
            if TAVUS_API_KEY:
                persona_response = await create_tavus_persona(document_text, file.filename, document_id, user_id, db)
                persona_id = persona_response.get("persona_id")
        except Exception as persona_error:
            print(f"Persona creation failed: {persona_error}")
            # Continue without persona - document processing was successful
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "persona_id": persona_id,
            "file_size": file_size,
            "message": "Document processed and vectorized successfully" + 
                      (" with persona created" if persona_id else " (persona creation failed)")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        # Update document status to failed if it exists
        if 'document_id' in locals():
            db_document = db.query(Document).filter(Document.id == document_id).first()
            if db_document:
                db_document.processing_status = "failed"
                db_document.error_message = str(e)
                db.commit()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/create-conversation")
async def create_conversation_endpoint(
    request: ConversationRequest, 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Create a Tavus conversation with persona"""
    try:
        user_id = user_data.get("sub")
        
        # Check if conversation already exists for this persona and user
        existing_conversation = db.query(Conversation).filter(
            Conversation.persona_id == request.persona_id,
            Conversation.user_id == user_id,
            Conversation.status == "active"
        ).first()
        
        if existing_conversation:
            return {
                "conversation_id": existing_conversation.conversation_id,
                "conversation_url": existing_conversation.conversation_url,
                "message": "Using existing active conversation"
            }
        
        conversation_response = await create_tavus_conversation(request.persona_id, request.document_name, user_id, db)
        return conversation_response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Conversation creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.post("/generate-speech")
async def generate_speech(
    request: SpeechRequest, 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Generate speech/video from text"""
    try:
        user_id = user_data.get("sub")
        
        if not TAVUS_API_KEY:
            raise HTTPException(status_code=500, detail="Tavus API key not configured")
        
        speech_data = {
            "conversation_id": request.conversation_id,
            "text": request.text
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": TAVUS_API_KEY
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/speech",
                json=speech_data,
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to generate speech: {response.text}"
                )
            
            speech_response = response.json()
            
            # Save speech generation to database
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == request.conversation_id
            ).first()
            
            if conversation:
                speech_generation = SpeechGeneration(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation.id,
                    speech_id=speech_response.get("speech_id"),
                    input_text=request.text,
                    status="pending"
                )
                db.add(speech_generation)
                db.commit()
                
                # Log activity
                log_user_activity(db, user_id, "speech_generated")
            
            return speech_response
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Speech generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@app.get("/speech-status/{speech_id}")
async def get_speech_status(
    speech_id: str,
    user_data: dict = Depends(verify_clerk_token),
    db: Session = Depends(get_db)
):
    """Get speech generation status"""
    try:
        if not TAVUS_API_KEY:
            raise HTTPException(status_code=500, detail="Tavus API key not configured")
        
        headers = {
            "x-api-key": TAVUS_API_KEY
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{TAVUS_API_URL}/speech/{speech_id}",
                headers=headers
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get speech status: {response.text}"
                )
            
            speech_status = response.json()
            
            # Update speech generation status in database
            speech_generation = db.query(SpeechGeneration).filter(
                SpeechGeneration.speech_id == speech_id
            ).first()
            
            if speech_generation:
                speech_generation.status = speech_status.get("status", "unknown")
                speech_generation.video_url = speech_status.get("video_url")
                speech_generation.audio_url = speech_status.get("audio_url")
                speech_generation.duration_seconds = speech_status.get("duration_seconds")
                
                if speech_status.get("status") == "completed":
                    speech_generation.completed_date = datetime.utcnow()
                elif speech_status.get("status") == "failed":
                    speech_generation.error_message = speech_status.get("error_message")
                
                db.commit()
            
            return speech_status
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Speech status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get speech status: {str(e)}")

@app.post("/tavus-webhook")
async def tavus_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Tavus webhook events"""
    try:
        event = await request.json()
        
        print(f"Tavus webhook received: {event.get('type')} - {event.get('conversation_id')}")
        
        event_type = event.get("type")
        conversation_id = event.get("conversation_id")
        
        if event_type == "conversation.started":
            print(f"Conversation started: {conversation_id}")
            # Update conversation status
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id
            ).first()
            if conversation:
                conversation.started_date = datetime.utcnow()
                conversation.status = "active"
                db.commit()
                
                # Log activity
                log_user_activity(db, conversation.user_id, "video_conversation_started")
            
        elif event_type == "conversation.ended":
            print(f"Conversation ended: {conversation_id}")
            # Mark conversation as ended in database
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id
            ).first()
            if conversation:
                conversation.ended_date = datetime.utcnow()
                conversation.status = "ended"
                
                # Calculate duration if started
                if conversation.started_date:
                    duration = (datetime.utcnow() - conversation.started_date).total_seconds()
                    conversation.duration_seconds = int(duration)
                
                db.commit()
                
                # Log activity
                log_user_activity(db, conversation.user_id, "video_conversation_ended")
            
        elif event_type == "utterance":
            utterance = event.get('utterance')
            print(f"User utterance: {utterance}")
            
            # Save user message to database
            if conversation_id and utterance:
                # Find the conversation to get user and document info
                conversation = db.query(Conversation).join(Persona).join(Document).filter(
                    Conversation.conversation_id == conversation_id
                ).first()
                
                if conversation:
                    chat_message = ChatMessage(
                        user_id=conversation.user_id,
                        document_id=conversation.persona.document_id,
                        conversation_id=conversation_id,
                        role="user",
                        content=utterance,
                        message_type="voice"
                    )
                    db.add(chat_message)
                    db.commit()
                    
                    # Log activity
                    log_user_activity(db, conversation.user_id, "voice_message_sent")
            
        elif event_type == "tool_call":
            tool_name = event.get("tool_name")
            print(f"Tool call requested: {tool_name}")
            
            if tool_name == "lookup_doc":
                # Handle document lookup
                query = event.get("parameters", {}).get("query", "")
                
                # Find relevant document content using vector search
                # This would need the conversation's associated document
                result = f"Found information related to: {query}"
                
                return {"result": result}
        
        return {"success": True}
    
    except Exception as e:
        print(f"Webhook processing error: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/chat")
async def chat_with_document(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Chat with AI about the document using vector search"""
    try:
        user_id = user_data.get("sub")
        
        # Create or update user profile
        user_profile = create_or_update_user_profile(db, user_data)
        
        if not request.document_name:
            raise HTTPException(status_code=400, detail="Document name is required")
        
        if not embeddings:
            raise HTTPException(status_code=500, detail="Embeddings not available")
            
        if not llm:
            raise HTTPException(status_code=500, detail="Language model not available")

        print(f"Chat request for document: {request.document_name}, query: {request.query}")
        
        # Validate request data
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        if len(request.query) > 2000:
            raise HTTPException(status_code=400, detail="Query too long (max 2000 characters)")
        
        # Get document for this user
        document = db.query(Document).filter(
            Document.filename == request.document_name,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=400, detail="Document not found")
        
        # Save user message to database with user_id
        user_message = ChatMessage(
            user_id=user_id,
            document_id=document.id,
            role="user",
            content=request.query,
            message_type="text"
        )
        db.add(user_message)
        
        # Check if vector store is in cache
        cache_key = f"{user_id}_{request.document_name}"
        if cache_key not in vector_stores_cache:
            # Try to load from database for this user
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == document.id).first()
            if not vector_store_record or not os.path.exists(vector_store_record.vector_store_path):
                raise HTTPException(status_code=400, detail="Vector store not found")
            
            # Load vector store
            try:
                vector_stores_cache[cache_key] = FAISS.load_local(
                    vector_store_record.vector_store_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                # Update last accessed
                vector_store_record.last_accessed = datetime.utcnow()
                db.commit()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")
        
        # Get vector store for the document
        vector_store = vector_stores_cache[cache_key]
        
        # Create retrieval QA chain with LLM
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        
        # Get AI response using the LLM
        start_time = datetime.utcnow()
        try:
            ai_response = qa_chain.invoke(request.query)
            
            # Extract text from response
            response_text = ai_response.get('result', ai_response) if isinstance(ai_response, dict) else str(ai_response)
        except Exception as e:
            print(f"LLM processing error: {e}")
            # Fallback to similarity search
            relevant_docs = vector_store.similarity_search(request.query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            response_text = f"Based on the document content, here's what I found relevant to your question '{request.query}':\n\n{context[:500]}..."
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Save assistant message to database with user_id
        assistant_message = ChatMessage(
            user_id=user_id,
            document_id=document.id,
            role="assistant",
            content=response_text,
            message_type="text",
            response_time_ms=int(response_time)
        )
        db.add(assistant_message)
        db.commit()
        
        # Log activity
        log_user_activity(db, user_id, "chat_message_sent", document.id)
        
        return {"response": response_text}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

@app.post("/api/summarize")
async def summarize_document(
    request: SummarizeRequest, 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Generate a summary of the document"""
    try:
        user_id = user_data.get("sub")
        
        print(f"Summarize request for document: {request.document_name}")
        
        # Get document for this user
        document = db.query(Document).filter(
            Document.filename == request.document_name,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=400, detail="Document not found")
        
        # Check if vector store exists in cache
        cache_key = f"{user_id}_{request.document_name}"
        if cache_key not in vector_stores_cache:
            # Try to load from database for this user
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == document.id).first()
            if not vector_store_record or not os.path.exists(vector_store_record.vector_store_path):
                raise HTTPException(status_code=400, detail="Vector store not found")
            
            # Load vector store
            vector_stores_cache[cache_key] = FAISS.load_local(
                vector_store_record.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        
        vector_store = vector_stores_cache[cache_key]
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        )
        
        # Generate summary
        summary_query = "Please provide a comprehensive summary of this document, highlighting the key points, main topics, and important information."
        summary_response = qa_chain.invoke(summary_query)
        
        # Extract text from response
        summary_text = summary_response.get('result', summary_response) if isinstance(summary_response, dict) else str(summary_response)
        
        # Save messages to database with user_id
        user_message = ChatMessage(
            user_id=user_id,
            document_id=document.id,
            role="user",
            content="Can you summarize this document?",
            message_type="text"
        )
        db.add(user_message)
        
        assistant_message = ChatMessage(
            user_id=user_id,
            document_id=document.id,
            role="assistant",
            content=summary_text,
            message_type="text"
        )
        db.add(assistant_message)
        db.commit()
        
        # Log activity
        log_user_activity(db, user_id, "document_summarized", document.id)
        
        return {"summary": summary_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize document: {str(e)}")

@app.get("/api/chat-history/{document_name}")
async def get_chat_history(
    document_name: str, 
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Get chat history for a document"""
    try:
        user_id = user_data.get("sub")
        
        # Get document for this user
        document = db.query(Document).filter(
            Document.filename == document_name,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise HTTPException(status_code=400, detail="Document not found")
        
        messages = db.query(ChatMessage).filter(
            ChatMessage.document_id == document.id,
            ChatMessage.user_id == user_id
        ).order_by(ChatMessage.timestamp).all()
        
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "message_type": msg.message_type
                }
                for msg in messages
            ]
        }
    except Exception as e:
        print(f"Chat history error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@app.get("/documents")
async def get_documents(
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Get all uploaded documents for the authenticated user"""
    try:
        user_id = user_data.get("sub")
        
        documents = db.query(Document).filter(
            Document.user_id == user_id
        ).order_by(Document.upload_date.desc()).all()
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "original_filename": doc.original_filename,
                    "file_size": doc.file_size,
                    "upload_date": doc.upload_date.isoformat(),
                    "processed": doc.processed,
                    "processing_status": doc.processing_status
                }
                for doc in documents
            ]
        }
    except Exception as e:
        print(f"Documents error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.get("/api/documents")
async def list_documents(
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """List all available documents and their vector stores for the authenticated user"""
    try:
        user_id = user_data.get("sub")
        
        # Get documents from database for this user
        documents_from_db = db.query(Document).filter(
            Document.user_id == user_id
        ).all()
        
        # Get vector stores from memory cache for this user
        user_vector_stores = [key for key in vector_stores_cache.keys() if key.startswith(f"{user_id}_")]
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "original_filename": doc.original_filename,
                    "file_path": doc.file_path,
                    "file_size": doc.file_size,
                    "upload_date": doc.upload_date.isoformat() if doc.upload_date else None,
                    "processed": doc.processed,
                    "processing_status": doc.processing_status,
                    "has_vector_store": f"{user_id}_{doc.filename}" in vector_stores_cache
                }
                for doc in documents_from_db
            ],
            "vector_stores_in_memory": len(user_vector_stores),
            "total_documents": len(documents_from_db),
            "total_vector_stores": len(user_vector_stores)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/personas")
async def get_personas(
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Get all created personas for the authenticated user"""
    try:
        user_id = user_data.get("sub")
        
        # Get personas for documents owned by this user
        personas = db.query(Persona).join(Document).filter(
            Document.user_id == user_id,
            Persona.active == True
        ).order_by(Persona.created_date.desc()).all()
        
        return {
            "personas": [
                {
                    "id": persona.id,
                    "persona_id": persona.persona_id,
                    "document_name": persona.document.filename,
                    "persona_name": persona.persona_name,
                    "created_date": persona.created_date.isoformat(),
                    "tavus_replica_id": persona.tavus_replica_id
                }
                for persona in personas
            ]
        }
    except Exception as e:
        print(f"Personas error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get personas: {str(e)}")

@app.get("/api/user-profile")
async def get_user_profile(
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Get user profile information"""
    try:
        user_id = user_data.get("sub")
        
        # Create or update user profile
        user_profile = create_or_update_user_profile(db, user_data)
        
        # Get user statistics
        total_documents = db.query(Document).filter(Document.user_id == user_id).count()
        total_messages = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).count()
        total_conversations = db.query(Conversation).filter(Conversation.user_id == user_id).count()
        
        return {
            "user_profile": {
                "id": user_profile.id,
                "email": user_profile.email,
                "first_name": user_profile.first_name,
                "last_name": user_profile.last_name,
                "profile_image_url": user_profile.profile_image_url,
                "created_at": user_profile.created_at.isoformat(),
                "last_login": user_profile.last_login.isoformat() if user_profile.last_login else None
            },
            "statistics": {
                "total_documents": total_documents,
                "total_messages": total_messages,
                "total_conversations": total_conversations
            }
        }
    except Exception as e:
        print(f"User profile error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@app.get("/api/user-analytics")
async def get_user_analytics(
    db: Session = Depends(get_db),
    user_data: dict = Depends(verify_clerk_token)
):
    """Get user activity analytics"""
    try:
        user_id = user_data.get("sub")
        
        # Get recent activity
        recent_activity = db.query(UsageAnalytics).filter(
            UsageAnalytics.user_id == user_id
        ).order_by(UsageAnalytics.date.desc()).limit(50).all()
        
        return {
            "recent_activity": [
                {
                    "action_type": activity.action_type,
                    "date": activity.date.isoformat(),
                    "document_id": activity.document_id,
                    "metadata": activity.metadata
                }
                for activity in recent_activity
            ]
        }
    except Exception as e:
        print(f"User analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mira AI Tutor API is running with database persistence and Clerk authentication"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embeddings_available": embeddings is not None,
        "tavus_configured": TAVUS_API_KEY is not None,
        "database": "connected",
        "auth": "clerk_enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)