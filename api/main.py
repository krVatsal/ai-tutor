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

# Import database models and functions
from database import (
    create_tables, get_db, Document, Persona, Conversation, 
    ChatMessage, VectorStore, UserProfile, SpeechGeneration,
    UserSession, UsageAnalytics, VideoCallUsage, engine,
    log_user_activity, check_video_call_constraints, update_video_call_usage, get_video_call_usage_status
)

# Import Google OAuth authentication
from auth import google_auth, verify_token, get_current_user, create_or_update_user_profile

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

# Pydantic models for request bodies
class ChatRequest(BaseModel):
    query: str
    document_name: Optional[str] = None

class SummarizeRequest(BaseModel):
    document_name: str

class DocContext(BaseModel):
    docs: List[str]
    document_name: str

class PersonaRequest(BaseModel):
    document_text: str
    document_name: str

class ConversationRequest(BaseModel):
    persona_id: str
    document_name: str

class SpeechRequest(BaseModel):
    conversation_id: str
    text: str

# Google OAuth models
class GoogleTokenRequest(BaseModel):
    google_token: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

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


# Debug Tavus configuration at startup
print(f"TAVUS Configuration at startup:")
print(f"  TAVUS_API_URL: {TAVUS_API_URL}")
print(f"  TAVUS_API_KEY: {'*' * 10 if TAVUS_API_KEY else 'NOT SET'}")
print(f"  TAVUS_REPLICA_ID: {TAVUS_REPLICA_ID}")
print(f"  Tavus configured: {TAVUS_API_KEY is not None}")

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

def extract_and_chunk_documents(file_path: str) -> List[str]:
    """Extract text from PDF and chunk it for Tavus context"""
    try:
        # Load PDF document
        loader = PyPDFLoader(file_path)
        doc_list = loader.load()
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Larger chunks for Tavus context
            chunk_overlap=200,
        )
        splits = text_splitter.split_documents(doc_list)
        
        # Extract text content from chunks
        chunks = [doc.page_content for doc in splits]
        return chunks
    except Exception as e:
        print(f"Error extracting and chunking document: {e}")
        return []

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

@app.post("/api/create_persona")
async def create_persona(
    ctx: DocContext,
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Create a Tavus persona with document context"""
    try:
        user_id = current_user.id
        
        if not TAVUS_API_KEY:
            raise HTTPException(status_code=500, detail="Tavus API key not configured")
        
        # Extract and chunk documents
        chunks = ctx.docs if ctx.docs else []
        
        # Combine all chunks into context (limit to Tavus context size)
        full_context = "\n\n".join(chunks)
        max_context_length = 8000  # Tavus context limit
        
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        # Create persona payload
        persona_payload = {
            "persona_name": f"Mira - AI Tutor for {ctx.document_name}",
            "system_prompt": f"""You are Mira, an expert AI tutor specializing in helping students understand documents. 

You have access to the following document content:
{full_context}

Your role is to:
- Help students understand the document content
- Answer questions about the material  
- Provide explanations and clarifications
- Break down complex concepts into simpler terms
- Engage in educational discussions about the document
- Use the lookup_doc tool when you need to find specific information

Always be helpful, patient, and encouraging in your responses. Keep your answers concise but informative.""",
            "pipeline_mode": "full",
            "context": full_context,
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

                "perception": {
                    "perception_model": "raven-0"
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": TAVUS_API_KEY
        }
        
        # Make request to Tavus API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/personas",
                json=persona_payload,
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
            
            # Find the document in database
            document = db.query(Document).filter(
                Document.filename == ctx.document_name,
                Document.user_id == user_id
            ).first()
            
            if document:
                # Save persona to database
                db_persona = Persona(
                    id=str(uuid.uuid4()),
                    document_id=document.id,
                    persona_id=persona_response.get("persona_id"),
                    persona_name=persona_payload["persona_name"],
                    system_prompt=persona_payload["system_prompt"],
                    context=full_context,
                    tavus_replica_id=TAVUS_REPLICA_ID
                )
                db.add(db_persona)
                db.commit()
                
                # Log activity
                log_user_activity(db, user_id, "persona_created", document.id)
            
            return persona_response
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Tavus API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Tavus API request failed: {str(e)}")
    except Exception as e:
        print(f"Error creating persona: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create persona: {str(e)}")

@app.post("/api/create_conversation")
async def create_conversation(
    request: ConversationRequest,
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Create a Tavus conversation with persona"""
    try:
        user_id = current_user.id
        
        if not TAVUS_API_KEY:
            raise HTTPException(status_code=500, detail="Tavus API key not configured")
        
        # Check video call constraints
        can_start, message = check_video_call_constraints(db, user_id)
        if not can_start:
            raise HTTPException(status_code=429, detail=message)
        
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
        
        # Create conversation payload with 20-minute limit
        conversation_payload = {
            "replica_id": TAVUS_REPLICA_ID,
            "persona_id": request.persona_id,
            "conversation_name": f"Document Discussion: {request.document_name}",
            "conversational_context": f"The user has uploaded a document titled '{request.document_name}' and wants to discuss it. Help them understand the content, answer questions, and provide educational guidance about the material.",
            "callback_url": f"{os.getenv('APP_URL', 'http://localhost:3000')}/api/tavus-webhook",
            "properties": {
                "max_call_duration": 1200,  # 20 minutes (1200 seconds)
                "participant_left_timeout": 60,
                "language": "english",
                "enable_recording": False
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": TAVUS_API_KEY
        }
        
        # Make request to Tavus API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/conversations",
                json=conversation_payload,
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
                persona_id=request.persona_id,
                conversation_id=conversation_response.get("conversation_id"),
                conversation_url=conversation_response.get("conversation_url"),
                conversation_name=conversation_payload["conversation_name"]
            )
            db.add(db_conversation)
            db.commit()
            
            # Update video call usage (increment call count)
            update_video_call_usage(db, user_id)
            
            # Log activity
            log_user_activity(db, user_id, "video_conversation_created")
            
            return conversation_response
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="Tavus API request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Tavus API request failed: {str(e)}")
    except Exception as e:
        print(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.post("/api/webhook")
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
                    
                    # Update video call usage with actual duration
                    update_video_call_usage(db, conversation.user_id, int(duration))
                
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
                # Handle document lookup using vector search
                query = event.get("parameters", {}).get("query", "")
                
                # Find the conversation and associated document
                conversation = db.query(Conversation).join(Persona).join(Document).filter(
                    Conversation.conversation_id == conversation_id
                ).first()
                
                if conversation and conversation.persona.document:
                    # Use vector search to find relevant content
                    cache_key = f"{conversation.user_id}_{conversation.persona.document.filename}"
                    
                    if cache_key in vector_stores_cache:
                        vector_store = vector_stores_cache[cache_key]
                        relevant_docs = vector_store.similarity_search(query, k=3)
                        
                        # Combine relevant content
                        result_content = "\n\n".join([doc.page_content for doc in relevant_docs])
                        result = f"Found relevant information about '{query}':\n\n{result_content[:1000]}..."
                    else:
                        result = f"Document lookup for '{query}' - vector store not available"
                else:
                    result = f"Document lookup for '{query}' - document not found"
                
                return {"result": result}
        
        return {"success": True}
    
    except Exception as e:
        print(f"Webhook processing error: {e}")
        return {"success": False, "error": str(e)}

async def create_tavus_persona_with_document(document_text: str, document_name: str, document_id: str, user_id: str, db: Session) -> Dict[str, Any]:
    """Create a Tavus persona with document context - integrated version"""
    
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

Always be helpful, patient, and encouraging in your responses. Keep your answers concise but informative.""",
        "context": document_text,
        "default_replica_id": TAVUS_REPLICA_ID
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": TAVUS_API_KEY
    }
    
    try:
        print(f"Making request to: {TAVUS_API_URL}/personas")
        print(f"Headers: {headers}")
        print(f"Persona data keys: {list(persona_data.keys())}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TAVUS_API_URL}/personas",
                json=persona_data,
                headers=headers
            )
            
            print(f"Tavus API response status: {response.status_code}")
            print(f"Tavus API response body: {response.text}")
            
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
    current_user: UserProfile = Depends(get_current_user)
):
    """Upload and process PDF document with Tavus persona creation"""
    document_id = None
    file_path = None
    
    try:
        user_id = current_user.id
        print(f"DEBUG: Starting upload for user: {user_id}, file: {file.filename}")
        
        # Validate file exists and has content
        if not file or not file.filename:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "No file provided",
                    "message": "Please select a file to upload",
                    "code": "NO_FILE",
                    "suggestion": "Select a PDF file using the file picker or drag and drop area"
                }
            )
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Invalid file type",
                    "message": f"Only PDF files are supported. You uploaded: {file.filename.split('.')[-1] if '.' in file.filename else 'unknown'}",
                    "code": "INVALID_FILE_TYPE"
                }
            )
        
        # Validate file size (10MB limit)
        try:
            content = await file.read()
            file_size = len(content)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Failed to read file",
                    "message": "There was an error reading your file. Please try again with a different file.",
                    "code": "FILE_READ_ERROR"
                }
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Empty file",
                    "message": "The uploaded file appears to be empty. Please check your file and try again.",
                    "code": "EMPTY_FILE"
                }
            )
        
        if file_size > 10 * 1024 * 1024:  # 10MB
            file_size_mb = round(file_size / (1024 * 1024), 2)
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "File too large",
                    "message": f"File size ({file_size_mb}MB) exceeds the 10MB limit. Please use a smaller file.",
                    "code": "FILE_TOO_LARGE"
                }
            )
        
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
        
        # Save file to disk with error handling
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            print(f"File saved successfully to: {file_path}")
        except OSError as e:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "File save failed",
                    "message": "Failed to save the file to server. Please try again.",
                    "code": "FILE_SAVE_ERROR"
                }
            )
        except Exception as e:
            print(f"Unexpected error saving file: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "File save failed",
                    "message": "An unexpected error occurred while saving the file.",
                    "code": "UNEXPECTED_SAVE_ERROR"
                }
            )
        
        # Save document to database with user_id
        try:
            db_document = Document(
                id=document_id,
                user_id=user_id,
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
            print(f"Document record created in database: {document_id}")
        except Exception as e:
            print(f"Database error: {e}")
            # Clean up the file if database operation failed
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database error",
                    "message": "Failed to save document information to database. Please try again.",
                    "code": "DATABASE_ERROR"
                }
            )
        
        # Process document and create vector store with comprehensive error handling
        try:
            print(f"Starting document processing for: {file.filename}")
            vector_store, doc_list, vector_db_path = process_document(file_path, document_id, user_id, db)
            print(f"Document processing completed successfully")
        except Exception as e:
            print(f"Document processing error: {e}")
            # Update document status to failed
            try:
                db_document.processing_status = "failed"
                db_document.error_message = str(e)
                db.commit()
            except:
                pass
            
            # Determine specific error type
            error_message = str(e).lower()
            if "pdf" in error_message or "corrupt" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "PDF processing failed",
                        "message": "The PDF file appears to be corrupted or invalid. Please try with a different PDF file.",
                        "code": "INVALID_PDF"
                    }
                )
            elif "memory" in error_message or "size" in error_message:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Document too complex",
                        "message": "The document is too large or complex to process. Please try with a simpler PDF file.",
                        "code": "DOCUMENT_TOO_COMPLEX"
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Processing failed",
                        "message": "Failed to process the document. Please try again or contact support if the issue persists.",
                        "code": "PROCESSING_ERROR"
                    }
                )
        
        # Extract text content with error handling
        try:
            document_text = extract_text_from_documents(doc_list)
            if not document_text or len(document_text.strip()) < 50:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "No text content",
                        "message": "Could not extract readable text from the PDF. Please ensure the PDF contains text (not just images).",
                        "code": "NO_TEXT_CONTENT"
                    }
                )
            print(f"Extracted {len(document_text)} characters of text")
        except HTTPException:
            raise
        except Exception as e:
            print(f"Text extraction error: {e}")
            try:
                db_document.processing_status = "failed"
                db_document.error_message = f"Text extraction failed: {str(e)}"
                db.commit()
            except:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Text extraction failed",
                    "message": "Failed to extract text from the PDF. Please try with a different file.",
                    "code": "TEXT_EXTRACTION_ERROR"
                }
            )
        
        # Update document with text content and mark as processed
        try:
            db_document.text_content = document_text
            db_document.processed = True
            db_document.processing_status = "completed"
            db.commit()
            print(f"Document marked as completed in database")
        except Exception as e:
            print(f"Database update error: {e}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Database update failed",
                    "message": "Document was processed but failed to update database. Please try again.",
                    "code": "DATABASE_UPDATE_ERROR"
                }
            )
        
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
        
        # Create Tavus persona with document context (optional, non-blocking)
        persona_id = None
        persona_error = None
        print(f"Starting persona creation section")
        
        try:
            if TAVUS_API_KEY and TAVUS_REPLICA_ID:
                print(f"Creating Tavus persona for document: {file.filename}")
                persona_response = await create_tavus_persona_with_document(document_text, file.filename, document_id, user_id, db)
                persona_id = persona_response.get("persona_id")
                print(f"Tavus persona created successfully: {persona_id}")
            else:
                print("Tavus API not fully configured - skipping persona creation")
                persona_error = "Tavus API not configured"
        except Exception as e:
            print(f"Persona creation failed: {e}")
            persona_error = str(e)
            # Continue without persona - document processing was successful
        
        # Prepare success response
        response_data = {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "persona_id": persona_id,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "text_length": len(document_text),
            "message": "Document processed and vectorized successfully"
        }
        
        if persona_id:
            response_data["message"] += " with AI tutor persona created"
        elif persona_error:
            response_data["persona_warning"] = f"Document processed successfully, but persona creation failed: {persona_error}"
            response_data["message"] += " (video chat may not be available)"
        
        return response_data
    
    except HTTPException as http_error:
        # Re-raise HTTP exceptions (these are expected errors with proper error details)
        raise http_error
    except Exception as e:
        print(f"Unexpected upload error: {e}")
        import traceback
        traceback.print_exc()
        
        # Clean up resources if something went wrong
        try:
            if 'document_id' in locals() and document_id:
                # Update document status to failed if it exists
                db_document = db.query(Document).filter(Document.id == document_id).first()
                if db_document:
                    db_document.processing_status = "failed"
                    db_document.error_message = str(e)
                    db.commit()
                
                # Clean up file if it was created
                if 'file_path' in locals() and file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        print(f"Cleaned up file: {file_path}")
                    except:
                        pass
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")
        
        # Return a generic error for unexpected issues
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Unexpected error",
                "message": "An unexpected error occurred while processing your document. Please try again or contact support if the issue persists.",
                "code": "UNEXPECTED_ERROR"
            }
        )

@app.post("/api/generate-speech")
async def generate_speech(
    request: SpeechRequest, 
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Generate speech/video from text"""
    try:
        user_id = current_user.id
        
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

@app.get("/api/speech-status/{speech_id}")
async def get_speech_status(
    speech_id: str,
    current_user: UserProfile = Depends(get_current_user),
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

@app.post("/api/chat")
async def chat_with_document(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Chat with AI about the document using vector search"""
    try:
        user_id = current_user.id
        
        # Validate request data first
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Empty query",
                    "message": "Please enter a question or message",
                    "code": "EMPTY_QUERY"
                }
            )
            
        if len(request.query) > 2000:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Query too long",
                    "message": f"Your message is too long ({len(request.query)} characters). Please keep it under 2000 characters.",
                    "code": "QUERY_TOO_LONG"
                }
            )
        
        if not request.document_name or not request.document_name.strip():
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "No document selected",
                    "message": "Please upload and select a document first before chatting",
                    "code": "NO_DOCUMENT"
                }
            )
        
        if not embeddings:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "AI service unavailable",
                    "message": "The AI embedding service is currently unavailable. Please try again later.",
                    "code": "EMBEDDINGS_UNAVAILABLE"
                }
            )
            
        if not llm:
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "AI service unavailable",
                    "message": "The AI language model is currently unavailable. Please try again later.",
                    "code": "LLM_UNAVAILABLE"
                }
            )

        print(f"Chat request for user {user_id}, document: {request.document_name}, query: {request.query[:100]}...")
        
        # Get document for this user
        document = db.query(Document).filter(
            Document.filename == request.document_name,
            Document.user_id == user_id
        ).first()
        
        if not document:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Document not found",
                    "message": f"The document '{request.document_name}' was not found in your uploads. Please upload it first.",
                    "code": "DOCUMENT_NOT_FOUND"
                }
            )
        
        if not document.processed or document.processing_status != "completed":
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "Document not ready",
                    "message": f"The document '{request.document_name}' is still being processed. Please wait a moment and try again.",
                    "code": "DOCUMENT_NOT_PROCESSED"
                }
            )
        
        # Save user message to database with user_id
        try:
            user_message = ChatMessage(
                user_id=user_id,
                document_id=document.id,
                role="user",
                content=request.query,
                message_type="text"
            )
            db.add(user_message)
        except Exception as e:
            print(f"Error saving user message: {e}")
            # Continue processing even if message save fails
        
        # Check if vector store is in cache
        cache_key = f"{user_id}_{request.document_name}"
        if cache_key not in vector_stores_cache:
            # Try to load from database for this user
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == document.id).first()
            if not vector_store_record:
                raise HTTPException(
                    status_code=400, 
                    detail={
                        "error": "Document not vectorized",
                        "message": f"The document '{request.document_name}' hasn't been properly processed for chat. Please re-upload the document.",
                        "code": "NO_VECTOR_STORE"
                    }
                )
            
            if not os.path.exists(vector_store_record.vector_store_path):
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "error": "Vector store missing",
                        "message": "The document's search index is missing. Please re-upload the document.",
                        "code": "VECTOR_STORE_MISSING"
                    }
                )
            
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
                print(f"Loaded vector store for {request.document_name}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                raise HTTPException(
                    status_code=500, 
                    detail={
                        "error": "Failed to load document index",
                        "message": "There was an error loading the document's search index. Please re-upload the document.",
                        "code": "VECTOR_STORE_LOAD_ERROR"
                    }
                )
        
        # Get vector store for the document
        vector_store = vector_stores_cache[cache_key]
        
        # Create retrieval QA chain with LLM
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 3})
            )
        except Exception as e:
            print(f"Error creating QA chain: {e}")
            raise HTTPException(
                status_code=500, 
                detail={
                    "error": "AI setup failed",
                    "message": "Failed to set up the AI chat system. Please try again in a moment.",
                    "code": "QA_CHAIN_ERROR"
                }
            )
        
        # Get AI response using the LLM
        start_time = datetime.utcnow()
        try:
            ai_response = qa_chain.invoke(request.query)
            
            # Extract text from response
            response_text = ai_response.get('result', ai_response) if isinstance(ai_response, dict) else str(ai_response)
            
            if not response_text or response_text.strip() == "":
                response_text = "I couldn't find relevant information to answer your question. Could you try rephrasing it or asking about a different aspect of the document?"
                
        except Exception as e:
            print(f"LLM processing error: {e}")
            # Fallback to similarity search
            try:
                relevant_docs = vector_store.similarity_search(request.query, k=3)
                if relevant_docs:
                    context = "\n\n".join([doc.page_content for doc in relevant_docs])
                    response_text = f"I found some relevant information about your question:\n\n{context[:500]}..."
                else:
                    response_text = "I couldn't find relevant information to answer your question. Could you try asking about a different aspect of the document?"
            except Exception as fallback_error:
                print(f"Fallback search error: {fallback_error}")
                response_text = "I'm having trouble processing your question right now. Please try again in a moment."
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Save assistant message to database with user_id
        try:
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
        except Exception as e:
            print(f"Error saving assistant message: {e}")
            # Continue and return response even if save fails
        
        return {"response": response_text}
    
    except HTTPException as http_error:
        # Re-raise HTTP exceptions (these have proper error details)
        raise http_error
    except Exception as e:
        print(f"Unexpected chat error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Unexpected error",
                "message": "An unexpected error occurred while processing your message. Please try again.",
                "code": "UNEXPECTED_CHAT_ERROR"
            }
        )

@app.post("/api/summarize")
async def summarize_document(
    request: SummarizeRequest, 
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Generate a summary of the document"""
    try:
        user_id = current_user.id
        
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
    current_user: UserProfile = Depends(get_current_user)
):
    """Get chat history for a document"""
    try:
        user_id = current_user.id
        
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
    current_user: UserProfile = Depends(get_current_user)
):
    """Get all uploaded documents for the authenticated user"""
    try:
        user_id = current_user.id
        
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
    current_user: UserProfile = Depends(get_current_user)
):
    """List all available documents and their vector stores for the authenticated user"""
    try:
        user_id = current_user.id
        
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

@app.get("/api/personas")
async def get_personas(
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Get all created personas for the authenticated user"""
    try:
        user_id = current_user.id
        
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
    current_user: UserProfile = Depends(get_current_user)
):
    """Get user profile information"""
    try:
        user_id = current_user.id
        
        # User profile is already available as current_user
        
        # Get user statistics
        total_documents = db.query(Document).filter(Document.user_id == user_id).count()
        total_messages = db.query(ChatMessage).filter(ChatMessage.user_id == user_id).count()
        total_conversations = db.query(Conversation).filter(Conversation.user_id == user_id).count()
        
        return {
            "user_profile": {
                "id": current_user.id,
                "email": current_user.email,
                "first_name": current_user.first_name,
                "last_name": current_user.last_name,
                "profile_image_url": current_user.profile_image_url,
                "created_at": current_user.created_at.isoformat(),
                "last_login": current_user.last_login.isoformat() if current_user.last_login else None
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
    current_user: UserProfile = Depends(get_current_user)
):
    """Get user activity analytics"""
    try:
        user_id = current_user.id
        
        # Get recent activity
        recent_activity = db.query(UsageAnalytics).filter(
            UsageAnalytics.user_id == user_id        ).order_by(UsageAnalytics.date.desc()).limit(50).all()
        
        return {
            "recent_activity": [
                {
                    "action_type": activity.action_type,
                    "date": activity.date.isoformat(),
                    "document_id": activity.document_id,
                    "additional_data": activity.additional_data
                }
                for activity in recent_activity
            ]
        }
    except Exception as e:
        print(f"User analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")

@app.get("/api/video-call-usage")
async def get_video_call_usage(
    db: Session = Depends(get_db),
    current_user: UserProfile = Depends(get_current_user)
):
    """Get current video call usage status for the authenticated user"""
    try:
        user_id = current_user.id
        
        usage_status = get_video_call_usage_status(db, user_id)
        
        return {
            "usage": usage_status,
            "can_start_call": usage_status["calls_remaining"] > 0 and usage_status["total_duration_remaining_seconds"] > 0
        }
    except Exception as e:
        print(f"Video call usage error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get video call usage: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mira AI Tutor API is running with Tavus integration and Google OAuth authentication"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embeddings_available": embeddings is not None,
        "tavus_configured": TAVUS_API_KEY is not None,
        "database": "connected",
        "auth": "google_oauth_enabled"
    }

# Google OAuth Authentication Endpoints
@app.post("/auth/google", response_model=LoginResponse)
async def google_login(token_request: GoogleTokenRequest, db: Session = Depends(get_db)):
    """Login with Google OAuth token"""
    try:
        # Verify Google token and get user info
        user_info = google_auth.verify_google_token(token_request.google_token)
        
        # Create or update user profile
        user_profile = create_or_update_user_profile(db, user_info)
        
        # Create JWT access token
        access_token = google_auth.create_access_token(user_info)
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            user={
                "id": user_profile.id,
                "email": user_profile.email,
                "first_name": user_profile.first_name,
                "last_name": user_profile.last_name,
                "profile_image_url": user_profile.profile_image_url
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/auth/me")
async def get_current_user_info(current_user: UserProfile = Depends(get_current_user)):
    """Get current authenticated user information"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "first_name": current_user.first_name,
        "last_name": current_user.last_name,
        "profile_image_url": current_user.profile_image_url,
        "created_at": current_user.created_at,
        "last_login": current_user.last_login
    }
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
