from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
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
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Import database models and functions
from database import (
    create_tables, get_db, Document, Persona, Conversation, 
    ChatMessage, VectorStore, engine
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

# Create database tables on startup
create_tables()

# Create directories for storing uploads and vector DB
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# In-memory cache for vector stores (loaded from DB on startup)
vector_stores_cache = {}

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

class ChatRequest(BaseModel):
    query: str
    document_name: Optional[str] = None

def load_vector_stores_from_db(db: Session):
    """Load vector stores from database on startup"""
    vector_stores = db.query(VectorStore).all()
    for vs in vector_stores:
        try:
            if os.path.exists(vs.vector_store_path):
                vector_store = FAISS.load_local(vs.vector_store_path, embeddings, allow_dangerous_deserialization=True)
                # Get document info
                doc = db.query(Document).filter(Document.id == vs.document_id).first()
                if doc:
                    vector_stores_cache[doc.filename] = vector_store
        except Exception as e:
            print(f"Failed to load vector store {vs.id}: {e}")

def process_document(file_path: str, document_id: str, db: Session):
    """Process PDF document and create vector store"""
    # Load PDF document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(documents)
    
    # Create and save vector store
    vector_store = FAISS.from_documents(splits, embeddings)
    
    # Save vector store to disk
    vector_store_path = f"vector_db/{document_id}"
    vector_store.save_local(vector_store_path)
    
    # Save vector store info to database
    db_vector_store = VectorStore(
        id=str(uuid.uuid4()),
        document_id=document_id,
        vector_store_path=vector_store_path
    )
    db.add(db_vector_store)
    db.commit()
    
    return vector_store, documents

def extract_text_from_documents(documents_list):
    """Extract text content from document list"""
    text_content = ""
    for doc in documents_list:
        text_content += doc.page_content + "\n\n"
    return text_content.strip()

async def create_tavus_persona(document_text: str, document_name: str, document_id: str, db: Session) -> Dict[str, Any]:
    """Create a Tavus persona with document context"""
    
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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TAVUS_API_URL}/personas",
            json=persona_data,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create persona: {response.text}"
            )
        
        persona_response = response.json()
        
        # Save persona to database
        db_persona = Persona(
            id=str(uuid.uuid4()),
            document_id=document_id,
            document_name=document_name,
            persona_id=persona_response.get("persona_id"),
            persona_name=persona_data["persona_name"],
            system_prompt=persona_data["system_prompt"],
            context=document_text
        )
        db.add(db_persona)
        db.commit()
        
        return persona_response

async def create_tavus_conversation(persona_id: str, document_name: str, db: Session) -> Dict[str, Any]:
    """Create a Tavus conversation with persona"""
    
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
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TAVUS_API_URL}/conversations",
            json=conversation_data,
            headers=headers,
            timeout=30.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create conversation: {response.text}"
            )
        
        conversation_response = response.json()
        
        # Save conversation to database
        db_conversation = Conversation(
            id=str(uuid.uuid4()),
            persona_id=persona_id,
            conversation_id=conversation_response.get("conversation_id"),
            conversation_url=conversation_response.get("conversation_url"),
            document_name=document_name
        )
        db.add(db_conversation)
        db.commit()
        
        return conversation_response

@app.on_event("startup")
async def startup_event():
    """Load vector stores from database on startup"""
    db = next(get_db())
    load_vector_stores_from_db(db)
    print("Vector stores loaded from database")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Upload and process PDF document"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check if document already exists
        existing_doc = db.query(Document).filter(Document.filename == file.filename).first()
        if existing_doc:
            # Load existing vector store
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == existing_doc.id).first()
            if vector_store_record and os.path.exists(vector_store_record.vector_store_path):
                vector_stores_cache[file.filename] = FAISS.load_local(
                    vector_store_record.vector_store_path, 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
            
            # Get existing persona
            existing_persona = db.query(Persona).filter(Persona.document_id == existing_doc.id).first()
            
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
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document and create vector store
        vector_store, doc_list = process_document(file_path, document_id, db)
        
        # Extract text content for persona creation
        document_text = extract_text_from_documents(doc_list)
        
        # Save document to database
        db_document = Document(
            id=document_id,
            filename=file.filename,
            file_path=file_path,
            content_type=file.content_type,
            text_content=document_text,
            processed=True
        )
        db.add(db_document)
        db.commit()
        
        # Cache vector store
        vector_stores_cache[file.filename] = vector_store
        
        # Create Tavus persona with document context
        try:
            persona_response = await create_tavus_persona(document_text, file.filename, document_id, db)
            
            return {
                "success": True,
                "document_id": document_id,
                "filename": file.filename,
                "persona_id": persona_response.get("persona_id"),
                "message": "Document processed, vectorized, and persona created successfully"
            }
        except Exception as persona_error:
            print(f"Persona creation failed: {persona_error}")
            # Still return success for document processing
            return {
                "success": True,
                "document_id": document_id,
                "filename": file.filename,
                "persona_id": None,
                "message": "Document processed and vectorized successfully (persona creation failed)"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/create-conversation")
async def create_conversation_endpoint(request: ConversationRequest, db: Session = Depends(get_db)):
    """Create a Tavus conversation with persona"""
    try:
        # Check if conversation already exists for this persona
        existing_conversation = db.query(Conversation).filter(
            Conversation.persona_id == request.persona_id,
            Conversation.active == True
        ).first()
        
        if existing_conversation:
            return {
                "conversation_id": existing_conversation.conversation_id,
                "conversation_url": existing_conversation.conversation_url,
                "message": "Using existing active conversation"
            }
        
        conversation_response = await create_tavus_conversation(request.persona_id, request.document_name, db)
        return conversation_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest, db: Session = Depends(get_db)):
    """Generate speech/video from text"""
    try:
        speech_data = {
            "conversation_id": request.conversation_id,
            "text": request.text
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": TAVUS_API_KEY
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{TAVUS_API_URL}/speech",
                json=speech_data,
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to generate speech: {response.text}"
                )
            
            return response.json()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate speech: {str(e)}")

@app.get("/speech-status/{speech_id}")
async def get_speech_status(speech_id: str):
    """Get speech generation status"""
    try:
        headers = {
            "x-api-key": TAVUS_API_KEY
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{TAVUS_API_URL}/speech/{speech_id}",
                headers=headers,
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to get speech status: {response.text}"
                )
            
            return response.json()
    
    except Exception as e:
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
            
        elif event_type == "conversation.ended":
            print(f"Conversation ended: {conversation_id}")
            # Mark conversation as ended in database
            conversation = db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id
            ).first()
            if conversation:
                conversation.ended_date = datetime.utcnow()
                conversation.active = False
                db.commit()
            
        elif event_type == "utterance":
            utterance = event.get('utterance')
            print(f"User utterance: {utterance}")
            
            # Save user message to database
            if conversation_id and utterance:
                chat_message = ChatMessage(
                    document_name="video_chat",
                    role="user",
                    content=utterance,
                    conversation_id=conversation_id
                )
                db.add(chat_message)
                db.commit()
            
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
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@app.post("/chat")
async def chat_with_document(request: ChatRequest, db: Session = Depends(get_db)):
    """Chat with AI about the document using vector search"""
    try:
        if not request.document_name:
            raise HTTPException(status_code=400, detail="Document name is required")
        
        # Save user message to database
        user_message = ChatMessage(
            document_name=request.document_name,
            role="user",
            content=request.query
        )
        db.add(user_message)
        
        # Check if vector store is in cache
        if request.document_name not in vector_stores_cache:
            # Try to load from database
            document = db.query(Document).filter(Document.filename == request.document_name).first()
            if not document:
                raise HTTPException(status_code=400, detail="Document not found")
            
            vector_store_record = db.query(VectorStore).filter(VectorStore.document_id == document.id).first()
            if not vector_store_record or not os.path.exists(vector_store_record.vector_store_path):
                raise HTTPException(status_code=400, detail="Vector store not found")
            
            # Load vector store
            vector_stores_cache[request.document_name] = FAISS.load_local(
                vector_store_record.vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        
        # Get vector store for the document
        vector_store = vector_stores_cache[request.document_name]
        
        # Search for relevant content
        relevant_docs = vector_store.similarity_search(request.query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Simple response generation (you could integrate with your LLM here)
        response_text = f"Based on the document content, here's what I found relevant to your question '{request.query}':\n\n{context[:500]}..."
        
        # Save assistant message to database
        assistant_message = ChatMessage(
            document_name=request.document_name,
            role="assistant",
            content=response_text
        )
        db.add(assistant_message)
        db.commit()
        
        return {"response": response_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

@app.get("/chat-history/{document_name}")
async def get_chat_history(document_name: str, db: Session = Depends(get_db)):
    """Get chat history for a document"""
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.document_name == document_name
        ).order_by(ChatMessage.timestamp).all()
        
        return {
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chat history: {str(e)}")

@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    """Get all uploaded documents"""
    try:
        documents = db.query(Document).order_by(Document.upload_date.desc()).all()
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "upload_date": doc.upload_date.isoformat(),
                    "processed": doc.processed
                }
                for doc in documents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@app.get("/personas")
async def get_personas(db: Session = Depends(get_db)):
    """Get all created personas"""
    try:
        personas = db.query(Persona).filter(Persona.active == True).order_by(Persona.created_date.desc()).all()
        
        return {
            "personas": [
                {
                    "id": persona.id,
                    "persona_id": persona.persona_id,
                    "document_name": persona.document_name,
                    "persona_name": persona.persona_name,
                    "created_date": persona.created_date.isoformat()
                }
                for persona in personas
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get personas: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mira AI Tutor API is running with database persistence"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)