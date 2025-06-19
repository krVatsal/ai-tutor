from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import shutil
import uuid
import httpx
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

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

# Create directories for storing uploads and vector DB
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Initialize Gemini embeddings and LLM
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Store document data and vector stores
documents = {}
vector_stores = {}
personas = {}

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

def process_document(file_path: str):
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
      # Create vector store
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store, documents

def extract_text_from_documents(documents_list):
    """Extract text content from document list"""
    text_content = ""
    for doc in documents_list:
        text_content += doc.page_content + "\n\n"
    return text_content.strip()

async def create_tavus_persona(document_text: str, document_name: str) -> Dict[str, Any]:
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
        
        return response.json()

async def create_tavus_conversation(persona_id: str, document_name: str) -> Dict[str, Any]:
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
        
        return response.json()

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF document"""
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate unique ID and save file
        document_id = str(uuid.uuid4())
        file_path = f"uploads/{document_id}.pdf"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document and create vector store
        vector_store, doc_list = process_document(file_path)
        
        # Extract text content for persona creation
        document_text = extract_text_from_documents(doc_list)
        
        # Store document info and vector store
        documents[file.filename] = {
            "id": document_id,
            "path": file_path,
            "vector_db_path": vector_db_path,
            "type": file.content_type,
            "text": document_text
        }
        vector_stores[file.filename] = vector_store
        
        # Create Tavus persona with document context
        try:
            persona_response = await create_tavus_persona(document_text, file.filename)
            personas[file.filename] = persona_response
            
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

@app.post("/create-persona")
async def create_persona_endpoint(request: PersonaRequest):
    """Create a Tavus persona with document context"""
    try:
        persona_response = await create_tavus_persona(request.document_text, request.document_name)
        return persona_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create persona: {str(e)}")

@app.post("/create-conversation")
async def create_conversation_endpoint(request: ConversationRequest):
    """Create a Tavus conversation with persona"""
    try:
        conversation_response = await create_tavus_conversation(request.persona_id, request.document_name)
        return conversation_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")

@app.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
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
async def tavus_webhook(request: Request):
    """Handle Tavus webhook events"""
    try:
        event = await request.json()
        
        print(f"Tavus webhook received: {event.get('type')} - {event.get('conversation_id')}")
        
        event_type = event.get("type")
        
        if event_type == "conversation.started":
            print(f"Conversation started: {event.get('conversation_id')}")
            
        elif event_type == "conversation.ended":
            print(f"Conversation ended: {event.get('conversation_id')}")
            
        elif event_type == "utterance":
            print(f"User utterance: {event.get('utterance')}")
            # Handle user speech input here
            
        elif event_type == "tool_call":
            tool_name = event.get("tool_name")
            print(f"Tool call requested: {tool_name}")
            
            if tool_name == "lookup_doc":
                # Handle document lookup
                query = event.get("parameters", {}).get("query", "")
                conversation_id = event.get("conversation_id")
                
                # Implement document search logic here
                # You could use the vector store to find relevant content
                result = f"Found information related to: {query}"
                
                # Return the result (this would typically be sent back to Tavus)
                return {"result": result}
        
        return {"success": True}
    
    except Exception as e:
        print(f"Webhook processing error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """Chat with AI about the document using vector search"""
    try:
        if not request.document_name or request.document_name not in vector_stores:
            raise HTTPException(status_code=400, detail="Document not found or not processed")
        
        # Get vector store for the document
        vector_store = vector_stores[request.document_name]
        
        # Search for relevant content
        relevant_docs = vector_store.similarity_search(request.query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Simple response generation (you could integrate with your LLM here)
        response = f"Based on the document content, here's what I found relevant to your question '{request.query}':\n\n{context[:500]}..."
        
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Mira AI Tutor API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)