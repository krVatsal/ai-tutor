from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
import uuid
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
import openai
import ssl

# Load environment variables
load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
openai.api_key = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Configure CORS with HTTPS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploads and vector DB
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store document data and vector stores
documents = {}
vector_stores = {}

class ChatRequest(BaseModel):
    query: str
    document_name: Optional[str] = None

class SummarizeRequest(BaseModel):
    document_name: str

class ChatResponse(BaseModel):
    response: str

class SummaryResponse(BaseModel):
    summary: str

def process_document(file_path: str, file_extension: str):
    # Load document based on file type
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    splits = text_splitter.split_documents(documents)
    
    # Create and save vector store
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        document_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in ['.pdf', '.docx', '.txt']:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        file_path = f"uploads/{document_id}{file_extension}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document and create vector store
        vector_store = process_document(file_path, file_extension)
        
        # Store document info and vector store
        documents[file.filename] = {
            "id": document_id,
            "path": file_path,
            "type": file.content_type,
        }
        vector_stores[file.filename] = vector_store
        
        return {"success": True, "document_id": document_id, "filename": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.document_name or request.document_name not in vector_stores:
            return ChatResponse(response="Please upload a document first.")
        
        vector_store = vector_stores[request.document_name]
        
        # Initialize Gemini chat model
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        
        # Create conversational chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )
        
        # Get response
        result = qa_chain({"question": request.query, "chat_history": []})
        
        return ChatResponse(response=result["answer"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_document(request: SummarizeRequest):
    try:
        if request.document_name not in vector_stores:
            raise HTTPException(status_code=404, detail="Document not found")
        
        vector_store = vector_stores[request.document_name]
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        
        # Create summary chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True,
        )
        
        # Get summary
        result = qa_chain({"question": "Please provide a comprehensive summary of this document.", "chat_history": []})
        
        return SummaryResponse(summary=result["answer"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # SSL context for HTTPS
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile="localhost.crt",
        keyfile="localhost.key"
    )
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_certfile="localhost.crt",
        ssl_keyfile="localhost.key"
    )