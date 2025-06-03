from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
import shutil
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
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

# Create directories for storing uploads and vector DB
os.makedirs("uploads", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Initialize Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store document data and vector stores
documents = {}
vector_stores = {}

def process_document(file_path: str):
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
    return vector_store

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
        vector_store = process_document(file_path)
        
        # Store document info and vector store
        documents[file.filename] = {
            "id": document_id,
            "path": file_path,
            "type": file.content_type,
        }
        vector_stores[file.filename] = vector_store
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "message": "Document processed and vectorized successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)