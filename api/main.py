from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional
import os
import shutil
import uuid
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

def load_existing_vector_stores():
    """Load existing vector stores from disk on startup"""
    try:
        if os.path.exists("vector_db"):
            for folder_name in os.listdir("vector_db"):
                folder_path = os.path.join("vector_db", folder_name)
                if os.path.isdir(folder_path):
                    try:
                        # Reconstruct document name from folder name
                        document_name = folder_name.replace('_', ' ') + '.pdf'
                        
                        # Load vector store from disk
                        vector_store = FAISS.load_local(folder_path, embeddings, allow_dangerous_deserialization=True)
                        vector_stores[document_name] = vector_store
                        
                        # Also add to documents dict (basic info)
                        documents[document_name] = {
                            "id": folder_name,
                            "vector_db_path": folder_path,
                            "type": "application/pdf"
                        }
                        
                        print(f"Loaded vector store for: {document_name}")
                    except Exception as e:
                        print(f"Failed to load vector store from {folder_path}: {e}")
        
        print(f"Total vector stores loaded: {len(vector_stores)}")
        print(f"Available documents: {list(vector_stores.keys())}")
    except Exception as e:
        print(f"Error loading vector stores: {e}")

def process_document(file_path: str, document_name: str):
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
    
    # Save vector store to disk
    vector_db_path = f"vector_db/{document_name.replace('.pdf', '').replace(' ', '_')}"
    print(f"Saving vector store to: {vector_db_path}")
    
    # Ensure the directory exists
    os.makedirs(vector_db_path, exist_ok=True)
    vector_store.save_local(vector_db_path)
    
    print(f"Vector store saved successfully to: {vector_db_path}")
    return vector_store, vector_db_path

@app.post("/api/upload")
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
        vector_store, vector_db_path = process_document(file_path, file.filename)
        
        # Store document info and vector store
        documents[file.filename] = {
            "id": document_id,
            "path": file_path,
            "vector_db_path": vector_db_path,
            "type": file.content_type,
        }
        vector_stores[file.filename] = vector_store
        
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "vector_db_path": vector_db_path,
            "message": "Document processed and vectorized successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.post("/api/chat")
async def chat_with_document(request: ChatRequest):
    try:
        print(f"Received request - document_name: '{request.document_name}', query: '{request.query}'")
        print(f"Available documents: {list(vector_stores.keys())}")
        
        if not request.document_name or request.document_name not in vector_stores:
            raise HTTPException(status_code=400, detail=f"No valid document found. Available documents: {list(vector_stores.keys())}")
        
        # Get the vector store for the document
        vector_store = vector_stores[request.document_name]
        
        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
          # Get response
        response = qa_chain.invoke(request.query)
        
        # Extract the result text from the response
        result_text = response.get('result', response) if isinstance(response, dict) else str(response)
        
        return {"response": result_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat request: {str(e)}")

@app.post("/api/summarize")
async def summarize_document(request: SummarizeRequest):
    try:
        if request.document_name not in vector_stores:
            raise HTTPException(status_code=400, detail="Document not found")
        
        # Get the vector store for the document
        vector_store = vector_stores[request.document_name]
          # Create retrieval QA chain for summarization
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5})
        )
        
        # Generate summary
        summary_query = "Please provide a comprehensive summary of this document, highlighting the key points, main topics, and important information."
        summary_response = qa_chain.invoke(summary_query)
        
        # Extract the result text from the response
        summary_text = summary_response.get('result', summary_response) if isinstance(summary_response, dict) else str(summary_response)
        
        return {"summary": summary_text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize document: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """List all available documents and their vector stores"""
    return {
        "documents": {name: type(name).__name__ for name in documents.keys()},
        "vector_stores": {name: f"'{name}' (length: {len(name)})" for name in vector_stores.keys()},
        "total_documents": len(documents),
        "total_vector_stores": len(vector_stores)
    }

# Load existing vector stores on startup
load_existing_vector_stores()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)