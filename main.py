import os
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_engine import process_pdf_into_memory, ask_socratic_ai

load_dotenv()

app = FastAPI()

# Allow requests from ANY website
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    class_id: str

class IngestRequest(BaseModel):
    file_url: str 
    class_id: str

@app.get("/")
def read_root():
    return {"status": "AI Brain v3 (Refactored) is Online"}

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    try:
        # Download file
        print(f"Downloading from: {request.file_url}") 
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(request.file_url, headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Blocked download (Status: {response.status_code}). URL: {request.file_url}")
            
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        # We pass the file path and class_id
        result_message = process_pdf_into_memory(tmp_path, request.class_id)

        os.unlink(tmp_path)
        
        return {"status": "success", "message": result_message}

    except Exception as e:
        print(f"Error: {e}") 
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        ai_response = ask_socratic_ai(request.question, request.class_id)
        
        return {"answer": ai_response}
    except Exception as e:
        return {"answer": "Error: " + str(e)}