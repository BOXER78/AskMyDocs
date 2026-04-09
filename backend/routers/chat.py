import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag.vectorstore import process_and_store_document, ask_question, process_and_store_repo
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

class QueryRequest(BaseModel):
    query: str

class RepoRequest(BaseModel):
    path: str

import traceback

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        process_and_store_document(temp_file_path)
        return {"message": "Document processed and stored successfully."}
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/analyze-repo")
async def analyze_repository(request: RepoRequest):
    is_url = request.path.startswith("http")
    if not is_url and not os.path.exists(request.path):
        raise HTTPException(status_code=400, detail="Local repository path does not exist.")
    
    try:
        process_and_store_repo(request.path)
        return {"message": "Repository indexed successfully."}
    except Exception as e:
        print(f"Error indexing repo: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to index repository: {str(e)}")

@router.post("/ask")
async def ask_chatbot(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        answer = ask_question(request.query)
        return {"answer": answer}
    except Exception as e:
        error_str = str(e).lower()
        print(f"Error in chat: {str(e)}")
        traceback.print_exc()
        if "rate limit" in error_str or "429" in error_str:
            raise HTTPException(status_code=429, detail="AI Rate limit reached. Please try again in a few minutes.")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")


