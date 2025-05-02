from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from your_retriever_module import answer_question_with_sources  # Import your existing function

# Load environment variables
load_dotenv()

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    answer = answer_question_with_sources(query.question)
    return {"answer": answer, "sources": ["Case 1", "Case 2"]}  

