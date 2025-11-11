
from fastapi import FastAPI
from pydantic import BaseModel
from data_ingestion import DataIngestion
from rag_chain import RAGChain
import os

app = FastAPI()

class Question(BaseModel):
    question: str

pdf_path = os.path.join(os.getcwd(), "data", "Transformer.pdf")
ingestion = DataIngestion(pdf_path)
vector_store = ingestion.load_and_vectorize()
rag_chain = RAGChain(vector_store)

@app.get("/")
def index():
    return "Welcome to the Persona Agent API!"

@app.post("/ask")
def ask_question(question: Question):
    try:
        answer, context = rag_chain.answer(question.question)
        return {"question": question.question, "answer": answer, "context": context}
    except Exception as e:
        print(f"Error answering question: {e}")
        return {"error": "Failed to answer question"}