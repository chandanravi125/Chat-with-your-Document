from fastapi import FastAPI, status
from pydantic import BaseModel
from response_generator import ResponseGenerator
from rag_retriever import VectorDatabase

app = FastAPI()
vector_db = VectorDatabase()
response_generator = ResponseGenerator(vector_db)

class Question(BaseModel):
    question: str

@app.get("/")
def index():
    return "Welcome to the Chat with me! Hey I am AI Transformer Pdf"

@app.post("/ask", status_code=status.HTTP_200_OK)
async def ask_question(question: Question):
    try:
        user_query = question.question.strip()
        if not user_query:
            return {"error": "Query cannot be empty"}, status.HTTP_400_BAD_REQUEST
        response = response_generator.generate_response(user_query)
        if response is None:
            return {"error": "Failed to generate response"}, status.HTTP_500_INTERNAL_SERVER_ERROR
        return response
    except Exception as e:
        return {"error": str(e)}, status.HTTP_500_INTERNAL_SERVER_ERROR