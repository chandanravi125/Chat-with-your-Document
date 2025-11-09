from fastapi import FastAPI
from pydantic import BaseModel
from data_ingestion import vector_store
from rag_chain import RAGChain

app = FastAPI()

class Question(BaseModel):
    """
    A Pydantic model representing a question.

    Attributes:
    ----------
    question : str
        The question to answer.
    """
    question: str

rag_chain = RAGChain(vector_store)

@app.post("/ask")
async def ask_question(question: Question):
    """
    Answers a question using the RAG chain.

    Parameters:
    ----------
    question : Question
        The question to answer.

    Returns:
    -------
    dict
        A dictionary containing the question, answer, and context.
    """
    try:
        answer, context = rag_chain.answer(question.question)
        return {"question": question.question, "answer": answer, "context": context}
    except Exception as e:
        print(f"Error answering question: {e}")
        return {"error": "Failed to answer question"}
