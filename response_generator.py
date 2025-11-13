from rag_retriever import VectorDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
import os

PROMPT_TEMPLATE = PromptTemplate.from_template("""
You are a helpful AI assistant that answers questions *only* based on the provided PDF content. If the question cannot be answered using the context below, reply exactly with: "This query is not related to the provided PDF content."
Context:{context}
Question: {question}
Answer:
""")

class ResponseGenerator:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_response(self, query):
        try:
            relevant_docs = self.vector_db.get_relevant_docs(query)
            context = "\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant KB found."
            filled_prompt = PROMPT_TEMPLATE.format(question=query, context=context)
            response = self.model.predict(filled_prompt)
            return {
                "question": query,
                "answer": response,
                "context": context
            }
        except Exception as e:
            print(f"Error generating response: {e}")
            return None