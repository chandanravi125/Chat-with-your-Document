import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import RAGAgent

class RAGChain:
    """
    A class used to create a RAG chain for question answering.

    Attributes:
    ----------
    vector_store : FAISS
        The FAISS vector store.
    google_key : str
        The Google API key.
    llm : ChatGoogleGenerativeAI
        The language model.
    """

    def __init__(self, vector_store):
        """
        Initializes the RAGChain class.

        Parameters:
        ----------
        vector_store : FAISS
            The FAISS vector store.
        """
        self.vector_store = vector_store
        self.google_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables") 
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=self.google_key, temperature=0.2)


    def answer(self, question):
        """
        Answers a question using the RAG chain.

        Parameters:
        ----------
        question : str
            The question to answer.

        Returns:
        -------
        tuple
            A tuple containing the answer and the context.
        """
        # Retrieve relevant documents from the vector store
        docs = self.vector_store.similarity_search(question, k=3)
        # Use the LLM to generate an answer based on the retrieved documents
        answer = self.llm.generate_answer(question, docs)
        # Get the context from the retrieved documents
        context = [doc.page_content for doc in docs]
        return answer, context