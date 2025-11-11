# import os
# from langchain_google_genai import ChatGoogleGenerativeAI

# class RAGChain:
#     """
#     A class used to create a RAG chain for question answering.

#     Attributes:
#     ----------
#     vector_store : FAISS
#         The FAISS vector store.
#     google_key : str
#         The Google API key.
#     llm : ChatGoogleGenerativeAI
#         The language model.
#     """

#     def __init__(self, vector_store):
#         """
#         Initializes the RAGChain class.

#         Parameters:
#         ----------
#         vector_store : FAISS
#             The FAISS vector store.
#         """
#         self.vector_store = vector_store
#         # self.google_key = os.getenv("GOOGLE_API_KEY")   MISTRAL_API_KEY
#         self.mistral_key = os.getenv("MISTRAL_API_KEY")   # mistral_key
#         if not self.mistral_key:
#             raise ValueError("GOOGLE_API_KEY not found in environment variables") 
#         self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", mistral_api_key=self.mistral_key, temperature=0.2)


#     def answer(self, question):
#         """
#         Answers a question using the RAG chain.

#         Parameters:
#         ----------
#         question : str
#             The question to answer.

#         Returns:
#         -------
#         tuple
#             A tuple containing the answer and the context.
#         """
#         # Retrieve relevant documents from the vector store
#         docs = self.vector_store.similarity_search(question, k=3)
#         # Use the LLM to generate an answer based on the retrieved documents
#         answer = self.llm.generate_answer(question, docs)
#         # Get the context from the retrieved documents
#         context = [doc.page_content for doc in docs]
#         return answer, context
    

import os
from mistralai import Mistral

class RAGChain:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.mistral_key = os.getenv("MISTRAL_API_KEY")
        if not self.mistral_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        self.client = Mistral(api_key=self.mistral_key)

    def answer(self, question):
        docs = self.vector_store.similarity_search(question, k=3)
        # You need to implement the logic to generate answer using Mistral API
        # Here's a simple example:
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": question}],
            context=[doc.page_content for doc in docs]
        )
        answer = response.choices[0].message.content
        context = [doc.page_content for doc in docs]
        return answer, context
