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
        context = [doc.page_content for doc in docs]
        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": question + "\n\nContext: " + "\n".join(context)}],
            max_tokens=70,
            temperature=0.7,
            top_p=0.9,
            stop=["\n\n", "###"]
)
        answer = response.choices[0].message.content
        return answer