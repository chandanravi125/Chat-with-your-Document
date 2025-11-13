
import os
import re
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging

logging.basicConfig(level=logging.ERROR)

class VectorDatabase:
    def __init__(self, pdf_path="data/Transformer.pdf", index_path="Faiss_indexes"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def preprocess_text(self, text):
        text = re.sub(r'\[[0-9]+\]', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?;:\(\)\[\]\{\}\'\"]', '', text)
        return text

    def create_vector_db(self):
        """
        Creates a vector database from the documents in the pdf file.
        
        Returns:
            vector_store: The created vector store.
        """
        try:
            loader = UnstructuredLoader(self.pdf_path, autodetect_encoding=True)
            docs = loader.load()
            if not docs:
                raise ValueError("No documents loaded")
            for doc in docs:
                doc.page_content = self.preprocess_text(doc.page_content)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            filtered_chunks = filter_complex_metadata(chunks)
            self.vector_store = FAISS.from_documents(filtered_chunks, embedding=self.embeddings)
            self.vector_store.save_local(self.index_path)
            return self.vector_store
        except Exception as e:
            logging.error(f"Error: {e}")

    def load_vector_db(self):
        try:
            self.vector_store = FAISS.load_local(self.index_path, self.embeddings)
        except Exception as e:
            logging.error(f"Error: {e}")

    def get_relevant_docs(self, query):
        """
        Retrieves relevant documents from the vector store based on the query.
        
        Args:
            query (str): The query to search for.
        
        Returns:
            list: A list of relevant documents.
        """
        try:
            if self.vector_store is None:
                if not os.path.exists(self.index_path):
                    self.create_vector_db()
                else:
                    self.load_vector_db()
            results = self.vector_store.similarity_search(query, k=2)
            return results
        except Exception as e:
            logging.error(f"Error: {e}")

# Usage vector_db = VectorDatabase()
# vector_db.create_vector_db()
# results = vector_db.get_relevant_docs("query")
