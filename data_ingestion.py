"""
Simple script to load, preprocess, and create a FAISS vector store
using LangChain and Hugging Face embeddings.
"""

import os
import re
import nltk
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import filter_complex_metadata
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings



# Download required NLTK data silently
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Load environment variables (if needed)
load_dotenv()


class DataIngestion:
    """
    Handles loading, preprocessing, and vectorizing documents.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.output_dir = os.path.join(os.getcwd(), "Faiss_indexes")
        os.makedirs(self.output_dir, exist_ok=True)

        # Use open-source Hugging Face embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize reusable NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def preprocess_text(self, text: str) -> str:
        """
        Cleans and lemmatizes input text.
        """
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in text.split()
            if word not in self.stop_words
        ]
        return ' '.join(tokens)

    def load_and_vectorize(self):
        """
        Loads document, preprocesses it, splits into chunks,
        and creates FAISS vector store.
        """
        try:
            # Load PDF or text file
            loader = UnstructuredLoader(self.pdf_path)
            docs = loader.load()

            # Preprocess text
            for doc in docs:
                doc.page_content = self.preprocess_text(doc.page_content)

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            chunks = filter_complex_metadata(chunks)

            # Build FAISS store
            vector_store = FAISS.from_documents(chunks, embedding=self.embeddings)
            vector_store.save_local(os.path.join(self.output_dir, "faiss_Pdf_Index"))

            print(f"✅ FAISS index created successfully at: {self.output_dir}")
            return vector_store

        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return None


