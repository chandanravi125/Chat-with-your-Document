"""
This script loads documents, preprocesses the text, 
and creates vector stores using the LangChain library.

The vector stores can be used to answer questions based on documents.

"""
import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Load env variables from .env file
load_dotenv()

class DataIngestion:
    """
    A class used to ingest and preprocess PDF data.

    Attributes:
    ----------
    pdf_path : str
        The path to the PDF file.
    google_API_key : str
        The Google API key.
    OUT_DIR : str
        The directory to save the FAISS index.
    """

    def __init__(self, pdf_path):
        """
        Initializes the DataIngestion class.

        Parameters:
        ----------
        pdf_path : str
            The path to the PDF file.
        """
        self.pdf_path = pdf_path
        self.google_API_key = os.getenv("GOOGLE_API_KEY")
        self.OUT_DIR = os.path.join(os.getcwd(), "Faiss_indexes")
        os.makedirs(self.OUT_DIR, exist_ok=True)

    def preprocess_text(self, text):
        """
        Preprocesses the text by converting to lowercase, removing special characters and digits, 
        tokenizing, removing stopwords, and lemmatizing.

        Parameters:
        ----------
        text : str
            The text to preprocess.

        Returns:
        -------
        str
            The preprocessed text.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Join tokens back into a string
        text = ' '.join(tokens)
        return text

    def load_pdf(self):
        """
        Loads the PDF file, preprocesses the text, splits it into chunks, 
        and creates a FAISS vector store.

        Returns:
        -------
        FAISS
            The FAISS vector store.
        """
        try:
            # Load the Pdf
            loader = UnstructuredLoader(self.pdf_path)
            docs = loader.load()
            # Preprocess text
            preprocessed_docs = [self.preprocess_text(doc.page_content) for doc in docs]
            # Update doc content
            for i, doc in enumerate(docs):
                doc.page_content = preprocessed_docs[i]
            # Split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            # Filter complex metadata
            filter_chunks = filter_complex_metadata(chunks)
            # Embedding:
            emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vectorstore_pdf = FAISS.from_documents(filter_chunks, embedding=emb)
            vectorstore_pdf.save_local(os.path.join(self.OUT_DIR, "faiss_Pdf_Index"))
            return vectorstore_pdf
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return None

pdf_path = os.path.join(os.getcwd(), "data", "Transformer.pdf")
data_ingestion = DataIngestion(pdf_path)
vector_store = data_ingestion.load_pdf()
