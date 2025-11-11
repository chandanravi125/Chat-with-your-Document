Project Overview

- Build a "Chat with Document" application using RAG, FastAPI, and Streamlit.
- Load a PDF document, split it into chunks, and store embeddings in FAISS.
- Create a FastAPI endpoint to answer questions about the document.
- Use a free LLM (e.g., Google Gemini) for generation.
- Containerize the application using Docker.

Folder Structure

- data/
    - Transformer.pdf
- data_ingestion.py
- rag_chain.py
- main.py
- streamlit_app.py
- requirements.txt
- Dockerfile.fastapi
- Dockerfile.streamlit
- docker-compose.yml

Core Components

1. Data Ingestion: Load PDF, split into chunks, generate embeddings.
2. RAG Chain: Use LangChain for retrieval and generation.
3. FastAPI Endpoint: POST /ask to answer questions.
4. Streamlit App: Send questions to FastAPI endpoint and display answers.

Setup and Run

1. Clone the repository and install dependencies: pip install -r requirements.txt
2. Run FastAPI locally: python main.py
3. Build and run with Docker: docker-compose up --build
4. Run Streamlit app: streamlit run streamlit_app.py

Tech Stack

- Backend: FastAPI, LangChain, FAISS
- Embeddings: Hugging Face (all-MiniLM-L6-v2)
- LLM: Free / Public API (e.g., Gemini)
- Frontend: Streamlit
- Containerization: Docker & Docker Compose