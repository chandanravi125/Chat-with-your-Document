## Chat with Document — RAG + FAISS + FastAPI + Streamlit
Objective
Build, containerize, and test a simple "Chat with Document" application using Retrieval-Augmented Generation (RAG), API Development (FastAPI), and MLOps (Dockerization).

## Folder Structure
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

## Core Components
1. Data Ingestion: Load PDF, split into chunks, generate embeddings.
2. RAG Chain: Use LangChain for retrieval and generation.
3. FastAPI Endpoint: POST /ask to answer questions.
4. Streamlit App: Send questions to FastAPI endpoint and display answers.

## Setup and Run
1. Clone the repository and install dependencies:

bash
git clone <your-repo-link>
cd <your-repo-name>
pip install -r requirements.txt

2. Run FastAPI locally:

python -m uvicorn main:app --reload

Open the API docs in your browser: http://127.0.0.1:8000/docs
3. Build and run with Docker:

bash
docker-compose up --build

4. Run Streamlit app:

bash
streamlit run streamlit_app.py

Visit http://localhost:8501

Tech Stack
- Backend: FastAPI, LangChain, FAISS
- Embeddings: Hugging Face (all-MiniLM-L6-v2)
- LLM: Free / Public API (e.g., Gemini)
- Frontend: Streamlit
- Containerization: Docker & Docker Compose

Docker Commands
- Build FastAPI image: docker build -f Dockerfile.fastapi -t rag-fastapi .
- Run FastAPI container: docker run -p 8000:8000 rag-fastapi
- Build and run with Docker Compose: docker-compose up --build