# Chat with  Document — RAG + FAISS + FastAPI + Streamlit
## Objective

Your goal is to build, containerize, and test a simple “Chat with Document” application.
This project demonstrates end-to-end skills in Retrieval-Augmented Generation (RAG), API Development (FastAPI), and MLOps (Dockerization).

🗂 Folder Structure
|---- data/
|     |---- Transformer.pdf
|---- data_ingestion.py
|---- rag_chain.py
|---- main.py
|---- streamlit_app.py
|---- requirements.txt
|---- Dockerfile.fastapi
|---- Dockerfile.streamlit
|---- docker-compose.yml
|---- .env
|---- .gitignore

## Core Components
Part 1: The RAG API 
A FastAPI service that allows users to ask questions about the document.
## Requirements
Data Ingestion:
On startup, load the PDF.
Split it into text chunks using RecursiveCharacterTextSplitter.
Generate embeddings.
Vector Store:
Use FAISS to store and retrieve embeddings locally.
Embeddings:
Use any open-source model (e.g., all-MiniLM-L6-v2 from Hugging Face).
RAG Chain:
Use LangChain for retrieval and generation.
Use a free or open LLM (e.g., Google Gemini).

API Endpoint:
POST /ask
{
  "question": "What is Attention in Transformer?"
}


Response:

{
  "question": "Your question here",
  "answer": "The generated answer from the LLM.",
  "context": "The specific chunk(s) of text retrieved from the vector store."
}

Part 2: MLOps — Dockerization (Mandatory)
Requirements

Create a Dockerfile (Dockerfile.fastapi) for your FastAPI service.

It should:

Install all dependencies from requirements.txt.

Optionally use a multi-stage build for optimization.

Run the FastAPI app when the container starts.

A working docker-compose.yml should orchestrate both FastAPI and Streamlit services.

Part 3: UI — Streamlit (Bonus)

Create a simple Streamlit app (streamlit_app.py) that:

Takes a question as input.

Sends it to the FastAPI endpoint.

Displays the question, answer, and context neatly.

Does not contain the RAG logic itself (only calls the API).

🧾 Data Source

Use the original Transformer paper:
📄 Attention Is All You Need (arXiv)

Save it as:

data/Transformer.pdf

⚙️ Setup & Run Instructions
1️⃣ Environment Setup
git clone <your-repo-link>
cd <your-repo-name>
pip install -r requirements.txt

2️⃣ Run FastAPI Locally
python main.py


Then open the API docs in your browser:
👉 http://127.0.0.1:8000/docs

3️⃣ Build & Run with Docker
Build FastAPI Image
docker build -f Dockerfile.fastapi -t rag-fastapi .

Run Container
docker run -p 8000:8000 rag-fastapi


Or use Docker Compose:

docker-compose up --build

4️⃣ Run Streamlit App (Bonus)
streamlit run streamlit_app.py


Then visit 👉 http://localhost:8501

🧰 Tech Stack
Component	Technology Used
Backend	FastAPI
Retrieval	LangChain + FAISS
Embeddings	Hugging Face (all-MiniLM-L6-v2)
LLM	Free / Public API (e.g. Gemini)
Frontend (Bonus)	Streamlit
Containerization	Docker & Docker Compose
✅ Deliverables

Complete GitHub repository containing:

All Python code (main.py, data_ingestion.py, rag_chain.py, streamlit_app.py)

Dockerfile.fastapi, Dockerfile.streamlit, and docker-compose.yml

requirements.txt

README.md