# RAG-Based PDF QA System

A system that uses Retrieval-Augmented Generation (RAG) to answer questions based on a provided PDF document.

Navigate to the project directory: `cd RAG_PDF`
## Setup Environment: Create a virtualenv and install requirements
python -m venv venv 
source venv/bin/activate 
`pip install -r requirements.txt`

## Run FastAPI Server Locally
`uvicorn main:app --host 0.0.0.0 --port 8000`

## Build and Run Docker Container
1. Build the Docker image: `docker build -t rag-pdf-qa-system .`
2. Run the Docker container: `docker run -p 8000:8000 rag-pdf-qa-system`

## Run Streamlit App
`streamlit run frontend/app.py`


API Endpoints:

- GET /: Welcome to "Chat with your Document"
- POST /upload: Upload a PDF document
- POST /ask: Ask a question about the uploaded document
    - Request Body (JSON): { "question": "Your question here" }
    - Response Body (JSON): { 
        "question": "Your question here", 
        "answer": "The generated answer from the LLM.", 
        "context": "The specific chunk(s) of text retrieved from the vector store." }


 # a docker-compose.yml file to define the services and how they communicate with each other
