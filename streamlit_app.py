import streamlit as st
import requests
import os

def main():
    st.title("Question Answering System")
    question = st.text_input("Enter your question")
    if st.button("Ask"):
        fastapi_url = os.environ.get("FASTAPI_URL", "http://localhost:8000")
        response = requests.post(f"{fastapi_url}/ask", json={"question": question})
        if response.status_code == 200:
            data = response.json()
            st.write("Answer:", data["answer"])
            st.write("Context:", data["context"])
        else:
            st.write("Error:", response.text)

if __name__ == "__main__":
    main()