import streamlit as st
import requests
import os

def main():
    st.title("Question Answering System")
    question = st.text_input("Enter your question")

    if st.button("Ask"):
        fastapi_url = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")
        try:
            response = requests.post(f"{fastapi_url}/ask", json={"question": question})
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                st.error(data["error"])
            elif "answer" in data and "context" in data:
                st.write("Answer:", data["answer"])
                st.write("Context:", data["context"])
            else:
                st.error("Invalid response from server")
        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()