import streamlit as st
import requests

# ==============================
# Configuration
# ==============================

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="📄",
    layout="wide"
)

# ==============================
# Sidebar
# ==============================

with st.sidebar:
    st.title("📚 RAG Document QA")

    st.markdown(
        """
        ### About

        Ask questions about the documents stored in the **knowledge_base** folder.

        **Pipeline**
        - 📄 PDF Loading
        - ✂️ Text Chunking
        - 🧠 HuggingFace Embeddings
        - 📦 FAISS Vector Database
        - 🤖 HuggingFace LLM
        """
    )

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ==============================
# Main Page
# ==============================

st.title("📄 Document Question Answering System")

st.caption(
    "Ask questions based on the documents in your knowledge base."
)

# ==============================
# Chat History
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your Document QA Assistant.\n\nAsk me anything about your uploaded PDFs."
        }
    ]

for message in st.session_state.messages:

    role = "assistant" if message["role"] == "assistant" else "user"

    with st.chat_message(role):
        st.markdown(message["content"])

# ==============================
# User Input
# ==============================

prompt = st.chat_input("Ask a question...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):

            try:

                response = requests.post(
                    API_URL,
                    json={
                        "text": prompt
                    },
                    timeout=120
                )

                response.raise_for_status()

                answer = response.json()["answer"]

            except requests.exceptions.ConnectionError:

                answer = "❌ Unable to connect to the FastAPI server."

            except requests.exceptions.Timeout:

                answer = "⏳ Request timed out."

            except Exception as e:

                answer = f"❌ Error: {e}"

            st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )