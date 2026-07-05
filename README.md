# 📄 Document Question Answering System (RAG)

A Retrieval-Augmented Generation (RAG) based Document Question Answering System that enables users to ask natural language questions about PDF documents. The application retrieves the most relevant document chunks using semantic search and generates context-aware answers using a Large Language Model hosted on Hugging Face.

---

## 🚀 Features

- 📂 Load PDF documents automatically from the `knowledge_base` folder
- ✂️ Intelligent text chunking using LangChain
- 🧠 Semantic embeddings using Hugging Face Embedding Models
- 📦 FAISS Vector Database for fast similarity search
- 🤖 Hugging Face Inference API for answer generation
- ⚡ FastAPI backend
- 💬 Streamlit chat interface
- 🔍 Context-aware document retrieval
- 🔒 Environment variable support using `.env`

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Frontend UI |
| FastAPI | Backend API |
| LangChain | RAG Pipeline |
| FAISS | Vector Database |
| Hugging Face Embeddings | Semantic Embeddings |
| Hugging Face Inference API | Large Language Model |
| PyPDF | PDF Processing |
| dotenv | Environment Variables |

---

## 📁 Project Structure

```text
Assignment_7/
│
├── backend/
│   ├── index.py          # RAG Pipeline
│   └── main.py           # FastAPI Server
│
├── frontend/
│   └── app.py            # Streamlit Application
│
├── knowledge_base/
│   └── *.pdf             # Documents
│
├── .env
├── requirements.txt
└── README.md
```

---

# ⚙️ RAG Pipeline

The application follows the Retrieval-Augmented Generation (RAG) workflow:

```
PDF Documents
      │
      ▼
Document Loading
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
(Hugging Face)
      │
      ▼
FAISS Vector Store
      │
      ▼
User Question
      │
      ▼
Similarity Search
      │
      ▼
Relevant Context
      │
      ▼
Hugging Face LLM
      │
      ▼
Generated Answer
```

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/Dinesh-jangir/Assignment_7.git
```

Move into the project directory

```bash
cd Assignment_7
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the root directory.

```env
HF_TOKEN=your_huggingface_api_key

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

---

# ▶️ Run the Backend

```bash
cd backend

uvicorn main:app --reload
```

Backend will start at

```
http://127.0.0.1:8000
```

---

# ▶️ Run the Frontend

Open another terminal

```bash
cd frontend

streamlit run app.py
```

---

# 💬 Example Questions

- What is Object-Oriented Programming?
- Summarize the uploaded document.
- Explain the main concept discussed in Chapter 2.
- What are the advantages of web scraping?
- What is Retrieval-Augmented Generation (RAG)?

---

# 📚 How It Works

1. Load PDF documents from the `knowledge_base` directory.
2. Extract text from each document.
3. Split the text into smaller chunks.
4. Generate embeddings using a Hugging Face embedding model.
5. Store embeddings in a FAISS vector database.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant document chunks.
8. Send the retrieved context and question to the Hugging Face LLM.
9. Display the generated answer in the Streamlit interface.

---

# 📷 Application Workflow

```
User
 │
 ▼
Streamlit UI
 │
 ▼
FastAPI Backend
 │
 ▼
FAISS Retriever
 │
 ▼
Relevant Chunks
 │
 ▼
Hugging Face LLM
 │
 ▼
Answer
```

---

# 📌 Future Improvements

- Multiple PDF Upload
- Source Citation with Page Numbers
- Chat History Persistence
- Docker Support
- Authentication
- ChromaDB Support
- Conversation Memory
- Hybrid Search (BM25 + Vector Search)

---

# 👨‍💻 Author

**Dinesh Jangir**

B.Tech (Computer Science Engineering)

Poornima University, Jaipur

GitHub:
https://github.com/Dinesh-jangir

---

## ⭐ If you found this project useful, consider giving it a star!
