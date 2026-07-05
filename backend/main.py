import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

FAISS_PATH = BASE_DIR / "faiss_index"
DATA_PATH = BASE_DIR / "knowledge_base"

os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", os.getenv("HUGGINGFACEHUB_API_TOKEN", ""))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_or_load_vector_store():
    if (FAISS_PATH / "index.faiss").exists() and (FAISS_PATH / "index.pkl").exists():
        return FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {DATA_PATH}")

    txt_loader = DirectoryLoader(str(DATA_PATH), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    pdf_loader = DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader)

    txt_docs = txt_loader.load()
    pdf_docs = pdf_loader.load()
    docs = txt_docs + pdf_docs

    if not docs:
        raise FileNotFoundError(f"No documents found in {DATA_PATH}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)

    FAISS_PATH.mkdir(parents=True, exist_ok=True)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(str(FAISS_PATH))
    return db


try:
    db = build_or_load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    rag_ready = True
except Exception as e:
    db = None
    retriever = None
    rag_ready = False
    startup_error = str(e)

model_error = None
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    model_error = "HUGGINGFACEHUB_API_TOKEN is not set"

SYSTEM_PROMPT = '''
You are a helpful assistant.
Use the context to answer the question in max three sentences.
If you don't know, say you don't know.
Context: {context}
'''

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}")
])

rag_chain = None
if rag_ready and retriever is not None:
    rag_chain = retriever

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/query")
def query_rag(query: Query):
    if not rag_ready or rag_chain is None:
        return {"answer": f"RAG is not ready: {startup_error or model_error}"}

    try:
        if rag_chain is None:
            return {"answer": f"RAG is not ready: {startup_error or model_error}"}

        docs = rag_chain.invoke(query.text)
        if docs:
            context = "\n\n".join(doc.page_content for doc in docs[:3])
            return {"answer": f"\n\nRelevant context:\n{context}"}
        return {"answer": "No relevant documents found."}
    except Exception as exc:
        return {"answer": f"Generation failed: {exc}"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)