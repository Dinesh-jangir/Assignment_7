#load split embeddings store

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_PATH = BASE_DIR / "knowledge_base"
FAISS_PATH = BASE_DIR / "faiss_index"

print("Loading documents...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Knowledge base folder not found: {DATA_PATH}")

files = [str(p.name) for p in DATA_PATH.iterdir()]
print(f"Files found in knowledge base: {files}")

txt_loader = DirectoryLoader(str(DATA_PATH), glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
txt_docs = txt_loader.load()

pdf_loader = DirectoryLoader(str(DATA_PATH), glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

docs = txt_docs + pdf_docs

if not docs:
    raise FileNotFoundError(
        f"No readable documents found in {DATA_PATH}. Add .txt or .pdf files."
    )

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(docs)

print("Creating embeddings...")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

FAISS_PATH.mkdir(parents=True, exist_ok=True)
db = FAISS.from_documents(docs, embeddings)
db.save_local(str(FAISS_PATH))

print("FAISS index created successfully and saved locally!")


