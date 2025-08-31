import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma

load_dotenv()
os.environ["USER_AGENT"] = "MyChatbot/1.0"

persist_directory = "./chroma_db"

# ---------------- Loaders ---------------- #
def WebsiteLoader(urls):
    loader = WebBaseLoader(urls)
    return loader.load()

def CSVFileLoader(file_paths):
    docs = []
    for file_path in file_paths:
        loader = CSVLoader(file_path=file_path)
        docs.extend(loader.load())
    return docs

def PDFLoader(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(file_path=pdf_file)
        docs.extend(loader.load())
    return docs


# ---------------- Build & Save Vectorstore ---------------- #
def build_vectorstore(docs):
    if not docs:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        length_function=len,
    )

    split_docs = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    vectorstore.persist()
    print("✅ Data successfully stored in ChromaDB!")
    return vectorstore
