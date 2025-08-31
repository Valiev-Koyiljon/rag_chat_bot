import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate  
from rag_backend import CSVFileLoader, PDFLoader, WebsiteLoader, build_vectorstore, persist_directory

from dotenv import load_dotenv
load_dotenv()

# ---- Streamlit UI ---- #
st.set_page_config(layout="wide")
st.title("📚 My Local Chatbot")

st.sidebar.header("Settings")
MODEL = st.sidebar.selectbox("Choose a Model", ["exaone3.5:2.4b"], index=0)
MAX_HISTORY = st.sidebar.number_input("Max History", 1, 10, 2)

# ---- Session State ---- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# ---- Upload Section ---- #
st.sidebar.subheader("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True
)

# ---- Web URL Section ---- #
st.sidebar.subheader("Add Website")
default_url = "https://www.yu.ac.kr/english/index.do"
web_url = st.sidebar.text_input("Enter Website URL", value=default_url)

# ---- Process Button ---- #
if st.sidebar.button("📥 Process Data (Files + URL)"):
    docs = []
    csv_files, pdf_files = [], []

    # Save uploaded files
    for file in uploaded_files:
        file_path = os.path.join("uploads", file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.type == "text/csv":
            csv_files.append(file_path)
        elif file.type == "application/pdf":
            pdf_files.append(file_path)

    # Load CSV/PDF docs
    if csv_files:
        docs.extend(CSVFileLoader(csv_files))
    if pdf_files:
        docs.extend(PDFLoader(pdf_files))

    # Load website docs
    if web_url:
        docs.extend(WebsiteLoader([web_url]))

    # Build embeddings + store
    build_vectorstore(docs)
    st.sidebar.success("✅ Data processed & added to ChromaDB!")

# ---- LangChain Components ---- #
llm = ChatOllama(model=MODEL, streaming=True)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity")


instruction_template = """
You are an assistant that ONLY answers questions based on the provided documents.
Do not use your general knowledge. If the answer is not in the documents, say 'I do not know.
If user says greetings like "hello", "hi", say that you are AI bot to answer about Yeongnam university'

Documents:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(template=instruction_template, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)


# ---- Display Chat History ---- #
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Chat Input ---- #
if prompt := st.chat_input("💬 Say something..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_container = st.empty()
        result = qa.invoke({"query": prompt})
        response = result.get("result", "⚠️ No response generated.")
        response_container.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
