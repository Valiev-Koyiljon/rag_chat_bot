
# 📚 My Local RAG Chatbot 

A **Streamlit-based local chatbot** powered by **LangChain**, **Ollama**, and **ChromaDB**.
This chatbot answers questions **strictly based on uploaded documents or website content**, without using any external knowledge or API keys.

---

## **Features**

* Upload CSV or PDF files for contextual Q\&A
* Add website URLs for scraping and embedding
* Use local Ollama LLM models (`exaone3.5:2.4b`)
* Chat history with session memory
* Answers are restricted to the provided documents
* Stores embeddings locally in ChromaDB

---

## **Setup**

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo>
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file (no API key required):

```env
USER_AGENT=MyChatbot/1.0
```

---

### 5. Pull Ollama Models Locally

```bash
ollama pull exaone3.5:2.4b
ollama pull mxbai-embed-large
```

> These models are required for the LLM and embeddings.

---

### 6. Run the App

```bash
streamlit run app.py
```

---

## **Usage**

1. Upload CSV or PDF files via the sidebar.
2. Add a website URL (optional) to scrape content.
3. Click **Process Data** to generate embeddings and store them in ChromaDB.
4. Type your question in the chat box.
5. The bot will answer **strictly based on the uploaded documents**.

---

## **Project Structure**

```
.
├─ app.py                # Main Streamlit app
├─ rag_backend.py        # Data loaders & vectorstore builder
├─ requirements.txt      # Python dependencies
├─ .env                  # Environment variables (local)
├─ uploads/              # Uploaded files
└─ chroma_db/            # Persisted embeddings
```

---

## **Requirements**

* Python 3.10+
* Streamlit
* LangChain
* LangChain Ollama
* LangChain Chroma
* ChromaDB
* python-dotenv
* Additional dependencies listed in `requirements.txt`

---

