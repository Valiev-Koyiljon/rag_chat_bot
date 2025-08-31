
````markdown
# 📚 Local RAG Chatbot

A local chatbot that answers questions using your uploaded PDFs, CSVs, and website content.  
Built with **LangChain**, **Ollama LLM**, and **ChromaDB** for embeddings and retrieval.

---

## 🚀 Features

- Upload **CSV** and **PDF** files for document retrieval.  
- Add **website URLs** to fetch content dynamically.  
- Build embeddings locally using **Ollama embeddings**.  
- Use **Ollama LLM** (`exaone3.5:2.4b`) to answer questions based only on the provided documents.  
- Interactive **Streamlit** interface with chat history.

---

## 🛠 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Valiev-Koyiljon/rag_chat_bot/edit/main/README.MDd
cd rag_chat_bot
````

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate local-rag-bot
```

**`environment.yml` includes:**

* Python 3.10
* Streamlit
* LangChain & Ollama integrations
* ChromaDB
* BeautifulSoup4, Requests, LXML for web scraping
* PyPDF for PDF loading

### 3. Install Ollama Runtime

* Download for your platform from [Ollama Downloads](https://ollama.com/download)
* Install it (Windows `.exe` or macOS `.dmg`)

### 4. Pull the required Ollama models

```bash
# For embeddings
ollama pull embedding:mxbai-embed-large

# For chat / LLM
ollama pull exaone3.5:2.4b
```

---

## 🖥 Running the Chatbot

```bash
streamlit run app.py
```

* Upload **PDF/CSV** files from the sidebar.
* Add a **website URL** to include its content.
* Click **“Process Data”** to build embeddings and store them in ChromaDB.
* Ask questions in the chat input — the bot will only answer based on the loaded documents.

---

## ⚙️ Project Structure

```
mychatbot/
├─ app.py                 # Main Streamlit app
├─ rag_backend.py         # Loaders & vectorstore builder
├─ environment.yml        # Conda environment
├─ uploads/               # Uploaded documents
└─ chroma_db/             # Persisted embeddings
```

### Backend Highlights

* `WebsiteLoader(urls)`: Loads web page content using `WebBaseLoader`.
* `CSVFileLoader(file_paths)`: Loads CSV documents.
* `PDFLoader(pdf_files)`: Loads PDFs.
* `build_vectorstore(docs)`: Splits documents and stores embeddings in **ChromaDB**.

---

## 📌 Notes

* Ollama runtime must be installed and running for the chatbot to work.
* The bot will **only answer based on the uploaded documents or website**; it will not use general knowledge.
* Default URL included: [Yeongnam University](https://www.yu.ac.kr/english/index.do)

---

## 🔧 Dependencies

* Python 3.10
* Streamlit
* LangChain (`langchain`, `langchain-ollama`, `langchain-community`, `langchain-chroma`, `langchain-text-splitters`)
* ChromaDB
* Ollama (`ollama`)
* BeautifulSoup4, Requests, LXML
* PyPDF

---

## 📷 Screenshot (Optional)

![LLM RAG Screenshot](img/llm_rag.jpg)
![LLM RAG Screenshot 2](img/llm_rag1.jpg)


---

## 💡 Tips

* To add more websites, just enter the URL in the sidebar.
* Keep your PDFs/CSVs in **uploads/** for easier management.
* ChromaDB persists your embeddings, so you don’t need to rebuild every time.

---


