# 📚 Local RAG Chatbot

Welcome to the **Local RAG Chatbot** project! This application enables you to build a conversational AI that can answer questions based on your own documents, such as PDFs, CSVs, and web content. It leverages LangChain, Ollama, ChromaDB, and Streamlit to provide an interactive and intelligent chatbot experience.

![RAG Chatbot Architecture](img/llm_rag.jpg)
*Example architecture of the RAG chatbot*

---

## 🛠️ Project Overview

This chatbot utilizes a Retrieval-Augmented Generation (RAG) approach, combining document retrieval with language model generation to provide accurate and context-aware responses. Key features include:

* **Document Upload**: Supports PDF and CSV uploads for content ingestion.
* **Web Scraping**: Extracts information from specified URLs.
* **ChromaDB Integration**: Stores and retrieves document embeddings for efficient search.
* **Ollama LLM**: Powers the conversational AI with advanced language models.
* **Streamlit Interface**: Provides an intuitive web interface for interaction.

![Chatbot Interface Example](img/llm_rag1.jpg)
*Example screenshot of the chatbot interface*

---

## 📦 Installation Guide

### 1. Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate local-rag-bot
```

### 2. Install Ollama Runtime

Download and install the Ollama runtime for your operating system:

* **Windows**: [Download Ollama for Windows](https://ollama.com/download)
* **macOS**: [Download Ollama for macOS](https://ollama.com/download)

### 3. Pull Required Models

After installing Ollama, pull the necessary models:

```bash
# For embeddings
ollama pull embedding:mxbai-embed-large

# For chat / LLM
ollama pull exaone3.5:2.4b
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Document Processing**: Upload PDFs, CSVs, or provide URLs. The application processes these inputs and stores them in ChromaDB.
2. **Embedding Generation**: Utilizes the `mxbai-embed-large` model to generate embeddings for the documents.
3. **Conversational Interaction**: The `exaone3.5:2.4b` model powers the chatbot, providing responses based on the ingested content.
4. **Chat History**: Maintains a conversation buffer to handle multi-turn interactions.

---

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following content:

```
MODEL=exaone3.5:2.4b
MAX_HISTORY=5
```

---

## 📂 Project Structure

```
mychatbot/
├── app.py                  # Main Streamlit application
├── rag_backend.py          # Backend processing logic
├── environment.yml         # Conda environment configuration
├── .env                    # Environment variables
├── img/
│   ├── llm_rag.jpg         # Architecture diagram
│   └── llm_rag1.jpg        # Chatbot interface screenshot
├── uploads/                # Folder for uploaded files
└── chroma_db/              # Folder for persistent embeddings
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### ✅ Notes for First-Time Users

1. Make sure **Ollama runtime** is installed and running.
2. Pull the required models before running the app.
3. The bot will **only answer questions based on the uploaded documents or web content**.
4. Default URL included: [Yeongnam University](https://www.yu.ac.kr/english/index.do)

---

