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



-------------------------


# 📚 Local RAG Chatbot

Welcome to the **Local RAG Chatbot** project! This application enables you to build a conversational AI that can answer questions based on your own documents, such as PDFs, CSVs, and web content. It leverages LangChain, Ollama, ChromaDB, and Streamlit to provide an interactive and intelligent chatbot experience.

<!-- Replace with actual image path or use relative path -->
![RAG Chatbot Architecture](./img/llm_rag.jpg)
*Example architecture of the RAG chatbot*

---

## 🛠️ Project Overview

This chatbot utilizes a Retrieval-Augmented Generation (RAG) approach, combining document retrieval with language model generation to provide accurate and context-aware responses. Key features include:

* **Document Upload**: Supports PDF and CSV uploads for content ingestion
* **Web Scraping**: Extracts information from specified URLs
* **ChromaDB Integration**: Stores and retrieves document embeddings for efficient search
* **Ollama LLM**: Powers the conversational AI with advanced language models
* **Streamlit Interface**: Provides an intuitive web interface for interaction

<!-- Replace with actual image path or use relative path -->
![Chatbot Interface Example](./img/llm_rag1.jpg)
*Example screenshot of the chatbot interface*

---

## 📦 Installation Guide

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd local-rag-chatbot
```

### 2. Create and Activate Conda Environment
```bash
conda env create -f environment.yml
conda activate local-rag-bot
```

### 3. Install Ollama Runtime
Download and install the Ollama runtime for your operating system:
* **Windows**: [Download Ollama for Windows](https://ollama.com/download)
* **macOS**: [Download Ollama for macOS](https://ollama.com/download)
* **Linux**: 
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### 4. Pull Required Models
After installing Ollama, pull the necessary models:
```bash
# For embeddings
ollama pull mxbai-embed-large

# For chat / LLM
ollama pull exaone3.5:2.4b
```

### 5. Create Required Directories
```bash
mkdir -p uploads chroma_db img
```

### 6. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### 7. Run the Streamlit Application
```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Document Processing**: Upload PDFs, CSVs, or provide URLs. The application processes these inputs and stores them in ChromaDB
2. **Embedding Generation**: Utilizes the `mxbai-embed-large` model to generate embeddings for the documents
3. **Conversational Interaction**: The `exaone3.5:2.4b` model powers the chatbot, providing responses based on the ingested content
4. **Chat History**: Maintains a conversation buffer to handle multi-turn interactions

---

## ⚙️ Configuration

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following content:

```env
# Model Configuration
MODEL=exaone3.5:2.4b
EMBEDDING_MODEL=mxbai-embed-large

# Chat Configuration
MAX_HISTORY=5
TEMPERATURE=0.7

# Database Configuration
CHROMA_DB_PATH=./chroma_db
UPLOAD_PATH=./uploads

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 📂 Project Structure

```
local-rag-chatbot/
├── app.py                  # Main Streamlit application
├── rag_backend.py          # Backend processing logic
├── environment.yml         # Conda environment configuration
├── requirements.txt        # Python package requirements
├── .env                    # Environment variables (create from .env.example)
├── .env.example           # Example environment variables
├── README.md              # This file
├── LICENSE                # License file
├── img/                   # Image assets
│   ├── llm_rag.jpg       # Architecture diagram
│   └── llm_rag1.jpg      # Interface screenshot
├── uploads/               # Folder for uploaded files (auto-created)
├── chroma_db/            # Folder for persistent embeddings (auto-created)
└── utils/                # Utility modules
    ├── __init__.py
    ├── document_processor.py
    └── embeddings.py
```

---

## 🚀 Usage

### Basic Usage
1. Start the application: `streamlit run app.py`
2. Navigate to the web interface (usually `http://localhost:8501`)
3. Upload documents or provide URLs for ingestion
4. Start chatting with your documents!

### Supported File Types
- **PDF**: Text extraction and processing
- **CSV**: Data analysis and question answering
- **Web URLs**: Content scraping and indexing

### Example Queries
- "What are the main topics covered in the uploaded document?"
- "Summarize the key findings from the CSV data"
- "What information is available about [specific topic]?"

---

## 🔧 Troubleshooting

### Common Issues

1. **Images not displaying in README**:
   ```bash
   # Ensure images exist in the correct path
   ls -la img/
   # If images are missing, add placeholder or update paths
   ```

2. **Ollama connection errors**:
   ```bash
   # Check if Ollama is running
   ollama list
   # Restart Ollama service if needed
   ```

3. **Model not found**:
   ```bash
   # Re-pull the models
   ollama pull mxbai-embed-large
   ollama pull exaone3.5:2.4b
   ```

4. **ChromaDB permissions**:
   ```bash
   # Ensure proper permissions for database directory
   chmod -R 755 chroma_db/
   ```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for the RAG framework
- [Ollama](https://ollama.com/) for local LLM deployment
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Streamlit](https://streamlit.io/) for the web interface

---

## ✅ Quick Start Checklist

- [ ] Ollama runtime installed and running
- [ ] Required models pulled (`mxbai-embed-large`, `exaone3.5:2.4b`)
- [ ] Conda environment created and activated
- [ ] Required directories created (`uploads/`, `chroma_db/`, `img/`)
- [ ] Environment variables configured
- [ ] Application running successfully

---

### 📝 Notes for First-Time Users

1. Make sure **Ollama runtime** is installed and running before starting the app
2. Pull the required models before running the application
3. The bot will **only answer questions based on the uploaded documents or web content**
4. Default URL included: [Yeongnam University](https://www.yu.ac.kr/english/index.do)
5. For best results, upload documents in supported formats and wait for processing to complete

---

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the [Issues](../../issues) page for similar problems
3. Create a new issue with detailed information about your problem
