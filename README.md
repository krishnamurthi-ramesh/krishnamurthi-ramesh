# 🚀 Enterprise RAG System with Comprehensive Evaluation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready **Retrieval Augmented Generation (RAG)** system with comprehensive evaluation capabilities. This project provides an end-to-end pipeline for document ingestion, intelligent retrieval, LLM-powered answering with citations, and robust performance metrics.

## ✨ Key Features

- 📄 **Multi-format Document Support**: PDF, DOCX, TXT, and Markdown files
- 🔍 **Advanced Vector Search**: FAISS-powered semantic similarity search
- 🤖 **Flexible LLM Integration**: Support for both local (Ollama) and cloud (OpenAI) models
- 📊 **Comprehensive Evaluation**: Built-in metrics for retrieval and answer quality
- 🎯 **Citation System**: Automatic source attribution for generated answers
- 🖥️ **Interactive Web Interface**: Streamlit-based demo with real-time visualization
- ⚡ **Performance Monitoring**: Latency tracking and retrieval metrics

## 🏗️ Architecture

```
Enterprise RAG System
├── 📁 data/
│   ├── raw_documents/          # Source documents (PDF, DOCX, TXT)
│   ├── vector_store/           # FAISS index and metadata
│   └── evaluation/             # Test datasets and ground truth
├── 📁 src/
│   ├── data_processing.py      # Document parsing and chunking
│   ├── vector_store.py         # Embedding and vector operations
│   ├── rag_pipeline.py         # Core RAG logic and LLM integration
│   ├── evaluation.py           # Metrics and performance analysis
│   └── utils.py                # Configuration and utilities
├── 📁 app/
│   └── streamlit_app.py        # Web interface and demo
├── config.yaml                 # System configuration
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embeddings** | Sentence Transformers | Document and query vectorization |
| **Vector DB** | FAISS | Fast similarity search and retrieval |
| **LLM** | Ollama (Llama2) / OpenAI | Answer generation and reasoning |
| **Evaluation** | Ragas + Custom Metrics | Performance assessment |
| **Frontend** | Streamlit | Interactive web interface |
| **Visualization** | Plotly + Matplotlib | Charts and performance graphs |
| **Processing** | LangChain + PyPDF2 | Document parsing and chunking |

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git (for cloning)
- 4GB+ RAM (for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RagSystemWithCompleteEvaluation
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### LLM Setup

#### Option 1: Local LLM with Ollama (Recommended)

1. **Install Ollama** <mcreference link="https://ollama.com/" index="0">0</mcreference>
   - Download from [ollama.com](https://ollama.com/)
   - Available for macOS, Windows, and Linux

2. **Pull the model**
   ```bash
   ollama pull llama2
   ```

3. **Start Ollama server** (if not auto-started)
   ```bash
   ollama serve
   ```

#### Option 2: OpenAI API (Alternative)

1. **Set environment variable**
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   # macOS/Linux
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Update configuration**
   - Edit `config.yaml`
   - Change `llm.provider` from `ollama` to `openai`

### Launch the Application

```bash
streamlit run app/streamlit_app.py
```

Open your browser to the provided URL (typically `http://localhost:8501`)

## 📖 Usage Guide

### 1. Document Upload
- Use the web interface to upload PDF, DOCX, TXT, or Markdown files
- Documents are automatically processed and chunked

### 2. Build Vector Index
- Click "Process & Build Index" to create searchable embeddings
- This step is required before querying

### 3. Ask Questions
- Enter questions in natural language
- View generated answers with source citations
- Explore retrieved document chunks and relevance scores

### 4. Evaluate Performance
- Use built-in evaluation tools for:
  - **Retrieval Metrics**: Recall@K, NDCG@K, Precision@K
  - **Answer Quality**: Relevance scoring with Ragas
  - **Performance**: Response latency and throughput

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Embedding model
embeddings:
  model_name: "all-MiniLM-L6-v2"

# Text chunking
chunking:
  chunk_size: 1000
  chunk_overlap: 200

# Retrieval settings
retrieval:
  top_k: 5

# LLM configuration
llm:
  provider: "ollama"  # or "openai"
  model: "llama2"
  temperature: 0.7
  max_tokens: 512
```

## 📊 Evaluation Framework

### Supported Metrics

- **Retrieval Quality**
  - Recall@1, @3, @5
  - NDCG@3, @5
  - Precision@3, @5
  - Response latency

- **Answer Quality**
  - Answer relevance (via Ragas)
  - Citation accuracy
  - Factual consistency

### Evaluation Dataset Format

Create `data/evaluation/dataset.json`:

```json
[
  {
    "question": "What are the key requirements for the Data Scientist position?",
    "relevant_sources": ["job_description.pdf"],
    "ground_truth": "The position requires Python, ML experience, and a Master's degree."
  }
]
```

## 🔧 Development

### Project Structure Details

- **`src/data_processing.py`**: Document parsing, text extraction, and intelligent chunking
- **`src/vector_store.py`**: FAISS index management and similarity search
- **`src/rag_pipeline.py`**: Core RAG logic, prompt engineering, and LLM integration
- **`src/evaluation.py`**: Comprehensive metrics calculation and performance analysis
- **`src/utils.py`**: Configuration management and utility functions

### Adding New Features

1. **Custom Embeddings**: Modify `vector_store.py` to use different embedding models
2. **New LLM Providers**: Extend `rag_pipeline.py` with additional LLM integrations
3. **Enhanced Evaluation**: Add custom metrics in `evaluation.py`
4. **UI Improvements**: Customize the Streamlit interface in `app/streamlit_app.py`

## 🚀 Production Deployment

### Performance Considerations

- **Vector Store**: Consider PostgreSQL with pgvector for production scale
- **Caching**: Implement Redis for frequently accessed embeddings
- **Load Balancing**: Use multiple LLM instances for high throughput
- **Monitoring**: Add logging, metrics, and health checks

### Security Best Practices

- Store API keys in environment variables or secure vaults
- Implement authentication and authorization
- Sanitize user inputs and validate file uploads
- Use HTTPS in production deployments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Q: "FileNotFoundError: dataset.json"**
- A: Create the evaluation dataset file or use the sample provided in `data/evaluation/`

**Q: "Ollama connection failed"**
- A: Ensure Ollama server is running (`ollama serve`) and the model is pulled (`ollama pull llama2`)

**Q: "Out of memory errors"**
- A: Reduce `chunk_size` in config.yaml or use a smaller embedding model

**Q: "Slow response times"**
- A: Decrease `top_k` retrieval count or optimize chunk sizes

### Getting Help

- 📧 Open an issue on GitHub
- 💬 Check existing issues and discussions
- 📖 Review the configuration documentation

---

**Built with ❤️ for the AI community**