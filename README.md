# 💬 RAG Chatbot with Local LLM & FAISS

This project is a simple Retrieval-Augmented Generation (RAG) chatbot built using:

- 🔍 FAISS for vector search
- 📄 PDF document ingestion
- 🧠 Local LLM (e.g., TinyLlama / Mistral via HuggingFace Transformers)
- 🧠 HuggingFace embeddings (`all-MiniLM-L6-v2`)
- 🖥️ No need for OpenAI API
- 🌐 Streamlit-based web app interface

---

## 🚀 Features

- Upload a PDF and ask questions based on its content.
- Uses local language models (no external API keys required).
- Displays sources for transparent answers.
- Fully offline (after initial model + dependency setup).

---

## 🧩 Tech Stack

| Component         | Tool/Library                                |
|------------------|----------------------------------------------|
| Interface        | Streamlit                                    |
| LLM              | TinyLlama / Mistral (via `transformers`)     |
| Embeddings       | `sentence-transformers/all-MiniLM-L6-v2`     |
| Vector Store     | FAISS                                         |
| RAG Framework    | LangChain                                     |
| PDF Parsing      | Unstructured / PyPDF / pdfminer              |

---

## 📦 Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install required dependencies
pip install -r requirements.txt
