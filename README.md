# 🧠 RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that combines a local language model with a document-based knowledge base to provide contextual answers.

## 📚 Overview

This project implements a question-answering chatbot using:

- 🔍 **Document retrieval** (based on local PDFs or text files)
- 🧠 **Local LLM inference** using a `gguf` quantized model (e.g., TinyLlama)
- 🗃️ **Vector database** with FAISS
- 🖥️ **Gradio UI** for easy interaction

## 💡 Motivation

The idea is to create a privacy-preserving assistant that can answer questions based on your own documents without relying on external APIs.

## 📁 Project Structure

``` bash
rag_chatbot/
│
docs/
├── refund_policy.txt
├── shipping_info.txt
└── product_faq.txt
│
├── models/
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
│
├── chatbot.py
└── requirements.txt

```

## 🚀 How to Run

### 1. Clone the repository

``` bash
git clone https://github.com/erika-chang/rag_chatbot.git
cd rag_chatbot
```

### 2. Set up your environment
We recommend using a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```
### 3. Add your model
Place your .gguf model file in the models/ folder (not tracked by Git).
You can use a model like TinyLlama.

Example:

``` bash
models/
└── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 4. Add your documents
Put your custom documents (PDFs, TXTs, etc.) in the data/ folder.

``` bash
data/
└── your-documents.pdf
```

### 5. Run ingestion

``` bash
python ingest.py
```

This will parse, split, and embed your documents, storing the vectors in a local FAISS database.

### 6. Start the chatbot

``` bash
python app.py
```

The Gradio interface will open in your browser at http://localhost:7860.

## 🛑 GitHub File Size Limit
The .gguf model is excluded from version control to respect GitHub's 100MB file size limit.
Make sure to add it manually to the models/ directory before running the app.

## 🧾 License
This project is open source and available under the MIT License.

Created by Erika Chang – 2025
