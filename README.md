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
### 3. Start the chatbot

``` bash
python chatbot.py
```

## 🛑 GitHub File Size Limit
The .gguf model is excluded from version control to respect GitHub's 100MB file size limit.
Make sure to add it manually to the models/ directory before running the app.

## 🧾 License
This project is open source and available under the MIT License.

Created by Erika Chang – 2025
