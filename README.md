# AI-Powered Telegram RAG Chatbot

An AI-powered Telegram chatbot built using Retrieval-Augmented Generation (RAG). The chatbot retrieves relevant information from uploaded PDF documents and uses a locally hosted Large Language Model (LLM) to generate contextual responses.

## 🚀 Features

- Telegram chatbot integration
- PDF document processing
- Text extraction and chunking
- Semantic search using embeddings
- FAISS vector database
- Retrieval-Augmented Generation (RAG)
- Local LLM integration using Ollama
- Qwen 2.5 Coder 3B model
- Prompt engineering
- Real-time responses through Telegram

## 🛠️ Technologies Used

- Python
- Telegram Bot API
- Ollama
- Qwen 2.5 Coder 3B
- Sentence Transformers
- FAISS
- PyPDF
- python-dotenv
- Git & GitHub

## 🏗️ Project Architecture

```text
User
  ↓
Telegram
  ↓
Telegram Bot API
  ↓
Python Backend
  ↓
User Query
  ↓
Sentence Transformer
  ↓
FAISS Vector Database
  ↓
Relevant Document Chunks
  ↓
Prompt + Context
  ↓
Ollama / Qwen 2.5 Coder 3B
  ↓
Generated Answer
  ↓
Telegram