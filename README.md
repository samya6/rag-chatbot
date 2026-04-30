# RAG PDF Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on a PDF document.

## Features
- Upload a PDF and ask questions
- Semantic search using FAISS
- AI-generated responses using a Hugging Face model
- Simple web interface with Gradio

## Tech Stack
- LangChain
- FAISS
- Hugging Face Transformers
- Sentence Transformers
- Gradio

## How it works
1. PDF is loaded and split into chunks
2. Text is converted into embeddings
3. FAISS retrieves relevant chunks
4. LLM generates answers based on retrieved context

## Run locally

```bash
python app.py

## Push to GitHub

If you skip this, the project has no value externally.

### In terminal:

```bash
git init
git add .
git commit -m "RAG PDF Chatbot"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main