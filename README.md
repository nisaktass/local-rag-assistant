# Local RAG Assistant ✈︎

A local, privacy-first Retrieval-Augmented Generation (RAG) application designed to answer questions from local documents with zero internet dependency.
Built with Streamlit, SQLite, SentenceTransformers, and local LLM inference via Ollama.

---

## Architecture & Tech Stack

This project implements a complete local RAG pipeline running entirely on a single machine:

* Frontend / UI: Streamlit (Interactive web interface)
* LLM Runtime / Inference: Ollama (Running local models with an OpenAI-compatible API)
* Embeddings: SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2) for semantic text vectorization.
* Vector Database: SQLite for lightweight, serverless local storage of document texts and vector embeddings.
* PDF Processing: pypdf for document ingestion and parsing.

---

## Project Structure

```text
local-rag-assistant/
├── app.py              # Main Streamlit application and RAG pipeline
└── requirements.txt    # Project Python dependencies
---

## Prerequisites & Installation

1. Clone the repository:
git clone https://github.com/nisaktass/local-rag-assistant.git
cd local-rag-assistant

2. Install dependencies:
pip install -r requirements.txt

3. Ensure Ollama is running locally:
Make sure Ollama is active on your machine and the required model is pulled.
The application connects via the local OpenAI-compatible endpoint (http://127.0.0.1:11434/v1).

---

## Running the Application

Launch the Streamlit interface with the following command:
streamlit run app.py

Open the provided local URL in your browser to start querying your documents offline!
