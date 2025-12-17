# 🎥 YouTube RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about any YouTube video using its transcript.

## Features
- Paste YouTube URL
- Automatic transcript extraction
- Vector search over transcript
- Grounded answers using OpenAI LLM
- Built with LangChain + Streamlit

## Tech Stack
- Python
- LangChain
- OpenAI API
- FAISS / Chroma
- Streamlit

## Setup

```bash
git clone https://github.com/yourusername/youtube-rag-chatbot.git
cd youtube-rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
