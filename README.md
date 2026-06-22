# 🧪 Rick Sanchez Bot

A RAG-based character chatbot that responds as Rick Sanchez from Rick and Morty, grounded in actual show dialogues.

## Architecture
User Query → Sentence Transformer → Query Embedding → Cosine Similarity over 420 Rick dialogues → System Prompt + Retrieved Dialogues + History → Groq/Llama → Rick-style Response

## Tech Stack
- LLM: Llama 3.1 8B via Groq API
- Embeddings: all-MiniLM-L6-v2 (Sentence Transformers)
- Retrieval: Cosine similarity over Rick and Morty transcripts
- Memory: Sliding window conversation history (last 6 messages)
- UI: Streamlit

## Setup
1. pip install -r requirements.txt
2. export GROQ_API_KEY=your_key
3. Download dataset: https://www.kaggle.com/datasets/andradaolteanu/rickmorty-scripts
4. streamlit run rick_bot.py
