from pathlib import Path
import os
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────
RICK_SYSTEM_PROMPT = """You are Rick Sanchez from Rick and Morty. You are a genius scientist who is cynical, sarcastic, and nihilistic. You burp mid-sentence sometimes. You call the user Morty. You use phrases like 'Wubba lubba dub dub'. You never break character."""

@st.cache_resource
def init_groq_client():
    """Initialize Groq client with proper error handling"""
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, "secrets") else None
    
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error(
            "🔑 **Missing GROQ_API_KEY**\n\n"
            "**Local:** Add to `.streamlit/secrets.toml`\n"
            "**Streamlit Cloud:** Add to Settings → Secrets"
        )
        st.stop()
    
    return Groq(api_key=api_key


# ── Load models & data ───────────────────────────────
@st.cache_resource
def load_models():
    client = Groq(api_key=GROQ_API_KEY)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embedding_model

@st.cache_resource
def load_rick_data():
    # Get the directory where this script is located
    base_dir = Path(__file__).parent
    csv_path = base_dir / "RickAndMortyScripts.csv"
    
    # Fallback for Streamlit Cloud deployment
    if not csv_path.exists():
        csv_path = Path("RickAndMortyScripts.csv")
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at {csv_path}. "
            f"Make sure RickAndMortyScripts.csv is in the same directory as rick_bot.py"
        )
    
    df = pd.read_csv(csv_path)
    rick_lines = df[df['name'] == 'Rick']['line'].tolist()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    rick_embeddings = embedding_model.encode(rick_lines, show_progress_bar=False)
    
    return rick_lines, rick_embeddings

# ── RAG Functions ────────────────────────────────────
def get_relevant_dialogues(query, rick_lines, rick_embeddings, top_k=3):
    _, embedding_model = load_models()
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, rick_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [rick_lines[i] for i in top_indices]

def rick_rag_chat(user_query, conversation_history, client, rick_lines, rick_embeddings):
    # Retrieve relevant dialogues
    relevant_dialogues = get_relevant_dialogues(user_query, rick_lines, rick_embeddings)
    context = "\n".join(relevant_dialogues)

    # Build messages — system + history + current query
    messages = [
        {
            "role": "system",
            "content": f"{RICK_SYSTEM_PROMPT}\n\nHere are some of your actual quotes:\n{context}"
        }
    ] + conversation_history + [
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=200
    )
    return response.choices[0].message.content

# ── Streamlit UI ─────────────────────────────────────
st.set_page_config(page_title="Rick Sanchez Bot", page_icon="🧪")
st.title("🧪 Rick Sanchez Bot")
st.caption("Wubba lubba dub dub, Morty!")

# Session state initialize
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Load everything
client, _ = load_models()
rick_lines, rick_embeddings = load_rick_data()

# Display chat history
for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Talk to Rick..."):
    # User message display
    with st.chat_message("user"):
        st.write(user_input)

    # Rick response
    with st.chat_message("assistant"):
        with st.spinner("Rick is thinking... *burp*"):
            response = rick_rag_chat(
                user_input,
                st.session_state.conversation_history[-6:],  # sliding window
                client,
                rick_lines,
                rick_embeddings
            )
        st.write(response)

    # History update
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
