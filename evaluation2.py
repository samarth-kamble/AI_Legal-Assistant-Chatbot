import os
import tempfile
import torch
import whisper
import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel

# ---------- Configuration ----------
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
COLLECTION_NAME = "document_qa_collection"

# ---------- Load Environment Variables ----------
load_dotenv()
api_token = os.getenv("GROQ_API_KEY")

# ---------- Load Whisper Model ----------
@st.cache_resource
def load_whisper_model():
    return whisper.load_model(WHISPER_MODEL_SIZE)

# ---------- Load BERT Model & Tokenizer ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# ---------- ChromaDB Setup ----------
chroma_client = chromadb.PersistentClient(path="chroma_db_audio")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---------- Helper Functions ----------
def get_bert_embedding(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def transcribe_audio(audio_file):
    whisper_model = load_whisper_model()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    try:
        result = whisper_model.transcribe(tmp_path)
        return result["text"]
    finally:
        os.unlink(tmp_path)

def query_documents(question, n_results=5):
    query_embedding = get_bert_embedding(question)
    if query_embedding:
        results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
        return [doc for sublist in results["documents"] for doc in sublist]
    else:
        return []

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an expert legal assistant. "
        "Only answer questions related to the Indian Penal Code (IPC). "
        "If the question is irrelevant, in another language, or lacks context, say 'No context found'. "
        "Use the context provided below to answer:\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-70b-8192",
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Groq API Error] {e}")
        return "Sorry, I couldn't generate a response."

# Optional: Typing Effect
def type_writer_effect(text, delay=0.02):
    placeholder = st.empty()
    output = ""
    for char in text:
        output += char
        placeholder.markdown(f"**LegalBot:** {output}")
        time.sleep(delay)

# ---------- UI Styling ----------
st.set_page_config(page_title="⚖️ Legal Assistant", layout="centered")

st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
    font-family: 'Segoe UI', sans-serif;
}
.stChatMessage[data-testid="stChatMessage"] {
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
    max-width: 85%;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
    background-color: #1F2937;
    border-left: 5px solid #00FFAA;
}
.stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
    background-color: #111827;
    border-right: 5px solid #00FFAA;
}
h1, h2, h3 {
    color: #00FFAA !important;
}
.stChatInput input {
    background-color: #1E1E1E !important;
    color: #FFFFFF !important;
    border: 1px solid #3A3A3A !important;
    padding: 10px;
}
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 8px;
}
section[data-testid="stFileUploader"] > label {
    color: #00FFAA;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- App Title ----------
st.title("⚖️ AI-Powered Legal Assistance")
st.markdown("##### Ask questions related to the **Indian Penal Code (IPC)** — with **voice or text**")

# ---------- Audio Upload ----------
with st.expander("🎙️ Upload an audio file (MP3)"):
    audio_bytes = st.file_uploader("Upload your legal query as audio", type=["mp3"], label_visibility="collapsed")

user_input = ""
if audio_bytes:
    with st.spinner("🧠 Transcribing audio..."):
        user_input = transcribe_audio(audio_bytes)
        st.session_state.audio_transcript = user_input

# ---------- Text Chat Input ----------
user_text = st.chat_input("💬 Ask your IPC-related question...")
user_input = user_text if user_text else st.session_state.get("audio_transcript", "")

# ---------- Chat Response Flow ----------
if user_input:
    if "audio_transcript" in st.session_state:
        del st.session_state.audio_transcript

    with st.chat_message("user"):
        st.markdown(f"**You:** {user_input}")

    with st.spinner("🔍 Searching for legal context..."):
        docs = query_documents(user_input)
        answer = generate_response(user_input, docs)

    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown("**LegalBot:**")
        st.markdown(answer)
        # type_writer_effect(answer)  # Use this instead for typing animation