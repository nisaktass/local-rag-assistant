import streamlit as st
import sqlite3
import json
import re
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import os

# =========================================================
# 1. SAYFA VE UYGULAMA YAPILANDIRMASI
# =========================================================
st.set_page_config(
    page_title="Yerel Doküman Asistanı",
    page_icon="✈️",
    layout="wide"
)

DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
DB_PATH = os.path.join(os.getcwd(), "rag_knowledge_base.db")

# CPU'yu yormayan hafif embedding modeli
@st.cache_resource
def load_embedder():
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return SentenceTransformer(model_name, local_files_only=True)

embedder = load_embedder()

@st.cache_resource
def get_openai_client():
    return OpenAI(
        base_url="http://127.0.0.1:11434/v1",
        api_key="ollama"
    )

client = get_openai_client()

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# =========================================================
# 2. KNOWLEDGE BASE (VERİTABANI VE İNDEKSLEME)
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        content TEXT,
        embedding TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

def add_file_to_db(uploaded_file):
    text_content = ""
    if uploaded_file.name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_content += extracted + "\n\n"
    else:
        text_content = uploaded_file.read().decode("utf-8")
        
    if not text_content.strip():
        return False, "Dosyadan metin okunamadı."

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text_content) if len(p.strip()) > 15]
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) < 1200:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
            
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        return False, "Metin parçalanamadı."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for idx, chunk in enumerate(chunks):
        vec = embedder.encode(chunk).tolist()
        cursor.execute(
            "INSERT INTO document_chunks (title, content, embedding) VALUES (?, ?, ?)",
            (f"{uploaded_file.name} - Parça {idx+1}", chunk, json.dumps(vec))
        )
        
    conn.commit()
    conn.close()
    return True, f"'{uploaded_file.name}' ({len(chunks)} parça) başarıyla indekslendi!"

def get_top_chunks(query, top_k=3):
    try:
        query_vec = embedder.encode(query).tolist()
    except Exception:
        return []

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT title, content, embedding FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []

    results = []
    for title, content, emb_json in rows:
        doc_vec = json.loads(emb_json)
        score = cosine_similarity(query_vec, doc_vec)
        results.append((score, title, content))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]

# =========================================================
# 3. STREAMLIT ARAYÜZÜ (GUI)
# =========================================================
st.title("🧠 Yerel Doküman Asistanı")
st.caption("Hafif ve Kararlı RAG Asistanı")

with st.sidebar:
    st.header("📂 Dosya & Veritabanı Yönetimi")
    st.text(f"Çalışma Dizini:\n{DESKTOP_PATH}")
    
    uploaded_files = st.file_uploader(
        "PDF veya TXT Yükle", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        index_btn = st.button("İndeksle", use_container_width=True)
    with col2:
        clear_btn = st.button("Sıfırla", use_container_width=True)

    if clear_btn:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM document_chunks")
        conn.commit()
        conn.close()
        st.session_state.messages = []
        st.success("Veritabanı sıfırlandı!")

    if index_btn and uploaded_files:
        with st.spinner("Dosyalar işleniyor..."):
            for f in uploaded_files:
                success, msg = add_file_to_db(f)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
                    
    st.markdown("---")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM document_chunks")
    count = cur.fetchone()[0]
    conn.close()
    st.metric(label="Kayıtlı Parça Sayısı", value=count)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    chunks = get_top_chunks(user_query, top_k=2)

    with st.chat_message("assistant"):
        if not chunks or chunks[0][0] < 0.30:
            bot_response = "Bu bilgi veritabanında bulunmuyor."
            st.markdown(bot_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )
        else:
            context_text = "\n\n---\n\n".join(
                [
                    f"[Kaynak: {c[1]} | Benzerlik: {c[0]:.2f}]\n{c[2]}"
                    for c in chunks
                ]
            )

            with st.expander("📚 Getirilen Bağlam (Context)"):
                st.text(context_text)

            system_prompt = """
Sen KATI KURALLARA bağlı bir arama motorusun. Senin dış dünya hakkında hiçbir bilgin YOKTUR. Sadece sana verilen METİN'i kullanabilirsin.

KURALLAR:
1. Sorunun cevabı METİN'in içinde net bir şekilde geçmiyorsa, ASLA kendi hafızandan, ezberinden veya dış dünyadan cevap verme.
2. Eğer aranan bilgi METİN'de yoksa, kesinlikle ve sadece şunu yaz:
   "Bu bilgi veritabanında bulunmuyor."
3. Asla yorum yapma, dış bilgi ekleme veya genel kültürünü kullanma.
"""
            
            user_prompt = f"METİN:\n{context_text}\n\nSORU:\n{user_query}\n\nCEVAP:"

            try:
                with st.spinner("Yanıt üretiliyor..."):
                    response = client.chat.completions.create(
                        model="qwen2.5:1.5b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=400
                    )
                    
                    bot_response = response.choices[0].message.content.strip()
                    st.markdown(bot_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": bot_response}
                    )
                    
            except Exception as e:
                st.error(f"❌ Hata oluştu: {e}")