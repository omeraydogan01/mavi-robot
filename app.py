import os
import streamlit as st
import sqlite3
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# ---------------------- DATABASE ----------------------
DB_FILE = "qa_logs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS qa_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_qa(question, answer):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO qa_logs (timestamp, question, answer) VALUES (datetime('now'), ?, ?)",
              (question, answer))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM qa_logs", conn)
    conn.close()
    return df

# ---------------------- APP ----------------------
def main():
    # Sidebar kapalı başlasın
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png", initial_sidebar_state="collapsed")
    
    init_db()  # DB hazırla

    # Header ve logo
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana Soru Sor")

    # API key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı. Lütfen secrets veya environment değişkeni ekleyin.")
        st.stop()

    # ---------------- PDF/DOCX ----------------
    uploaded_file = st.file_uploader("📂 Doküman yükleyin", type=["pdf", "docx"])
    text = ""
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            st.error("Sadece PDF veya Word dosyaları yükleyebilirsiniz.")
            st.stop()
        st.info(f"📄 Yüklenen doküman toplam {len(text.splitlines())} satır içeriyor.")

        # Splitter ve vektör oluşturma
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)

        vectorstore = create_vectorstore(chunks, embeddings)

    # ---------------- SORU ----------------
    user_question = st.text_input("Sorunuzu yazın 👇")
    if user_question and uploaded_file is not None:
        docs = vectorstore.similarity_search(user_question, k=5)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=user_question)

        st.subheader("💡 Cevap")
        st.success(answer)

        # DB kaydı
        log_qa(user_question, answer)

    # ---------------- RAPOR ----------------
    with st.sidebar:
        st.subheader("📊 Rapor")
        password = st.text_input("🔒 Rapor için şifre", type="password")
        if password == st.secrets.get("REPORT_PASSWORD"):
            df_logs = get_logs()
            if not df_logs.empty:
                st.download_button(
                    label="📥 Excel olarak indir",
                    data=df_logs.to_excel(index=False, engine='openpyxl'),
                    file_name="qa_logs.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Henüz kayıtlı soru yok.")
        elif password:
            st.error("Yanlış şifre!")

if __name__ == "__main__":
    main()
