import os
import io
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from openpyxl import Workbook
from openpyxl.writer.excel import save_virtual_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.workbook.protection import WorkbookProtection

# ---- CONFIG ----
LOG_FILE = "logs.xlsx"
st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png", layout="wide")

# ---- SIDEBAR KAPALI ----
st.markdown(
    """
    <style>
    /* Sidebar başlangıçta kapalı */
    .css-1d391kg {display: none;}
    </style>
    """, unsafe_allow_html=True
)

# ---- HEADER ----
col1, col2 = st.columns([1, 6])
with col1:
    st.image("logo.png", width=120)
with col2:
    st.header("📄 Dokümana Soru Sor ve Raporla")

# ---- API KEY ----
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ API key bulunamadı. Lütfen environment veya secrets ekleyin.")
    st.stop()

# ---- Rapor indirme şifresi ----
REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "süpergizlisifre123")

# ---- Dosya yükleme ----
uploaded_file = st.file_uploader("📂 PDF veya Word dosyası yükleyin", type=["pdf", "docx"])
if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif file_extension == "docx":
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        st.error("Sadece PDF veya Word dosyası yükleyebilirsiniz.")
        st.stop()
    
    st.info(f"📄 Yüklenen doküman toplam **{len(text.splitlines())}** satır içeriyor.")

    # ---- TEXT SPLITTER ----
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    # ---- EMBEDDINGS & VECTORSTORE ----
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
    @st.cache_resource
    def create_vectorstore(chunks, embeddings):
        return FAISS.from_texts(chunks, embeddings)
    vectorstore = create_vectorstore(chunks, embeddings)

# ---- Kullanıcı sorusu ----
user_question = st.text_input("Sorunuzu yazın 👇")
if uploaded_file and user_question:
    docs = vectorstore.similarity_search(user_question, k=5)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    chain = load_qa_chain(llm, chain_type="map_reduce")
    answer = chain.run(input_documents=docs, question=user_question)

    st.subheader("💡 Cevap")
    st.success(answer)

    # ---- LOG KAYDI EXCEL ----
    if os.path.exists(LOG_FILE):
        wb = pd.read_excel(LOG_FILE, engine="openpyxl")
        df = pd.read_excel(LOG_FILE, engine="openpyxl")
    else:
        df = pd.DataFrame(columns=["timestamp", "question", "answer"])
    
    new_row = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
               "question": user_question, "answer": answer}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel(LOG_FILE, index=False)

# ---- RAPOR İNDİRME ----
with st.sidebar:
    st.subheader("📥 Rapor İndir")
    password = st.text_input("Şifreyi girin", type="password")
    if st.button("Excel olarak indir"):
        if password != REPORT_PASSWORD:
            st.error("❌ Yanlış şifre!")
        else:
            if os.path.exists(LOG_FILE):
                df = pd.read_excel(LOG_FILE, engine="openpyxl")
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Rapor")
                    writer.save()
                st.success("✅ Rapor hazır! Aşağıdan indirilebilir.")
                st.download_button(
                    label="📥 Excel İndir",
                    data=output.getvalue(),
                    file_name="rapor.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Henüz rapor oluşturulmadı.")
