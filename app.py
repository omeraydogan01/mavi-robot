import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from io import BytesIO
from PIL import Image
import pytesseract

LOG_FILE = "logs.xlsx"

REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "1234")
RESET_PASSWORD = st.secrets.get("RESET_PASSWORD", "1234")

# Log kaydetme
def log_question(question, answer):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }])
    if os.path.exists(LOG_FILE):
        df_old = pd.read_excel(LOG_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_excel(LOG_FILE, index=False)

# Excel rapor
def get_report():
    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return output
    return None

# Sıfırlama
def reset_logs():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        st.success("✅ Soru-cevap geçmişi sıfırlandı!")

# Dosya/görselden metin çıkarma
def extract_text(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""
    if ext == "pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
    elif ext == "docx":
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext in ["jpg","jpeg","png"]:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
    else:
        st.warning(f"{uploaded_file.name} türü desteklenmiyor.")
    return text

# Vectorstore cache
@st.cache_resource
def create_vectorstore(text_chunks, embeddings):
    return FAISS.from_texts(text_chunks, embeddings)

# Ana fonksiyon
def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")
    col1, col2 = st.columns([1,6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Doküman veya Görsele Soru Sor")

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı.")
        st.stop()

    # Dosya yükleme
    uploaded_files = st.file_uploader(
        "📂 Bir veya birden fazla doküman/görsel yükleyin",
        type=["pdf","docx","jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        all_texts = [extract_text(f) for f in uploaded_files]
        full_text = "\n".join(all_texts)
        st.info(f"📚 {len(uploaded_files)} dosya/görsel yüklendi. Toplam {len(full_text.split())} kelime.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("Sorunuzu yazın 👇", key="question_input")
        if st.button("Sor", key="ask_button") and user_question.strip():
            docs = vectorstore.similarity_search(user_question, k=6)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)
            st.subheader("💡 Cevap")
            st.success(answer)
            log_question(user_question, answer)
            st.session_state.question_input = ""  # Input temizleme

    # Sidebar
    with st.sidebar.expander("📊 Rapor & Yönetim", expanded=False):
        st.subheader("📥 Rapor İndir")
        report_pass = st.text_input("Rapor şifresi", type="password")
        if st.button("📄 Excel İndir", key="download_btn"):
            if report_pass == REPORT_PASSWORD:
                report_file = get_report()
                if report_file:
                    st.download_button(
                        label="Excel indir",
                        data=report_file,
                        file_name="soru_cevap_raporu.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Henüz kaydedilmiş soru yok.")
            else:
                st.error("❌ Yanlış şifre!")

        st.subheader("⚠️ Soru Geçmişini Sıfırla")
        reset_pass = st.text_input("Sıfırlama şifresi", type="password")
        if st.button("🗑️ Sıfırla Geçmiş", key="reset_btn"):
            if reset_pass == RESET_PASSWORD:
                reset_logs()
            else:
                st.error("❌ Yanlış şifre!")

if __name__ == "__main__":
    main()
