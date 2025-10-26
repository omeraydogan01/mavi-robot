import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from io import BytesIO

LOG_FILE = "logs.xlsx"

# Secrets şifreleri
REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "1234")
RESET_PASSWORD = st.secrets.get("RESET_PASSWORD", "1234")

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

def get_report():
    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return output
    return None

def reset_logs():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        st.success("✅ Soru-cevap geçmişi sıfırlandı!")

# Basit vector store
class SimpleVectorStore:
    def __init__(self, chunks):
        self.chunks = chunks

    def similarity_search(self, query, k=5):
        return self.chunks[:k]

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")

    col1, col2 = st.columns([1,6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana Soru Sor")

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı.")
        st.stop()

    uploaded_files = st.file_uploader(
        "📂 PDF veya DOCX yükleyin",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    all_texts = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split(".")[-1].lower()
            file_text = ""
            if ext == "pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_text += page.extract_text() or ""
            elif ext == "docx":
                doc = Document(uploaded_file)
                file_text = "\n".join([p.text for p in doc.paragraphs])
            all_texts.append(file_text)

        full_text = "\n".join(all_texts)
        st.info(f"📚 {len(uploaded_files)} doküman yüklendi. Toplam {len(full_text.split())} kelime işlendi.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        vectorstore = SimpleVectorStore(chunks)

        user_question = st.text_input("Sorunuzu yazın 👇")
        if user_question:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            docs = vectorstore.similarity_search(user_question, k=6)
            chain = load_qa_chain(llm, chain_type="stuff")
            text_answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("💡 Cevap")
            st.success(text_answer)
            log_question(user_question, text_answer)

    with st.sidebar.expander("📊 Rapor & Yönetim", expanded=False):
        st.subheader("📥 Rapor İndir")
        report_pass = st.text_input("Rapor şifresi", type="password")
        if st.button("📄 Excel İndir"):
            if report_pass == REPORT_PASSWORD:
                report_file = get_report()
                if report_file:
                    st.download_button(
                        label="Excel olarak indir",
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
        if st.button("🗑️ Sıfırla Geçmiş"):
            if reset_pass == RESET_PASSWORD:
                reset_logs()
            else:
                st.error("❌ Yanlış şifre!")

if __name__ == "__main__":
    main()
