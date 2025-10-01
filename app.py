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

# Sabitler
LOG_FILE = "logs.csv"
REPORT_PASSWORD_KEY = "REPORT_PASSWORD"  # Streamlit secrets içindeki key

# Soru-cevapları kaydet
def log_question(question, answer):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
    }])
    if os.path.exists(LOG_FILE):
        df_old = pd.read_csv(LOG_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(LOG_FILE, index=False)

# Raporu Excel'e çevir
def generate_excel():
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Rapor")
    return output.getvalue()

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png", layout="wide")
    
    # Header ve logo
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("📚 Dokümana Soru Sor")

    # API Key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı! Secrets veya environment değişkeni ekleyin.")
        st.stop()

    # PDF/Word yükleme
    uploaded_file = st.file_uploader("📂 Doküman yükleyin", type=["pdf", "docx"])
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Sadece PDF veya Word yükleyebilirsiniz.")
            st.stop()
        st.info(f"📄 Doküman {len(text.splitlines())} satır içeriyor.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Embeddings ve FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)
        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("Sorunuzu yazın 👇")
        if user_question:
            docs = vectorstore.similarity_search(user_question, k=5)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="map_reduce")
            answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("💡 Cevap")
            st.success(answer)

            # Soru-cevap loglama
            log_question(user_question, answer)

    # Sidebar: Rapor indir
    with st.sidebar.expander("📊 Rapor İndir", expanded=False):
        password = st.text_input("Şifreyi girin", type="password")
        if password:
            report_password = os.getenv(REPORT_PASSWORD_KEY) or st.secrets.get(REPORT_PASSWORD_KEY)
            if password == report_password:
                excel_data = generate_excel()
                if excel_data:
                    st.download_button(
                        label="📥 Excel olarak indir",
                        data=excel_data,
                        file_name="rapor.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Henüz rapor yok.")
            else:
                st.error("❌ Yanlış şifre!")

if __name__ == "__main__":
    main()
