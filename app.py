import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import load_qa_chain
from datetime import datetime
from io import BytesIO

LOG_FILE = "logs.xlsx"

# Secrets şifreleri
REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "1234")
RESET_PASSWORD = st.secrets.get("RESET_PASSWORD", "1234")  # Sıfırlama şifresi

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

# Rapor indirilebilir Excel dosyası
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

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")

    # Header ve logo
    col1, col2 = st.columns([1,6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana Soru Sor")

    # API key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı. Lütfen secrets veya environment değişkeni ekleyin.")
        st.stop()

    # Çoklu dosya yükleme
    uploaded_files = st.file_uploader(
        "📂 Bir veya birden fazla doküman yükleyin",
        type=["pdf", "docx", "xlsx", "csv"],
        accept_multiple_files=True
    )

    all_texts = []
    excel_tables = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split(".")[-1].lower()
            file_text = ""

            if ext == "pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_text += page.extract_text() or ""
                all_texts.append(file_text)

            elif ext == "docx":
                doc = Document(uploaded_file)
                file_text = "\n".join([p.text for p in doc.paragraphs])
                all_texts.append(file_text)

            elif ext in ["xlsx", "csv"]:
                if ext == "xlsx":
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                excel_tables.append(df)
                file_text = df.to_string(index=False)
                all_texts.append(file_text)

        full_text = "\n".join(all_texts)
        st.info(f"📚 {len(uploaded_files)} doküman yüklendi. Toplam {len(full_text.split())} kelime işlendi.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_text(full_text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)
        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("Sorunuzu yazın 👇")
        if user_question:
            # LLM ile Excel tablolarını sorgula
            table_answers = []
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            for df in excel_tables:
                prompt = f"Sana bir Excel tablosu verdim:\n{df.head(10).to_string(index=False)}\nBu tabloya göre soruyu cevapla: {user_question}"
                table_answer = llm.call_as_llm(prompt)
                table_answers.append(table_answer)

            # Metin tabanlı belgeler için QA zinciri
            docs = vectorstore.similarity_search(user_question, k=6)
            chain = load_qa_chain(llm, chain_type="stuff")
            text_answer = chain.run(input_documents=docs, question=user_question)

            # Sonuçları birleştir
            final_answer = text_answer
            if table_answers:
                final_answer += "\n\n📊 Excel/CSV tablosundan alınan cevaplar:\n" + "\n".join(table_answers)

            st.subheader("💡 Cevap")
            st.success(final_answer)
            log_question(user_question, final_answer)

    # Sidebar
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
