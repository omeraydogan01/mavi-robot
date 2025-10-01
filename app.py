import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from io import BytesIO

LOG_FILE = "logs.csv"
REPORT_PASSWORD = "1234"  # 📌 Şifreni buraya yaz

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

def download_report():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_buffer.seek(0)

        st.download_button(
            label="📥 Raporu İndir (CSV)",
            data=csv_buffer,
            file_name="soru_raporu.csv",
            mime="text/csv"
        )

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.header("📚 PDF ile Sohbet")

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

    uploaded_file = st.file_uploader("Bir PDF yükleyin", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Sorunuzu yazın 👇")

        if user_question:
            docs = vectorstore.similarity_search(user_question, k=3)
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)

            st.write("💡 Cevap:")
            st.write(answer)

            log_question(user_question, answer)

    # 📌 Sidebar → rapor indirme alanı
    with st.sidebar:
        st.subheader("🔑 Rapor İndirme")
        password_input = st.text_input("Şifreyi giriniz:", type="password")
        if password_input == REPORT_PASSWORD:
            st.success("✅ Doğru şifre")
            download_report()
        elif password_input:
            st.error("❌ Yanlış şifre")

if __name__ == "__main__":
    main()
