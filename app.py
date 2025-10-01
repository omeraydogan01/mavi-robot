import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime

LOG_FILE = "logs.csv"

def log_question(question, answer):
    """Soruları CSV'ye kaydet"""
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

def show_report():
    """En çok sorulan soruları raporla"""
    st.subheader("📊 Raporlama")
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        st.write("Toplam soru sayısı:", len(df))

        top_questions = df["question"].value_counts().head(5)
        st.write("En çok sorulan sorular:")
        st.table(top_questions)
    else:
        st.info("Henüz rapor oluşturulacak veri yok. Lütfen birkaç soru sorun.")

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.header("📚 PDF ile Sohbet ve Raporlama")

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

            # 📌 Log kaydı
            log_question(user_question, answer)

    # 📊 Rapor kısmı her zaman göster
    show_report()

if __name__ == "__main__":
    main()
