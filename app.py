import os
import streamlit as st
import pandas as pd
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Soru-cevap logları
qa_logs = []

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png", layout="wide")

    # CSS - text_input mavi kalın çerçeve
    st.markdown("""
        <style>
        div[data-testid="stTextInput"] > div > input {
            border: 3px solid #1E90FF;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar: Rapor indirme
    st.sidebar.header("📑 Raporlama")
    password = st.sidebar.text_input("Rapor şifresi", type="password")
    if st.sidebar.button("📥 Raporu Excel Olarak İndir"):
        if password == "1234":  # İstediğin şifreyi değiştir
            if qa_logs:
                df = pd.DataFrame(qa_logs)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df.to_excel(writer, index=False, sheet_name="Q&A Logs")
                st.sidebar.download_button(
                    label="📊 Excel Raporunu İndir",
                    data=buffer,
                    file_name="rapor.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download-excel"
                )
            else:
                st.sidebar.warning("Henüz hiç soru sorulmadı.")
        else:
            st.sidebar.error("❌ Hatalı şifre!")

    # Header ve logo yan yana
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana Soru Sor")

    # API key kontrolü
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı. Lütfen secrets veya environment değişkeni ekleyin.")
        st.stop()

    uploaded_file = st.file_uploader("📂 Doküman yükleyin", type=["pdf", "docx"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        if file_extension == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif file_extension == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Sadece PDF veya Word dosyaları yükleyebilirsiniz.")
            st.stop()
        
        st.info(f"📄 Yüklenen doküman toplam **{len(text.splitlines())}** satır içeriyor.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Embedding modeli
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

        # FAISS vektör veritabanı (cache)
        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)

        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu (tek satır input)
        user_question = st.text_input("Sorunuzu yazın 👇")

        if user_question:
            docs = vectorstore.similarity_search(user_question, k=6)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("💡 Cevap")
            st.success(answer)

            # Log kaydı
            qa_logs.append({
                "question": user_question,
                "answer": answer
            })

if __name__ == "__main__":
    main()
