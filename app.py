import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    
    # Header ve logo
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=60)  # Logoyu biraz büyüttük
    with col2:
        st.header("📚 PDF ile Hızlı Sohbet")

    # API key'i Streamlit secrets veya ortam değişkeninden al
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

    uploaded_file = st.file_uploader("Bir PDF yükleyin", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # Başlık/alt başlıkları koruyan splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Daha büyük chunk → daha az parça → hızlı
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Daha doğru embedding modeli
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

        # FAISS vektör veritabanı
        vectorstore = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input("Sorunuzu yazın 👇")

        if user_question:
            # Daha az chunk getir → daha hızlı
            docs = vectorstore.similarity_search(user_question, k=3)

            # Daha hızlı LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",  # gpt-4o-mini yerine daha hızlı
                temperature=0,
                api_key=api_key
            )

            # Daha hızlı zincir
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)

            st.write("💡 Cevap:")
            st.write(answer)

if __name__ == "__main__":
    main()
