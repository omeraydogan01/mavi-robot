import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")
    
    # Header ve logo yan yana
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=40)  # logo.png dosyasının yolu ve boyutu
    with col2:
        st.header("Soru Sor")

    # API key'i Streamlit secrets veya ortam değişkeninden al
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

    uploaded_file = st.file_uploader("Bir Doküman Yükleyin", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        # Başlık/alt başlıkları koruyan splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
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
            # Daha fazla chunk getir → daha doğru sonuç
            docs = vectorstore.similarity_search(user_question, k=5)

            # Daha güçlü LLM
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )

            # Daha derin çıkarım için map_reduce zinciri
            chain = load_qa_chain(llm, chain_type="map_reduce")
            answer = chain.run(input_documents=docs, question=user_question)

            st.write("💡 Cevap:")
            st.write(answer)

if __name__ == "__main__":
    main()
