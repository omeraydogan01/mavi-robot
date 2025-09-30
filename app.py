import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.header("🤖 PDF'inle Sohbet Et")

    # 🔑 API key'i Streamlit secrets veya ortam değişkeninden al
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]

    uploaded_file = st.file_uploader("Bir PDF yükleyin", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # ✅ Recursive splitter ile daha iyi parçalama
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # ✅ Daha büyük embedding modeli
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

        # ✅ FAISS vektör veritabanı
        vectorstore = FAISS.from_texts(chunks, embeddings)

        # Kullanıcıdan soru al
        user_question = st.text_input("Sorunuzu buraya yazın:")

        if user_question:
            docs = vectorstore.similarity_search(user_question, k=5)  # 🔹 daha fazla chunk getir
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )

            # ✅ Daha güçlü chain (map_reduce veya refine da seçilebilir)
            chain = load_qa_chain(llm, chain_type="map_reduce")
            answer = chain.run(input_documents=docs, question=user_question)

            st.write("💡 Cevap:")
            st.write(answer)

if __name__ == "__main__":
    main()
