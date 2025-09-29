import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Lokal için .env dosyası
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit Cloud secrets
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="logo.png")
    st.header("📄 PDF'inle Sohbet Et")

    # PDF yükleme
    pdf = st.file_uploader("PDF yükle", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        # Metni parçalara böl
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # OpenAI Embeddings
        embeddings = OpenAIEmbeddings(
            model_name="text-embedding-3-small",  # ⚠️ model_name kullan
            openai_api_key=api_key               # ⚠️ openai_api_key kullan
        )
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("PDF hakkında sorunu yaz 👇")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # Chat LLM
            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key
            )
            chain = load_qa_chain(llm, chain_type="stuff")

            # Cevap üret
            response = chain.run(input_documents=docs, question=user_question)

            st.write("### 📌 Cevap:")
            st.write(response)

if __name__ == "__main__":
    main()
