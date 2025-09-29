import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# .env dosyasını yükle
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")
    
    # Header ve logo yan yana
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=40)  # logo.png dosyasının yolu ve boyutu
    with col2:
        st.header("Dokümana Soru Sor")

    # PDF yükleme
    pdf = st.file_uploader("PDF yükle", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Metin parçalara bölünüyor
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Embeddings oluşturuluyor
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Kullanıcıdan soru al
        user_question = st.text_input("Doküman hakkında sorunu yaz 👇")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # OpenAI LLM
            llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
            chain = load_qa_chain(llm, chain_type="stuff")

            # Cevap üret
            response = chain.run(input_documents=docs, question=user_question)

            st.write("### 📌 Cevap:")
            st.write(response)

if __name__ == "__main__":
    main()
