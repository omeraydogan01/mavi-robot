import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# .env dosyasını yükle (lokal çalışmada)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
    st.header("📄 PDF'inle Sohbet Et")

    # PDF yükleme
    pdf = st.file_uploader("PDF yükle", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()

        # Metin parçalama
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("PDF hakkında sorunu yaz 👇")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # LLM
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                openai_api_key=api_key
            )
            chain = load_qa_chain(llm, chain_type="stuff")

            # Cevap üret (invoke ile)
            response = chain.invoke(
                {"input_documents": docs, "question": user_question}
            )

            st.write("### 📌 Cevap:")
            st.write(response["output_text"])

if __name__ == "__main__":
    main()
