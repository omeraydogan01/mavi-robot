import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Mavi Robot", page_icon="🤖", layout="wide")

LOG_FILE = "logs.xlsx"

# 🧠 Embedding ve LLM ayarları
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini")

st.title("🤖 Mavi Robot - Doküman Sorgulama Asistanı")

uploaded_files = st.file_uploader(
    "PDF veya Word dosyalarınızı yükleyin",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            for page in reader.pages:
                all_text += page.extract_text()
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            for paragraph in doc.paragraphs:
                all_text += paragraph.text + "\n"

    if all_text.strip():
        st.success("✅ Belgeler başarıyla yüklendi!")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(all_text)

        vectorstore = FAISS.from_texts(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        qa_chain = create_stuff_documents_chain(llm, "Aşağıdaki belgelere göre yanıt ver:")
        retrieval_chain = create_retrieval_chain(retriever, qa_chain)

        question = st.text_input("Sorunuzu yazın:")

        if question:
            with st.spinner("Yanıt hazırlanıyor..."):
                response = retrieval_chain.invoke({"input": question})
                answer = response["answer"]
                st.write("💬 **Yanıt:**", answer)

                # Log kaydı
                data = {
                    "Tarih": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    "Soru": [question],
                    "Cevap": [answer]
                }

                df_new = pd.DataFrame(data)
                try:
                    df_old = pd.read_excel(LOG_FILE)
                    df = pd.concat([df_old, df_new], ignore_index=True)
                except FileNotFoundError:
                    df = df_new

                df.to_excel(LOG_FILE, index=False)

                st.success("🔹 Soru-cevap kaydedildi!")
