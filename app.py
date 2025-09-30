import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")
    
    # CSS - Browse files butonunu ve soru alanını özelleştir
    st.markdown(
        """
        <style>
        /* PDF yükle butonu */
        button[data-testid="stFileUploaderBrowseButton"] {
            background-color: #2e86de;
            color: white;
            border-radius: 8px;
        }
        button[data-testid="stFileUploaderBrowseButton"]::after {
            content: " Dosya Seç";
        }
        button[data-testid="stFileUploaderBrowseButton"] > div {
            display: none;
        }

        /* Soru alanı mavi çerçeve */
        div[data-testid="stTextArea"] textarea {
            border: 3px solid #1E90FF;
            border-radius: 8px;
            padding: 10px;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

    uploaded_file = st.file_uploader("📂 Doküman yükleyin", type="pdf")
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        
        st.info(f"📄 Yüklenen doküman toplam **{len(pdf_reader.pages)}** sayfa içeriyor.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        # Embedding modeli
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=api_key
        )

        # FAISS vektör veritabanı (cache'li)
        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)
        
        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu (mavi çerçeveli alan)
        user_question = st.text_area("Sorunuzu yazın 👇", height=130)

        if user_question:
            # Daha fazla chunk → daha sağlam cevap
            docs = vectorstore.similarity_search(user_question, k=6)

            # Daha hızlı ve ucuz model
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=api_key
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=docs, question=user_question)

            st.subheader("💡 Cevap")
            st.success(answer)  # yeşil kutuda göster

if __name__ == "__main__":
    main()
