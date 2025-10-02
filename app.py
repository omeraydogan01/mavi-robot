import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from datetime import datetime
from io import BytesIO

LOG_FILE = "logs.xlsx"

# Secrets şifreleri
REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "1234")
RESET_PASSWORD = st.secrets.get("RESET_PASSWORD", "1234")  # Sıfırlama şifresi

# Log kaydetme
def log_question(question, answer, score):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "confidence_score": score
    }])
    if os.path.exists(LOG_FILE):
        df_old = pd.read_excel(LOG_FILE)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_excel(LOG_FILE, index=False)

# Rapor indirilebilir Excel dosyası
def get_report():
    if os.path.exists(LOG_FILE):
        df = pd.read_excel(LOG_FILE)
        output = BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return output
    return None

# Sıfırlama
def reset_logs():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        st.success("✅ Soru-cevap geçmişi sıfırlandı!")

# Güven skoru hesaplama
def calculate_confidence(answer, similarity_scores):
    """
    Basit kombinasyon:
    - similarity_scores: doküman chunk'larıyla soru arasındaki benzerliklerin ortalaması
    - answer: LLM'den self-evaluation prompt ile dönen skor
    """
    if not similarity_scores:
        sim_score = 0
    else:
        sim_score = sum(similarity_scores)/len(similarity_scores)  # 0-1 arası

    # LLM self-evaluation (saf örnek: answer içerisindeki "confidence" ifadesi)
    # Burada basitçe sim_score ile %40 LLM etkisi kombine ediliyor
    llm_score = 1  # şimdilik varsayılan 1, istersen self-evaluation prompt ekleyebilirsin
    combined_score = 0.6*sim_score + 0.4*llm_score
    return round(combined_score*100, 2)  # % olarak

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")

    # Header ve logo
    col1, col2 = st.columns([1,6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana Soru Sor")

    # API key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ API key bulunamadı. Lütfen secrets veya environment değişkeni ekleyin.")
        st.stop()

    # Dosya yükleme
    uploaded_file = st.file_uploader("📂 Doküman yükleyin", type=["pdf","docx"])
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = "".join([page.extract_text() or "" for page in pdf_reader.pages])
        elif ext == "docx":
            doc = Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            st.error("Sadece PDF veya Word yükleyebilirsiniz.")
            st.stop()

        st.info(f"📄 Doküman {len(text.splitlines())} satır içeriyor.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)
        vectorstore = create_vectorstore(chunks, embeddings)

        # Kullanıcı sorusu
        user_question = st.text_input("Sorunuzu yazın 👇")
        if user_question:
            docs = vectorstore.similarity_search_with_score(user_question, k=6)
            # docs: [(Document, similarity_score)]
            chunks_docs = [d[0] for d in docs]
            similarity_scores = [d[1] for d in docs]

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer = chain.run(input_documents=chunks_docs, question=user_question)

            # Güven skoru hesapla
            confidence = calculate_confidence(answer, similarity_scores)

            st.subheader("💡 Cevap")
            st.success(answer)
            st.info(f"🔹 Güven Skoru: %{confidence}")

            # Log kaydı
            log_question(user_question, answer, confidence)

    # Sidebar: Rapor ve Sıfırlama
    with st.sidebar.expander("📊 Rapor & Yönetim", expanded=False):
        st.subheader("📥 Rapor İndir")
        report_pass = st.text_input("Rapor şifresi", type="password")
        if st.button("📄 Excel İndir"):
            if report_pass == REPORT_PASSWORD:
                report_file = get_report()
                if report_file:
                    st.download_button(
                        label="Excel olarak indir",
                        data=report_file,
                        file_name="soru_cevap_raporu.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.info("Henüz kaydedilmiş soru yok.")
            else:
                st.error("❌ Yanlış şifre!")

        st.subheader("⚠️ Soru Geçmişini Sıfırla")
        reset_pass = st.text_input("Sıfırlama şifresi", type="password")
        if st.button("🗑️ Sıfırla Geçmiş"):
            if reset_pass == RESET_PASSWORD:
                reset_logs()
            else:
                st.error("❌ Yanlış şifre!")

if __name__ == "__main__":
    main()
