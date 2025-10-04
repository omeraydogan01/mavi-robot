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
from PIL import Image
import openai

LOG_FILE = "logs.xlsx"

# Secrets şifreleri
REPORT_PASSWORD = st.secrets.get("REPORT_PASSWORD", "1234")
RESET_PASSWORD = st.secrets.get("RESET_PASSWORD", "1234")  

# Log kaydetme
def log_question(question, answer):
    df_new = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer
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

# Streamlit session_state ile input temizleme
if "user_question" not in st.session_state:
    st.session_state.user_question = ""

def main():
    st.set_page_config(page_title="Mavi Soru Robotu", page_icon="logo.png")
    
    # Header ve logo
    col1, col2 = st.columns([1,6])
    with col1:
        st.image("logo.png", width=120)
    with col2:
        st.header("Dokümana & Görsele Soru Sor")

    # API key
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    openai.api_key = api_key
    if not api_key:
        st.error("⚠️ API key bulunamadı. Lütfen secrets veya environment değişkeni ekleyin.")
        st.stop()

    # Dosya yükleme
    uploaded_files = st.file_uploader(
        "📂 Bir veya birden fazla doküman yükleyin",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    all_texts = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            ext = uploaded_file.name.split(".")[-1].lower()
            file_text = ""
            if ext == "pdf":
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    content = page.extract_text() or ""
                    file_text += content
            elif ext == "docx":
                doc = Document(uploaded_file)
                file_text = "\n".join([p.text for p in doc.paragraphs])
            all_texts.append(file_text)

    full_text = "\n".join(all_texts) if all_texts else ""
    if full_text:
        st.info(f"📚 {len(uploaded_files)} doküman yüklendi. Toplam {len(full_text.split())} kelime işlendi.")

        # Metin parçalama
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = text_splitter.split_text(full_text)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)

        @st.cache_resource
        def create_vectorstore(chunks, embeddings):
            return FAISS.from_texts(chunks, embeddings)
        vectorstore = create_vectorstore(chunks, embeddings)

    # Görsel yükleme
    uploaded_image = st.file_uploader("🖼️ Görsel yükleyin (opsiyonel)", type=["png", "jpg", "jpeg"])

    # Kullanıcı sorusu
    user_question = st.text_input("Sorunuzu yazın 👇", value=st.session_state.user_question)

    if st.button("Sor"):
        st.session_state.user_question = ""  # input temizle

        combined_answer = ""

        # Doküman tabanlı cevap
        if full_text:
            docs = vectorstore.similarity_search(user_question, k=6)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
            chain = load_qa_chain(llm, chain_type="stuff")
            answer_docs = chain.run(input_documents=docs, question=user_question)
            combined_answer += f"📄 Dokümandan Cevap:\n{answer_docs}\n\n"

        # Görsel tabanlı cevap
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            try:
                response = openai.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role":"user", "content": f"Bu görselle ilgili soruyu cevapla: {user_question}"}],
                    files=[("image.png", buf)]
                )
                answer_img = response.choices[0].message["content"]
                combined_answer += f"🖼️ Görselden Cevap:\n{answer_img}\n\n"
            except Exception as e:
                combined_answer += f"🖼️ Görsel analizi sırasında hata oluştu: {str(e)}\n\n"

        if combined_answer:
            st.subheader("💡 Cevap")
            st.success(combined_answer)
            log_question(user_question, combined_answer)
        else:
            st.info("Lütfen bir soru girin veya doküman/görsel yükleyin.")

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
