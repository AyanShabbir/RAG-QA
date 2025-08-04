import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pdfplumber

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Embedding model instance (reuse everywhere)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Chat model instance (reuse everywhere)
chat_model_name = "models/gemini-2.5-pro"  # Use a model you have access to
chat_model = ChatGoogleGenerativeAI(model=chat_model_name, temperature=0.3)

def get_pdf_text_and_pages(pdf_docs):
    all_text = ""
    page_texts = []

    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_file:
            for i, page in enumerate(pdf_file.pages):
                text = page.extract_text()
                if text:
                    all_text += text
                    page_texts.append({
                        "page_num": i + 1,
                        "text": text
                    })
    return all_text, page_texts

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(page_texts):
    texts = [p["text"] for p in page_texts]
    metadatas = [{"page": p["page_num"]} for p in page_texts]
    
    # Use embedding_model here, NOT chat_model!
    vector_store = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context," and don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use chat_model instance created globally
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question, k=2)  # top 2 matches

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        st.subheader("💬 Response:")
        st.write(response["output_text"])

        st.subheader("📄 Most Relevant Page(s):")
        for doc in docs:
            st.markdown(f"**Page {doc.metadata.get('page', 'N/A')}:**")
            st.code(doc.page_content[:1000])  # show relevant chunk
    except Exception as e:
        st.error(f"Error: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="💬")
    st.header("RAG-based Q&A system")
    st.write("Upload PDF files and ask questions to get detailed answers based on the content of the PDFs.")
    st.write("--------------------------------------------------------------------------------------------")
    st.write("1) Click on Browse files and select your PDF file")
    st.write("2) Click on Submit & Process")
    st.write("3) Ask all the questions you want")
    st.write("")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        raw_text, page_texts = get_pdf_text_and_pages(pdf_docs)
                        get_vector_store(page_texts)

                        st.success("Processing complete. You can now ask questions from the PDFs.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
