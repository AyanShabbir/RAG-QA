# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv
# import pdfplumber

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Embedding model instance (reuse everywhere)
# embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# # Chat model instance (reuse everywhere)
# chat_model_name = "models/gemini-2.5-pro"  # Use a model you have access to
# chat_model = ChatGoogleGenerativeAI(model=chat_model_name, temperature=0.3)

# def get_pdf_text_and_pages(pdf_docs):
#     all_text = ""
#     page_texts = []

#     for pdf in pdf_docs:
#         with pdfplumber.open(pdf) as pdf_file:
#             for i, page in enumerate(pdf_file.pages):
#                 text = page.extract_text()
#                 if text:
#                     all_text += text
#                     page_texts.append({
#                         "page_num": i + 1,
#                         "text": text
#                     })
#     return all_text, page_texts

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(page_texts):
#     texts = [p["text"] for p in page_texts]
#     metadatas = [{"page": p["page_num"]} for p in page_texts]
    
#     # Use embedding_model here, NOT chat_model!
#     vector_store = FAISS.from_texts(texts, embedding=embedding_model, metadatas=metadatas)
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
#     the provided context, just say, "The answer is not available in the context," and don't provide a wrong answer.\n\n
#     Context:\n{context}\n
#     Question:\n{question}\n
#     Answer:
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
#     # Use chat_model instance created globally
#     chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     try:
#         new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
#         docs = new_db.similarity_search(user_question, k=2)  # top 2 matches

#         chain = get_conversational_chain()
#         response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
#         st.subheader("💬 Response:")
#         st.write(response["output_text"])

#         st.subheader("📄 Most Relevant Page(s):")
#         for doc in docs:
#             st.markdown(f"**Page {doc.metadata.get('page', 'N/A')}:**")
#             st.code(doc.page_content[:1000])  # show relevant chunk
#     except Exception as e:
#         st.error(f"Error: {str(e)}")

# def main():
#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="💬")
#     st.header("RAG-based Q&A system")
#     st.write("Upload PDF files and ask questions to get detailed answers based on the content of the PDFs.")
#     st.write("--------------------------------------------------------------------------------------------")
#     st.write("1) Click on Browse files and select your PDF file")
#     st.write("2) Click on Submit & Process")
#     st.write("3) Ask all the questions you want")
#     st.write("")

#     user_question = st.text_input("Ask a Question from the PDF Files")
#     if user_question:
#         user_input(user_question)

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
#         if st.button("Submit & Process"):
#             if pdf_docs:
#                 with st.spinner("Processing..."):
#                     try:
#                         raw_text, page_texts = get_pdf_text_and_pages(pdf_docs)
#                         get_vector_store(page_texts)

#                         st.success("Processing complete. You can now ask questions from the PDFs.")
#                     except Exception as e:
#                         st.error(f"Error: {str(e)}")
#             else:
#                 st.error("Please upload at least one PDF file.")

# if __name__ == "__main__":
#     main()


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
import time
import hashlib
import pickle

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state for caching
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

@st.cache_resource
def get_embedding_model():
    """Cache the embedding model to avoid reinitialization"""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

@st.cache_resource
def get_chat_model():
    """Cache the chat model to avoid reinitialization"""
    return ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)  # Use faster model

def get_file_hash(uploaded_file):
    """Generate hash for uploaded file to check if it's already processed"""
    uploaded_file.seek(0)
    file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return file_hash

def get_pdf_text_optimized(pdf_docs):
    """Optimized PDF text extraction with progress tracking"""
    all_texts = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf in enumerate(pdf_docs):
        file_hash = get_file_hash(pdf)
        
        # Skip if already processed
        if file_hash in st.session_state.processed_files:
            status_text.text(f"File {pdf.name} already processed, skipping...")
            continue
            
        status_text.text(f"Processing {pdf.name}...")
        
        try:
            # Use PyPDF2 instead of pdfplumber for faster processing
            pdf_reader = PdfReader(pdf)
            text = ""
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}"
            
            if text.strip():
                all_texts.append(text)
                st.session_state.processed_files.add(file_hash)
                
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            continue
            
        progress_bar.progress((idx + 1) / len(pdf_docs))
    
    progress_bar.empty()
    status_text.empty()
    return "\n".join(all_texts)

@st.cache_data
def get_text_chunks_cached(text):
    """Cache text chunks to avoid reprocessing"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Reduced chunk size for faster processing
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store_optimized(text_chunks):
    """Create vector store with progress tracking"""
    if not text_chunks:
        st.error("No text chunks to process")
        return None
        
    embedding_model = get_embedding_model()
    
    # Process in smaller batches to avoid timeout
    batch_size = 50
    all_vectors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        status_text.text(f"Creating embeddings for batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}...")
        
        try:
            if i == 0:
                # Create initial vector store
                vector_store = FAISS.from_texts(batch, embedding_model)
            else:
                # Add to existing vector store
                batch_vectors = FAISS.from_texts(batch, embedding_model)
                vector_store.merge_from(batch_vectors)
                
        except Exception as e:
            st.error(f"Error creating embeddings for batch: {str(e)}")
            return None
            
        progress_bar.progress((i + batch_size) / len(text_chunks))
        time.sleep(0.1)  # Small delay to prevent rate limiting
    
    progress_bar.empty()
    status_text.empty()
    
    return vector_store

@st.cache_resource
def get_conversational_chain():
    """Cache the conversational chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say "The answer is not available in the context."
    
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chat_model = get_chat_model()
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handle user questions with error handling"""
    if st.session_state.vector_store is None:
        st.error("Please upload and process PDF files first.")
        return
    
    try:
        with st.spinner("Searching for relevant information..."):
            # Search for relevant documents
            docs = st.session_state.vector_store.similarity_search(user_question, k=3)
            
            if not docs:
                st.warning("No relevant information found in the uploaded documents.")
                return
            
            # Get conversational chain and generate response
            chain = get_conversational_chain()
            
            with st.spinner("Generating response..."):
                response = chain(
                    {"input_documents": docs, "question": user_question}, 
                    return_only_outputs=True
                )
            
            # Display response
            st.subheader("💬 Response:")
            st.write(response["output_text"])
            
            # Display sources
            with st.expander("📄 View Sources"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")
                    
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config(
        page_title="PDF Chat Assistant", 
        page_icon="💬",
        layout="wide"
    )
    
    st.title("💬 PDF Chat Assistant")
    st.markdown("Upload PDF files and ask questions to get answers based on their content.")
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask Questions")
        user_question = st.text_input("Enter your question:", placeholder="What would you like to know?")
        
        if user_question:
            user_input(user_question)
    
    with col2:
        st.subheader("Upload Documents")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Choose PDF files", 
            accept_multiple_files=True, 
            type=["pdf"],
            help="Upload one or more PDF files to chat with"
        )
        
        # Process button
        if st.button("🚀 Process Documents", type="primary"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    try:
                        # Extract text
                        st.info("Extracting text from PDFs...")
                        raw_text = get_pdf_text_optimized(pdf_docs)
                        
                        if not raw_text.strip():
                            st.error("No text could be extracted from the uploaded files.")
                            return
                        
                        # Create chunks
                        st.info("Creating text chunks...")
                        text_chunks = get_text_chunks_cached(raw_text)
                        
                        # Create vector store
                        st.info("Creating vector database...")
                        vector_store = create_vector_store_optimized(text_chunks)
                        
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success(f"✅ Successfully processed {len(pdf_docs)} PDF file(s)!")
                            st.info("You can now ask questions about your documents.")
                        else:
                            st.error("Failed to create vector database.")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Display processing status
        if st.session_state.vector_store is not None:
            st.success("✅ Documents ready for questions!")
        else:
            st.info("ℹ️ Upload and process documents to start chatting")

if __name__ == "__main__":
    main()
