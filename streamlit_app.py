import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Page configuration
st.set_page_config(page_title="AskMyDocs", page_icon="🦀", layout="wide")

# Load environment variables
load_dotenv()

# Initialize session state for chat and vector store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

# Sidebar for file upload
with st.sidebar:
    st.title("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file and uploaded_file.name != st.session_state.processed_file:
        with st.spinner("Analyzing document..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
                docs = splitter.split_documents(documents)
                
                # Create Vector Store
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
                st.session_state.processed_file = uploaded_file.name
                st.success(f"Successfully processed: {uploaded_file.name}")
                st.session_state.messages = [] # Clear chat for new doc
            except Exception as e:
                st.error(f"Error processing file: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    
    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🦀 AskMyDocs - AI Assistant")
st.caption("Ask questions about your uploaded PDF documents.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about the document?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    if not st.session_state.vector_db:
        with st.chat_message("assistant"):
            st.warning("Please upload a PDF document in the sidebar first.")
    else:
        with st.chat_message("assistant"):
            try:
                # Setup components
                retriever = st.session_state.vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})
                llm = ChatGroq(
                    temperature=0, 
                    model_name="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY")
                )
                
                template = """You are a detailed document analysis tool. Provide accurate information based strictly on the text below.
                
                Context:
                {context}
                
                Question: {question}
                
                Response:"""
                custom_rag_prompt = PromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | custom_rag_prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Stream the response (simulated for better UI experience)
                response = chain.invoke(prompt)
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                st.error(f"AI Error: {e}")
