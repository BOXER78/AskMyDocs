import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
import subprocess
import shutil
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
load_dotenv("backend/.env")

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
    st.title("📂 Repository Analysis")
    repo_input = st.text_input("Enter Local Path or GitHub URL", placeholder="e.g. https://github.com/user/repo")
    if st.button("Analyze Repo", use_container_width=True):
        if repo_input.strip():
            with st.spinner("Indexing repository..."):
                repo_path = repo_input.strip()
                is_github = repo_path.startswith("http") and "github.com" in repo_path
                temp_dir = None
                
                try:
                    if is_github:
                        temp_dir = tempfile.mkdtemp(prefix="st_repo_")
                        subprocess.run(["git", "clone", "--depth", "1", repo_path, temp_dir], check=True, capture_output=True)
                        effective_path = temp_dir
                    else:
                        if not os.path.exists(repo_path):
                            st.error(f"Local path does not exist: {repo_path}")
                            st.stop()
                        effective_path = repo_path

                    # Index common code and text files
                    supported_extensions = [".py", ".js", ".jsx", ".tsx", ".ts", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml"]
                    
                    documents = []
                    for ext in supported_extensions:
                        loader = DirectoryLoader(
                            effective_path,
                            glob=f"**/*{ext}",
                            loader_cls=TextLoader,
                            use_multithreading=True,
                            exclude=["**/node_modules/**", "**/venv/**", "**/.git/**", "**/dist/**", "**/build/**", "**/__pycache__/**"]
                        )
                        try:
                            documents.extend(loader.load())
                        except:
                            pass

                    if not documents:
                        st.error("No supported files found in the directory.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
                        docs = splitter.split_documents(documents)
                        
                        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                        st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
                        st.session_state.processed_file = repo_path.split('/')[-1] if repo_path.strip('/') else "Repo"
                        st.success(f"Successfully indexed: {st.session_state.processed_file}")
                        st.session_state.messages = [] # Clear chat for new context

                except Exception as e:
                    st.error(f"Error indexing repo: {e}")
                finally:
                    if is_github and temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
        else:
            st.warning("Please enter a path or URL.")

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
                
                template = """You are a detailed document analysis tool. Provide accurate information based strictly on the text below and the conversation history.
                
                ### YOUR MEMORY & CONTEXT:
                - Use the provided conversation history to understand follow-up questions.
                
                ### ANALYSIS RULES:
                - If the question is about the document but the information is truly missing, state: "Information not found."
                - Avoid speculation.
                
                Conversation History:
                {chat_history}
                
                Context:
                {context}
                
                Question: {question}
                
                Response:"""
                custom_rag_prompt = PromptTemplate.from_template(template)

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                def format_history(messages):
                    return "\n".join([f"{m['role']}: {m['content']}" for m in messages[-5:]])

                chain = (
                    {
                        "context": retriever | format_docs, 
                        "question": RunnablePassthrough(),
                        "chat_history": lambda _: format_history(st.session_state.messages)
                    }
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
