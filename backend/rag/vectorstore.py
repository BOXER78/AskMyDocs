import shutil
import tempfile
import traceback

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./.model_cache"

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

vector_db = None
qa_chain = None
chat_history = []
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_and_store_document(file_path: str):
    global vector_db, qa_chain, chat_history, embeddings
    chat_history = []

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = splitter.split_documents(documents)

    if vector_db is None:
        vector_db = FAISS.from_documents(docs, embeddings)
    else:
        vector_db.add_documents(docs)

    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    
    template = """You are a detailed document analysis tool. Provide accurate information based strictly on the text below and the conversation history.
 
 ### YOUR MEMORY & CONTEXT:
 - Use the provided conversation history to understand follow-up questions.
 
 ### DOCUMENT ANALYSIS RULES (CRITICAL):
 - Search the context carefully. If the information is present in different parts, combine it for a complete answer.
 - If the question is about the document but the information is truly missing, state: "Information not found in document."
 - Cite the context clearly where possible.
 - Avoid speculation.
 
 Conversation History:
 {chat_history}
 
 Context from Document:
 {context}
 
 Question: {question}
 
 Response:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(history):
        return "\n".join([f"{m['role']}: {m['content']}" for m in history])

    qa_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "chat_history": lambda _: format_history(chat_history)
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

def process_and_store_repo(repo_path: str):
    global vector_db, qa_chain, chat_history, embeddings
    chat_history = []
    
    is_github = repo_path.startswith("http") and "github.com" in repo_path
    temp_dir = None
    
    if is_github:
        print(f"Cloning GitHub repository: {repo_path}")
        temp_dir = tempfile.mkdtemp(prefix="rag_repo_")
        try:
            subprocess.run(["git", "clone", "--depth", "1", repo_path, temp_dir], check=True, capture_output=True)
            effective_path = temp_dir
        except subprocess.CalledProcessError as sc:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            error_details = sc.stderr.decode() if sc.stderr else str(sc)
            raise ValueError(f"Git Clone Failed: {error_details}")
        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise ValueError(f"Failed to clone repository: {str(e)}")
    else:
        if not os.path.exists(repo_path):
            raise ValueError(f"Local path does not exist: {repo_path}")
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
            show_progress=True,
            exclude=["**/node_modules/**", "**/venv/**", "**/.git/**", "**/dist/**", "**/build/**", "**/__pycache__/**"]
        )
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {ext} files: {e}")

    if not documents:
        if is_github and temp_dir:
            shutil.rmtree(temp_dir)
        raise ValueError("No supported files found in the directory.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = splitter.split_documents(documents)

    if vector_db is None:
        vector_db = FAISS.from_documents(docs, embeddings)
    else:
        vector_db.add_documents(docs)

    # Simplified re-retrieval setup
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 10})
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    
    template = """You are an expert code analyst and repository explorer. Analyze the Codebase snippets provided to answer the query.
    
### ANALYSIS RULES:
- Focus on the structure, logic, and patterns in the code.
- If the question is about how something works, find the relevant functions.
- If the information is missing, state: "Code pattern not found in repository."
- Cite the file names when explaining.

Conversation History:
{chat_history}

Codebase Context:
{context}

Question: {question}

Response:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([f"--- FILE: {d.metadata.get('source', 'Unknown')} ---\n{d.page_content}" for d in docs])

    def format_history(history):
        return "\n".join([f"{m['role']}: {m['content']}" for m in history])

    qa_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough(),
            "chat_history": lambda _: format_history(chat_history)
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Cleanup temp directory if it was a github clone
    if is_github and temp_dir and os.path.exists(temp_dir):
        print(f"Cleaning up temp repo: {temp_dir}")
        # Note: In a production app, you might want to keep the indexed vectors 
        # but cleanup the source files after indexing.
        shutil.rmtree(temp_dir)


def ask_question(query: str):
    global qa_chain, chat_history
    if not qa_chain:
        return "Please upload a document first. The chatbot needs context."
    
    answer = qa_chain.invoke(query)
    
    chat_history.append({"role": "User", "content": query})
    chat_history.append({"role": "Assistant", "content": answer})
    
    if len(chat_history) > 20:
        chat_history = chat_history[-20:]
        
    return answer



