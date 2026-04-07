import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

vector_db = None
qa_chain = None
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def process_and_store_document(file_path: str):
    global vector_db, qa_chain, embeddings

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
    
    template = """You are a detailed document analysis tool. Provide accurate information based strictly on the text below.
 
 ### DOCUMENT ANALYSIS RULES (CRITICAL):
 - Search the context carefully. If the information is present in different parts, combine it for a complete answer.
 - If the question is about the document but the information is truly missing, state: "Information not found in document."
 - Cite the context clearly where possible.
 - Avoid speculation.
 
 Context from Document:
 {context}
 
 Question: {question}
 
 Response:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": retriever | format_docs, 
            "question": RunnablePassthrough()
        }
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

def ask_question(query: str):
    global qa_chain
    if not qa_chain:
        return "Please upload a document first. The chatbot needs context."
    
    answer = qa_chain.invoke(query)
        
    return answer
