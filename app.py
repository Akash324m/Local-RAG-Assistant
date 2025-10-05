import streamlit as st
import time
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import OllamaEmbeddings
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os

# Streamlit page config
st.set_page_config(
    page_title="Local RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🧠 Local RAG Assistant</h1>', unsafe_allow_html=True)
st.write("Ask questions about your documents! Powered by Ollama and LangChain.")

# Initialize session state
if 'rag_chain' not in st.session_state:
    with st.spinner("Loading AI engine... This might take a minute"):
        # Initialize RAG system
        PERSIST_DIRECTORY = PATH  # Directory to save the vector store
        EMBEDDING_MODEL = "all-minilm"  # or "all-minilm" for faster results
        LLM_MODEL = "phi3:mini"  # or "mistral" for better quality
        
        if not os.path.exists(PERSIST_DIRECTORY):
            st.info("Building knowledge base for the first time...")
            loader = DirectoryLoader(PATH, glob="*.txt")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            all_splits = text_splitter.split_documents(documents)
            
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma.from_documents(
                documents=all_splits, 
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            vectorstore.persist()
        else:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
        
        # Setup RAG chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        llm = OllamaLLM(model=LLM_MODEL, temperature=0)
        
        template = """You are a helpful assistant. Use the context to answer the question.
        
        Context: {context}
        
        Question: {question}
        
        Answer clearly and concisely:"""
        custom_rag_prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        st.session_state.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
        )
    st.success("System ready!")

# Chat interface
question = st.text_input("💬 Ask a question about your documents:", placeholder="e.g., What is artificial intelligence?")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        start_time = time.time()
        answer = st.session_state.rag_chain.invoke(question)
        response_time = time.time() - start_time
        
        # Display results
        st.markdown(f"**Answer:**")
        st.info(answer)
        
        # Show performance metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Response Time", f"{response_time:.2f}s")
        with col2:
            st.metric("Model", "Phi-3 Mini")
        
        # Show source documents
        with st.expander("View source context"):
            embeddings = OllamaEmbeddings(model="all-minilm")
            vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings
            )
            relevant_docs = vectorstore.similarity_search(question, k=2)
            for i, doc in enumerate(relevant_docs):
                st.write(f"**Source {i+1}:**")
                st.write(doc.page_content)
                st.write("---")

# Sidebar with project info
with st.sidebar:
    st.header("📊 Project Info")
    st.write("""
    **Local RAG System**
    
    - Retrieval-Augmented Generation
    - Local LLM with Ollama
    - Chroma vector database
    - Private & offline capable
    """)
    
    st.header("⚙️ Technical Stack")
    st.write("""
    - **Framework:** LangChain
    - **LLM:** Ollama (Phi-3 Mini)
    - **Embeddings:** All-MiniLM
    - **Vector DB:** Chroma
    - **UI:** Streamlit
    """)
    
    if st.button("Clear Chat"):
        st.session_state.clear()
        st.rerun()