"""
Streamlit Web Interface for RAG System
A clean web UI for the RAG system with document upload and chat interface
"""

import streamlit as st
import os
from rag_system import RAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_rag_system():
    """Initialize the RAG system with session state management"""
    if 'rag_system' not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("Please set OPENAI_API_KEY in your environment or .env file")
            st.stop()
        
        st.session_state.rag_system = RAGSystem(api_key)
        st.session_state.documents_loaded = False
        st.session_state.chat_history = []

def load_sample_documents():
    """Load sample documents if they don't exist"""
    os.makedirs("sample_docs", exist_ok=True)
    
    if not os.path.exists("sample_docs/ai_overview.txt"):
        with open("sample_docs/ai_overview.txt", "w") as f:
            f.write("""
Artificial Intelligence Overview

Artificial Intelligence (AI) is a branch of computer science that aims to create 
machines capable of intelligent behavior. AI systems can perform tasks that typically 
require human intelligence, such as visual perception, speech recognition, 
decision-making, and language translation.

Key areas of AI include:
- Machine Learning: Algorithms that improve through experience
- Natural Language Processing: Understanding and generating human language
- Computer Vision: Interpreting visual information
- Robotics: Physical AI systems that interact with the world

AI has applications in healthcare, finance, transportation, entertainment, and many 
other fields. Recent advances in deep learning and neural networks have led to 
breakthroughs in areas like image recognition, language understanding, and game playing.

The future of AI holds promise for solving complex problems and augmenting human 
capabilities, though it also raises important questions about ethics, safety, and 
the future of work.
            """)
    
    if not os.path.exists("sample_docs/machine_learning_basics.txt"):
        with open("sample_docs/machine_learning_basics.txt", "w") as f:
            f.write("""
Machine Learning Fundamentals

Machine Learning (ML) is a subset of artificial intelligence that focuses on 
algorithms and statistical models that enable computer systems to improve their 
performance on a specific task through experience.

Types of Machine Learning:

1. Supervised Learning: Learning with labeled training data
   - Classification: Predicting categories or classes
   - Regression: Predicting continuous values
   - Examples: Email spam detection, house price prediction

2. Unsupervised Learning: Finding patterns in data without labels
   - Clustering: Grouping similar data points
   - Dimensionality Reduction: Reducing data complexity
   - Examples: Customer segmentation, anomaly detection

3. Reinforcement Learning: Learning through interaction and feedback
   - Agent learns by taking actions and receiving rewards
   - Examples: Game playing, autonomous vehicles

Key concepts include:
- Training data: Examples used to teach the model
- Features: Input variables used for prediction
- Model: The algorithm that makes predictions
- Overfitting: When a model performs well on training data but poorly on new data
- Cross-validation: Testing model performance on unseen data

Popular ML algorithms include linear regression, decision trees, random forests, 
support vector machines, and neural networks. The choice of algorithm depends on 
the problem type, data size, and desired accuracy.
            """)

def main():
    st.set_page_config(
        page_title="RAG System Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG System Demo")
    st.markdown("**Retrieval-Augmented Generation** with LangChain and OpenAI")
    
    # Initialize RAG system
    initialize_rag_system()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Load sample documents
        if st.button("Load Sample Documents", type="primary"):
            with st.spinner("Loading sample documents..."):
                load_sample_documents()
                sample_docs = [
                    "sample_docs/ai_overview.txt",
                    "sample_docs/machine_learning_basics.txt"
                ]
                
                # Load and process documents
                documents = st.session_state.rag_system.load_documents(sample_docs)
                chunks = st.session_state.rag_system.process_documents(documents)
                st.session_state.rag_system.create_vectorstore(chunks, recreate=True)
                st.session_state.rag_system.setup_qa_chain()
                st.session_state.documents_loaded = True
                
                st.success("Sample documents loaded successfully!")
        
        # File upload
        st.subheader("Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files to add to the knowledge base"
        )
        
        if uploaded_files and st.button("Process Uploaded Files"):
            with st.spinner("Processing uploaded files..."):
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)
                
                # Load and process documents
                documents = st.session_state.rag_system.load_documents(temp_paths)
                chunks = st.session_state.rag_system.process_documents(documents)
                st.session_state.rag_system.create_vectorstore(chunks, recreate=True)
                st.session_state.rag_system.setup_qa_chain()
                st.session_state.documents_loaded = True
                
                # Clean up temp files
                for temp_path in temp_paths:
                    os.remove(temp_path)
                
                st.success(f"Processed {len(uploaded_files)} files successfully!")
        
        # Document status
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded and ready")
        else:
            st.warning("⚠️ No documents loaded yet")
    
    # Main chat interface
    if not st.session_state.documents_loaded:
        st.info("👆 Please load some documents first using the sidebar")
        return
    
    # Chat interface
    st.header("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(source.page_content[:200] + "...")
                        st.write(f"*File: {source.metadata.get('source', 'Unknown')}*")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_system.ask_question(prompt)
                    
                    # Display answer
                    st.write(result["answer"])
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                    # Display sources
                    with st.expander("📖 Sources"):
                        for i, source in enumerate(result["sources"], 1):
                            st.write(f"**Source {i}:**")
                            st.write(source.page_content[:200] + "...")
                            st.write(f"*File: {source.metadata.get('source', 'Unknown')}*")
                            st.divider()
                            
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
