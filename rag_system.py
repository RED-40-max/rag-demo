"""
RAG System Implementation
A complete Retrieval-Augmented Generation system using LangChain
Built for demonstrating RAG capabilities with document processing and question answering
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

class RAGSystem:
    """
    Complete RAG system that processes documents and answers questions
    using retrieval-augmented generation
    """
    
    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize the RAG system with OpenAI API key
        
        Args:
            openai_api_key: OpenAI API key for embeddings and LLM
            persist_directory: Directory to persist vector database
        """
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = OpenAI(openai_api_key=openai_api_key, temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize vector store
        self.vectorstore = None
        self.qa_chain = None
        
    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from various file types
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            print(f"Loading document: {file_path}")
            
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                print(f"Unsupported file type: {file_path}")
                continue
                
            docs = loader.load()
            documents.extend(docs)
            
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for better retrieval
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed document chunks
        """
        print("Processing documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} document chunks")
        return chunks
    
    def create_vectorstore(self, documents: List[Document], recreate: bool = False):
        """
        Create or load vector store from documents
        
        Args:
            documents: List of document chunks
            recreate: Whether to recreate the vector store
        """
        if os.path.exists(self.persist_directory) and not recreate:
            print("Loading existing vector store...")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("Vector store created and saved")
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain with custom prompt
        """
        # Custom prompt template for better responses
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer based on the context, just say that you don't know, 
        don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        print("QA chain setup complete")
    
    def ask_question(self, question: str) -> dict:
        """
        Ask a question and get an answer with sources
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Run setup_qa_chain() first.")
        
        print(f"Question: {question}")
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    
    def get_similar_documents(self, query: str, k: int = 3) -> List[Document]:
        """
        Get similar documents for a query
        
        Args:
            query: Query string
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        return self.vectorstore.similarity_search(query, k=k)


def main():
    """
    Main function demonstrating RAG system usage
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in your environment or .env file")
        return
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(api_key)
    
    # Sample documents (you can replace these with your own)
    sample_docs = [
        "sample_docs/ai_overview.txt",
        "sample_docs/machine_learning_basics.txt"
    ]
    
    # Check if sample docs exist, if not create them
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
    
    # Load and process documents
    print("\nLoading documents...")
    documents = rag.load_documents(sample_docs)
    
    print("Processing documents...")
    chunks = rag.process_documents(documents)
    
    # Create vector store
    print("Creating vector store...")
    rag.create_vectorstore(chunks)
    
    # Setup QA chain
    print("Setting up QA chain...")
    rag.setup_qa_chain()
    
    # Interactive Q&A
    print("\n" + "="*50)
    print("RAG System Ready! Ask questions about AI and Machine Learning")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            result = rag.ask_question(question)
            print(f"\nAnswer: {result['answer']}")
            
            print(f"\nSources ({len(result['sources'])} documents):")
            for i, doc in enumerate(result['sources'], 1):
                print(f"{i}. {doc.page_content[:100]}...")
                print(f"   Source: {doc.metadata.get('source', 'Unknown')}")
                print()
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
