#!/usr/bin/env python3
"""
Quick demo script for the RAG system
Shows off key features with sample questions
"""

import os
from dotenv import load_dotenv
from rag_system import RAGSystem

def run_demo():
    """Run a quick demo of the RAG system"""
    print("🚀 RAG System Demo")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in your .env file")
        print("   Copy env_example.txt to .env and add your key")
        return
    
    # Initialize system
    print("🔧 Initializing RAG system...")
    rag = RAGSystem(api_key)
    
    # Create sample documents
    os.makedirs("sample_docs", exist_ok=True)
    
    sample_content = {
        "ai_overview.txt": """
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
        """,
        "machine_learning_basics.txt": """
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
        """
    }
    
    # Write sample files
    for filename, content in sample_content.items():
        filepath = f"sample_docs/{filename}"
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)
    
    # Load and process documents
    print("📚 Loading sample documents...")
    documents = rag.load_documents(["sample_docs/ai_overview.txt", "sample_docs/machine_learning_basics.txt"])
    
    print("✂️  Processing documents into chunks...")
    chunks = rag.process_documents(documents)
    
    print("🗄️  Creating vector store...")
    rag.create_vectorstore(chunks)
    
    print("🔗 Setting up QA chain...")
    rag.setup_qa_chain()
    
    print("\n✅ RAG system ready!")
    print("\n" + "=" * 50)
    
    # Demo questions
    demo_questions = [
        "What is artificial intelligence?",
        "What are the different types of machine learning?",
        "What is overfitting in machine learning?",
        "What are some applications of AI?",
        "How does reinforcement learning work?"
    ]
    
    print("🎯 Running demo questions...\n")
    
    for i, question in enumerate(demo_questions, 1):
        print(f"Q{i}: {question}")
        print("-" * 40)
        
        try:
            result = rag.ask_question(question)
            print(f"Answer: {result['answer']}")
            print(f"Sources: {len(result['sources'])} documents")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    print("🎉 Demo complete!")
    print("\nTo run the interactive version:")
    print("  python rag_system.py")
    print("\nTo run the web interface:")
    print("  streamlit run streamlit_app.py")

if __name__ == "__main__":
    run_demo()
