#!/usr/bin/env python3
"""
Simple RAG Demo - No Dependencies Required
Shows the core concepts without needing to install packages
"""

import os
import json
from typing import List, Dict

class SimpleRAGDemo:
    """
    Simplified RAG system demonstration
    Shows the core concepts without external dependencies
    """
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vector_store = {}
        
    def load_sample_documents(self):
        """Load sample documents for demonstration"""
        self.documents = [
            {
                "content": """
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
                """,
                "source": "ai_overview.txt"
            },
            {
                "content": """
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
                """,
                "source": "machine_learning_basics.txt"
            }
        ]
        print(f"📚 Loaded {len(self.documents)} sample documents")
    
    def chunk_documents(self, chunk_size: int = 200):
        """Split documents into smaller chunks"""
        self.chunks = []
        
        for doc in self.documents:
            content = doc["content"].strip()
            words = content.split()
            
            # Simple chunking by word count
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 50:  # Only keep substantial chunks
                    self.chunks.append({
                        "content": chunk_text,
                        "source": doc["source"],
                        "chunk_id": len(self.chunks)
                    })
        
        print(f"✂️  Created {len(self.chunks)} document chunks")
    
    def create_simple_vector_store(self):
        """Create a simple keyword-based vector store"""
        self.vector_store = {}
        
        for chunk in self.chunks:
            # Simple keyword extraction (in real RAG, this would be embeddings)
            keywords = self.extract_keywords(chunk["content"])
            
            for keyword in keywords:
                if keyword not in self.vector_store:
                    self.vector_store[keyword] = []
                self.vector_store[keyword].append(chunk)
        
        print(f"🗄️  Created vector store with {len(self.vector_store)} keywords")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simplified version)"""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common words and keep meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        
        keywords = []
        for word in words:
            # Clean word
            clean_word = word.strip(".,!?;:\"'()[]{}")
            if len(clean_word) > 3 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def search_similar_chunks(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar chunks based on keyword overlap"""
        query_keywords = self.extract_keywords(query)
        
        chunk_scores = {}
        
        for chunk in self.chunks:
            chunk_keywords = self.extract_keywords(chunk["content"])
            
            # Calculate simple similarity score
            common_keywords = set(query_keywords) & set(chunk_keywords)
            score = len(common_keywords)
            
            if score > 0:
                chunk_scores[chunk["chunk_id"]] = {
                    "chunk": chunk,
                    "score": score,
                    "common_keywords": list(common_keywords)
                }
        
        # Sort by score and return top k
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_chunks[:k]
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate a simple answer based on context (simplified LLM simulation)"""
        if not context_chunks:
            return "I don't have enough information to answer that question based on the available documents."
        
        # Combine context from top chunks
        context = "\n\n".join([chunk["chunk"]["content"] for chunk in context_chunks])
        
        # Simple answer generation (in real RAG, this would use an LLM)
        answer = f"Based on the available information:\n\n{context[:500]}..."
        
        if len(context) > 500:
            answer += "\n\n[Answer truncated - in a real RAG system, an LLM would generate a comprehensive response]"
        
        return answer
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question and get an answer with sources"""
        print(f"\n🤔 Question: {question}")
        
        # Step 1: Retrieve relevant chunks
        similar_chunks = self.search_similar_chunks(question, k=3)
        
        if not similar_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        
        # Step 2: Generate answer
        answer = self.generate_answer(question, similar_chunks)
        
        # Step 3: Prepare sources
        sources = []
        for chunk_info in similar_chunks:
            sources.append({
                "content": chunk_info["chunk"]["content"][:200] + "...",
                "source": chunk_info["chunk"]["source"],
                "relevance_score": chunk_info["score"],
                "matching_keywords": chunk_info["common_keywords"]
            })
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def run_demo(self):
        """Run the complete RAG demo"""
        print("🚀 Simple RAG System Demo")
        print("=" * 50)
        
        # Step 1: Load documents
        print("\n📚 Step 1: Loading documents...")
        self.load_sample_documents()
        
        # Step 2: Chunk documents
        print("\n✂️  Step 2: Chunking documents...")
        self.chunk_documents()
        
        # Step 3: Create vector store
        print("\n🗄️  Step 3: Creating vector store...")
        self.create_simple_vector_store()
        
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
            
            result = self.ask_question(question)
            
            print(f"Answer: {result['answer']}")
            print(f"\nSources ({len(result['sources'])} documents):")
            
            for j, source in enumerate(result['sources'], 1):
                print(f"\n{j}. {source['content']}")
                print(f"   Source: {source['source']}")
                print(f"   Relevance: {source['relevance_score']} matching keywords")
                print(f"   Keywords: {', '.join(source['matching_keywords'])}")
            
            print("\n" + "=" * 50)
        
        print("\n🎉 Demo complete!")
        print("\nThis demonstrates the core RAG concepts:")
        print("1. Document loading and chunking")
        print("2. Vector store creation (simplified keyword-based)")
        print("3. Similarity search and retrieval")
        print("4. Answer generation with source attribution")
        print("\nIn a real RAG system:")
        print("- Documents would be processed with proper text splitters")
        print("- Embeddings would be generated using models like OpenAI's")
        print("- Vector search would use similarity algorithms")
        print("- An LLM would generate comprehensive answers")

def main():
    """Main function to run the demo"""
    demo = SimpleRAGDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
