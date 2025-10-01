#!/usr/bin/env python3
"""
Test script for interactive web RAG system
"""

from web_rag_system import WebRAGSystem

def test_interactive():
    """Test the interactive functionality"""
    rag = WebRAGSystem()
    
    # Test with some sample URLs
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    print("🌐 Testing Web RAG System with sample URLs")
    print("=" * 50)
    
    # Scrape URLs
    print(f"\n🔍 Scraping {len(test_urls)} URLs...")
    for url in test_urls:
        print(f"  - {url}")
    
    documents = rag.scrape_multiple_urls(test_urls, max_depth=0)
    
    if not documents:
        print("❌ No content scraped")
        return
    
    # Process documents
    print("\n⚙️  Processing documents...")
    rag.chunk_documents()
    rag.create_vector_store()
    
    print("\n✅ RAG system ready!")
    print(f"📊 Stats: {len(rag.documents)} documents, {len(rag.chunks)} chunks, {len(rag.vector_store)} keywords")
    
    # Test questions
    test_questions = [
        "What is artificial intelligence?",
        "What are the main applications of AI?",
        "What is machine learning?",
        "What are the different types of machine learning?",
        "How does deep learning work?"
    ]
    
    print("\n🎯 Testing questions...\n")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Q{i}: {question}")
        print("-" * 40)
        
        result = rag.ask_question(question)
        
        print(f"💡 Answer: {result['answer'][:200]}...")
        print(f"📚 Sources: {len(result['sources'])} found")
        
        if result['sources']:
            print("Top sources:")
            for j, source in enumerate(result['sources'][:3], 1):
                print(f"  {j}. {source['title']} (Score: {source['relevance_score']})")
                print(f"     URL: {source['url']}")
                print(f"     Keywords: {', '.join(source['matching_keywords'][:5])}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_interactive()
