#!/usr/bin/env python3
"""
Interactive Demo of Web RAG System
Shows how users can input URLs and ask questions
"""

from web_rag_system import WebRAGSystem

def demo_interactive():
    """Demo the interactive functionality"""
    print("🌐 Web RAG System - Interactive Demo")
    print("=" * 50)
    
    # Create RAG system
    rag = WebRAGSystem()
    
    # Simulate user input - these are example URLs
    print("\n📝 Simulating user input of URLs...")
    example_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    print("User enters these URLs:")
    for url in example_urls:
        print(f"  - {url}")
    
    # Scrape the URLs
    print(f"\n🔍 Scraping {len(example_urls)} URLs...")
    documents = rag.scrape_multiple_urls(example_urls, max_depth=0)
    
    if not documents:
        print("❌ No content was successfully scraped.")
        return
    
    # Process documents
    print("\n⚙️  Processing documents...")
    rag.chunk_documents()
    rag.create_vector_store()
    
    print("\n✅ RAG system ready!")
    print(f"📊 Stats: {len(rag.documents)} documents, {len(rag.chunks)} chunks, {len(rag.vector_store)} keywords")
    
    # Simulate user questions
    print("\n💬 Simulating user questions...")
    user_questions = [
        "What is artificial intelligence?",
        "What are the main applications of AI?",
        "What is machine learning?",
        "What are the different types of machine learning?",
        "How does deep learning work?"
    ]
    
    for i, question in enumerate(user_questions, 1):
        print(f"\n{'='*60}")
        print(f"Q{i}: {question}")
        print("-" * 40)
        
        result = rag.ask_question(question)
        
        print(f"💡 Answer:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])} unique sources found):")
            for j, source in enumerate(result['sources'][:3], 1):  # Show top 3
                print(f"\n{j}. {source['title']}")
                print(f"   URL: {source['url']}")
                print(f"   Relevance Score: {source['relevance_score']}")
                print(f"   Matching Keywords: {', '.join(source['matching_keywords'][:5])}")
                print(f"   Content Preview: {source['content'][:100]}...")
        else:
            print("\n📚 No sources found")
    
    print(f"\n{'='*60}")
    print("🎉 Interactive demo complete!")
    print("\nTo run this yourself:")
    print("1. Run: python3 web_rag_system.py")
    print("2. Choose option 1 (Interactive mode)")
    print("3. Enter your own URLs")
    print("4. Ask questions about the scraped content")

if __name__ == "__main__":
    demo_interactive()
