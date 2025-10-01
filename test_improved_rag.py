#!/usr/bin/env python3
"""
Test the improved RAG system with better semantic understanding
"""

from web_rag_system import WebRAGSystem

def test_improved_rag():
    """Test the improved RAG system"""
    print("🧠 Testing Improved RAG System")
    print("=" * 50)
    
    # Create RAG system
    rag = WebRAGSystem()
    
    # Test with food-related URLs
    test_urls = [
        "https://pinchofyum.com/the-best-soft-chocolate-chip-cookies",
        "https://www.recipetineats.com/vanilla-cupcakes/"
    ]
    
    print(f"\n🔍 Scraping {len(test_urls)} food-related URLs...")
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
    
    print("\n✅ Improved RAG system ready!")
    print(f"📊 Stats: {len(rag.documents)} documents, {len(rag.chunks)} chunks, {len(rag.vector_store)} keywords")
    
    # Test improved queries
    test_queries = [
        "I am hungry",
        "I want a sweet treat",
        "I want to eat something",
        "I need a recipe",
        "I want to cook something"
    ]
    
    print("\n🎯 Testing improved semantic understanding...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Q{i}: {query}")
        print("-" * 40)
        
        result = rag.ask_question(query)
        
        print(f"💡 Answer:")
        print(result['answer'][:300] + "..." if len(result['answer']) > 300 else result['answer'])
        
        if result['sources']:
            print(f"\n📚 Sources ({len(result['sources'])} unique sources found):")
            for j, source in enumerate(result['sources'][:2], 1):  # Show top 2
                print(f"\n{j}. {source['title']}")
                print(f"   URL: {source['url']}")
                print(f"   Relevance Score: {source['relevance_score']:.1f}")
                print(f"   Matching Keywords: {', '.join(source['matching_keywords'][:5])}")
        else:
            print("\n📚 No sources found")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_improved_rag()
