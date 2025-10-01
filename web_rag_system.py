#!/usr/bin/env python3
"""
Web RAG System - Interactive Web Scraping RAG
Allows users to input URLs, scrape content, and ask questions
"""

import os
import re
import time
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
import json

class WebRAGSystem:
    """
    Web-based RAG system that scrapes websites and answers questions
    """
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.vector_store = {}
        self.scraped_urls = set()
        
        # Headers to mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_url(self, url: str) -> Optional[Dict]:
        """
        Scrape content from a single URL
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with scraped content or None if failed
        """
        try:
            print(f"🌐 Scraping: {url}")
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Check if already scraped
            if url in self.scraped_urls:
                print(f"⚠️  Already scraped: {url}")
                return None
            
            # Make request
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else "Untitled"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.post', '.entry',
                '.article-content', '#content', '.main-content'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            # If no main content found, get all text
            if not content:
                content = soup.get_text()
            
            # Clean up content
            content = self.clean_text(content)
            
            if len(content) < 100:  # Skip if too short
                print(f"⚠️  Content too short: {url}")
                return None
            
            # Extract links for potential further scraping
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                if self.is_valid_url(full_url, url):
                    links.append(full_url)
            
            document = {
                "url": url,
                "title": title_text,
                "content": content,
                "links": links[:10],  # Limit to 10 links
                "scraped_at": time.time()
            }
            
            self.scraped_urls.add(url)
            print(f"✅ Successfully scraped: {title_text[:50]}...")
            return document
            
        except Exception as e:
            print(f"❌ Failed to scrape {url}: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        # Remove very short lines
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        return ' '.join(lines)
    
    def is_valid_url(self, url: str, base_url: str) -> bool:
        """Check if URL is valid for scraping"""
        try:
            parsed = urlparse(url)
            base_parsed = urlparse(base_url)
            
            # Must be http/https
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Must be from same domain (optional - you can change this)
            if parsed.netloc != base_parsed.netloc:
                return False
            
            # Skip common non-content URLs
            skip_patterns = [
                r'\.(pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar)$',
                r'\.(jpg|jpeg|png|gif|svg|ico)$',
                r'#',  # Skip anchors
                r'mailto:',  # Skip email links
                r'tel:',  # Skip phone links
            ]
            
            for pattern in skip_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            
            return True
        except:
            return False
    
    def scrape_multiple_urls(self, urls: List[str], max_depth: int = 1) -> List[Dict]:
        """
        Scrape multiple URLs with optional depth
        
        Args:
            urls: List of URLs to scrape
            max_depth: How many levels deep to scrape (1 = only provided URLs)
            
        Returns:
            List of scraped documents
        """
        documents = []
        urls_to_scrape = urls.copy()
        
        for depth in range(max_depth + 1):
            if not urls_to_scrape:
                break
            
            current_urls = urls_to_scrape.copy()
            urls_to_scrape = []
            
            for url in current_urls:
                if url in self.scraped_urls:
                    continue
                
                doc = self.scrape_url(url)
                if doc:
                    documents.append(doc)
                    
                    # Add links for next depth level
                    if depth < max_depth:
                        urls_to_scrape.extend(doc.get('links', [])[:5])  # Limit links per page
                
                # Be respectful - add delay
                time.sleep(1)
        
        self.documents.extend(documents)
        print(f"\n📚 Total documents scraped: {len(documents)}")
        return documents
    
    def chunk_documents(self, chunk_size: int = 300):
        """Split documents into smaller chunks"""
        self.chunks = []
        
        for doc in self.documents:
            content = doc["content"]
            words = content.split()
            
            # Create chunks
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunk_text = " ".join(chunk_words)
                
                if len(chunk_text.strip()) > 50:
                    self.chunks.append({
                        "content": chunk_text,
                        "url": doc["url"],
                        "title": doc["title"],
                        "chunk_id": len(self.chunks)
                    })
        
        print(f"✂️  Created {len(self.chunks)} document chunks")
    
    def create_vector_store(self):
        """Create keyword-based vector store"""
        self.vector_store = {}
        
        for chunk in self.chunks:
            keywords = self.extract_keywords(chunk["content"])
            
            for keyword in keywords:
                if keyword not in self.vector_store:
                    self.vector_store[keyword] = []
                self.vector_store[keyword].append(chunk)
        
        print(f"🗄️  Created vector store with {len(self.vector_store)} keywords")
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        keywords = []
        for word in words:
            clean_word = word.strip(".,!?;:\"'()[]{}")
            if len(clean_word) > 3 and clean_word not in stop_words:
                keywords.append(clean_word)
        
        return keywords[:15]  # Top 15 keywords
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks"""
        query_keywords = self.extract_keywords(query)
        
        chunk_scores = {}
        
        for chunk in self.chunks:
            chunk_keywords = self.extract_keywords(chunk["content"])
            
            # Calculate similarity score
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
        """Generate answer based on context"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question based on the scraped content."
        
        # Combine context
        context_parts = []
        for chunk_info in context_chunks:
            chunk = chunk_info["chunk"]
            context_parts.append(f"From {chunk['title']} ({chunk['url']}):\n{chunk['content'][:300]}...")
        
        context = "\n\n".join(context_parts)
        
        # Simple answer generation
        answer = f"Based on the scraped content:\n\n{context}"
        
        return answer
    
    def ask_question(self, question: str) -> Dict:
        """Ask a question and get answer with sources"""
        print(f"\n🤔 Question: {question}")
        
        # Search for relevant chunks
        similar_chunks = self.search_similar_chunks(question, k=5)
        
        if not similar_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "sources": []
            }
        
        # Generate answer
        answer = self.generate_answer(question, similar_chunks)
        
        # Prepare sources - deduplicate by URL and combine scores
        sources_dict = {}
        for chunk_info in similar_chunks:
            chunk = chunk_info["chunk"]
            url = chunk["url"]
            
            if url not in sources_dict:
                sources_dict[url] = {
                    "title": chunk["title"],
                    "url": url,
                    "content": chunk["content"][:200] + "...",
                    "relevance_score": chunk_info["score"],
                    "matching_keywords": set(chunk_info["common_keywords"])
                }
            else:
                # Combine scores and keywords for duplicate URLs
                sources_dict[url]["relevance_score"] = max(
                    sources_dict[url]["relevance_score"], 
                    chunk_info["score"]
                )
                sources_dict[url]["matching_keywords"].update(chunk_info["common_keywords"])
        
        # Convert back to list and sort by relevance
        sources = list(sources_dict.values())
        for source in sources:
            source["matching_keywords"] = list(source["matching_keywords"])
        
        # Sort by relevance score (highest first)
        sources.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def interactive_mode(self):
        """Run interactive mode for user input"""
        print("🌐 Web RAG System - Interactive Mode")
        print("=" * 50)
        
        # Get URLs from user
        print("\n📝 Enter URLs to scrape (one per line, empty line to finish):")
        urls = []
        while True:
            url = input("URL: ").strip()
            if not url:
                break
            urls.append(url)
        
        if not urls:
            print("❌ No URLs provided. Exiting.")
            return
        
        # Scrape URLs
        print(f"\n🔍 Scraping {len(urls)} URLs...")
        self.scrape_multiple_urls(urls, max_depth=0)  # Only scrape provided URLs
        
        if not self.documents:
            print("❌ No content was successfully scraped. Exiting.")
            return
        
        # Process documents
        print("\n⚙️  Processing documents...")
        self.chunk_documents()
        self.create_vector_store()
        
        print("\n✅ RAG system ready!")
        print("\n" + "=" * 50)
        
        # Interactive Q&A
        print("\n💬 Ask questions about the scraped content (type 'quit' to exit):")
        
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                result = self.ask_question(question)
                
                print(f"\n💡 Answer:")
                print(result['answer'])
                
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"\n{i}. {source['title']}")
                        print(f"   URL: {source['url']}")
                        print(f"   Relevance: {source['relevance_score']} matching keywords")
                        print(f"   Content: {source['content']}")
                        print(f"   Keywords: {', '.join(source['matching_keywords'])}")
                else:
                    print("\n📚 No sources found")
                
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def demo_mode(self):
        """Run demo with sample URLs"""
        print("🌐 Web RAG System - Demo Mode")
        print("=" * 50)
        
        # Sample URLs for demo
        demo_urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning"
        ]
        
        print(f"\n🔍 Demo: Scraping {len(demo_urls)} sample URLs...")
        for url in demo_urls:
            print(f"  - {url}")
        
        # Scrape URLs
        self.scrape_multiple_urls(demo_urls, max_depth=0)
        
        if not self.documents:
            print("❌ Demo failed - no content scraped")
            return
        
        # Process documents
        print("\n⚙️  Processing documents...")
        self.chunk_documents()
        self.create_vector_store()
        
        print("\n✅ Demo RAG system ready!")
        
        # Demo questions
        demo_questions = [
            "What is artificial intelligence?",
            "What are the main applications of AI?",
            "What is machine learning?",
            "What are the different types of machine learning?",
            "How does deep learning work?"
        ]
        
        print("\n🎯 Running demo questions...\n")
        
        for i, question in enumerate(demo_questions, 1):
            print(f"Q{i}: {question}")
            print("-" * 40)
            
            result = self.ask_question(question)
            
            print(f"Answer: {result['answer'][:300]}...")
            print(f"Sources: {len(result['sources'])} found")
            
            for j, source in enumerate(result['sources'][:2], 1):  # Show top 2 sources
                print(f"  {j}. {source['title']} (Score: {source['relevance_score']})")
            
            print("\n" + "=" * 50)

def main():
    """Main function"""
    print("🌐 Web RAG System")
    print("Choose mode:")
    print("1. Interactive mode (enter your own URLs)")
    print("2. Demo mode (uses sample URLs)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    rag = WebRAGSystem()
    
    if choice == "1":
        rag.interactive_mode()
    elif choice == "2":
        rag.demo_mode()
    else:
        print("Invalid choice. Running demo mode...")
        rag.demo_mode()

if __name__ == "__main__":
    main()
