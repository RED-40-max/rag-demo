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
        """Extract meaningful keywords from text with semantic understanding"""
        words = text.lower().split()
        
        # Extended stop words - common words that don't add meaning
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", 
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "can", "may", "might", "must", "shall", "get", "got", "go", "goes", "going",
            "come", "comes", "came", "see", "saw", "seen", "know", "knew", "known",
            "think", "thought", "make", "made", "take", "took", "taken", "give", "gave",
            "given", "say", "said", "tell", "told", "ask", "asked", "want", "wanted",
            "need", "needed", "like", "liked", "love", "loved", "feel", "felt", "seem",
            "seemed", "look", "looked", "find", "found", "use", "used", "work", "worked",
            "try", "tried", "help", "helped", "turn", "turned", "start", "started",
            "show", "showed", "hear", "heard", "play", "played", "run", "ran", "move",
            "moved", "live", "lived", "believe", "believed", "hold", "held", "bring",
            "brought", "happen", "happened", "write", "wrote", "written", "sit", "sat",
            "stand", "stood", "lose", "lost", "pay", "paid", "meet", "met", "include",
            "included", "continue", "continued", "set", "put", "end", "ended", "follow",
            "followed", "stop", "stopped", "create", "created", "speak", "spoke", "spoken",
            "read", "allow", "allowed", "add", "added", "spend", "spent", "grow", "grew",
            "open", "opened", "walk", "walked", "win", "won", "offer", "offered", "remember",
            "remembered", "love", "loved", "consider", "considered", "appear", "appeared",
            "buy", "bought", "wait", "waited", "serve", "served", "die", "died", "send",
            "sent", "expect", "expected", "build", "built", "stay", "stayed", "fall",
            "fell", "cut", "reach", "reached", "kill", "killed", "remain", "remained",
            "suggest", "suggested", "raise", "raised", "pass", "passed", "sell", "sold",
            "require", "required", "report", "reported", "decide", "decided", "pull",
            "pulled", "return", "returned", "explain", "explained", "hope", "hoped",
            "develop", "developed", "carry", "carried", "break", "broke", "broken",
            "receive", "received", "agree", "agreed", "support", "supported", "hit",
            "hated", "produce", "produced", "eat", "ate", "eaten", "cover", "covered",
            "catch", "caught", "draw", "drew", "drawn", "choose", "chose", "chosen",
            "deal", "dealt", "win", "won", "question", "questions", "back", "front",
            "left", "right", "up", "down", "out", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "just", "now", "really", "well", "also", "even", "still", "much", "way",
            "good", "great", "new", "first", "last", "long", "little", "own", "other",
            "old", "right", "big", "high", "different", "small", "large", "next", "early",
            "young", "important", "few", "public", "bad", "same", "able", "free", "local",
            "sure", "better", "best", "better", "worse", "worst", "easy", "hard", "simple",
            "complex", "clear", "obvious", "possible", "impossible", "likely", "unlikely",
            "certain", "uncertain", "true", "false", "real", "fake", "actual", "virtual",
            "normal", "abnormal", "regular", "irregular", "special", "general", "specific",
            "general", "particular", "common", "uncommon", "rare", "frequent", "occasional",
            "constant", "temporary", "permanent", "recent", "ancient", "modern", "old",
            "new", "fresh", "stale", "clean", "dirty", "pure", "impure", "safe", "dangerous",
            "secure", "insecure", "stable", "unstable", "strong", "weak", "powerful",
            "powerless", "heavy", "light", "thick", "thin", "wide", "narrow", "deep",
            "shallow", "tall", "short", "long", "brief", "quick", "slow", "fast", "rapid",
            "sudden", "gradual", "immediate", "delayed", "early", "late", "soon", "later",
            "before", "after", "during", "while", "since", "until", "unless", "if", "when",
            "where", "why", "how", "what", "who", "which", "whose", "whom", "whether"
        }
        
        # Semantic keyword mapping for better understanding
        semantic_mappings = {
            # Food and hunger related
            "hungry": ["food", "eat", "eating", "meal", "hunger", "hungry", "starving", "appetite"],
            "food": ["food", "eat", "eating", "meal", "hunger", "hungry", "starving", "appetite", "cook", "cooking", "recipe", "ingredient", "taste", "flavor", "delicious", "yummy", "tasty"],
            "sweet": ["sweet", "sugar", "dessert", "cake", "cookie", "candy", "chocolate", "sugar", "honey", "treat", "snack"],
            "treat": ["treat", "dessert", "sweet", "snack", "indulgence", "reward", "pleasure"],
            "eat": ["eat", "eating", "food", "meal", "consume", "devour", "taste", "bite", "chew"],
            "cook": ["cook", "cooking", "recipe", "bake", "baking", "prepare", "kitchen", "ingredient"],
            "recipe": ["recipe", "cook", "cooking", "bake", "baking", "ingredient", "instruction", "method"],
            "cookie": ["cookie", "biscuit", "sweet", "dessert", "bake", "baking", "chocolate", "treat"],
            "cake": ["cake", "dessert", "sweet", "bake", "baking", "birthday", "celebration", "treat"],
            "cupcake": ["cupcake", "cake", "dessert", "sweet", "bake", "baking", "treat", "small"],
            
            # General concepts
            "want": ["want", "desire", "wish", "need", "require", "seek", "look for", "crave"],
            "need": ["need", "require", "necessary", "essential", "important", "must have"],
            "like": ["like", "enjoy", "love", "prefer", "favorite", "fond of", "appreciate"],
            "love": ["love", "adore", "enjoy", "like", "passion", "affection", "cherish"],
            "help": ["help", "assist", "support", "aid", "guide", "advice", "suggestion"],
            "learn": ["learn", "study", "education", "knowledge", "understand", "comprehend", "teach"],
            "know": ["know", "understand", "comprehend", "aware", "familiar", "expert", "knowledge"],
            "find": ["find", "discover", "locate", "search", "seek", "uncover", "reveal"],
            "make": ["make", "create", "build", "construct", "produce", "generate", "form"],
            "use": ["use", "utilize", "employ", "apply", "operate", "function", "work"],
            
            # Technology and devices
            "computer": ["computer", "pc", "laptop", "desktop", "machine", "device", "technology"],
            "phone": ["phone", "mobile", "cell", "smartphone", "device", "communication"],
            "drive": ["drive", "storage", "hard drive", "ssd", "memory", "disk", "usb", "flash"],
            "storage": ["storage", "memory", "drive", "disk", "space", "capacity", "save"],
            
            # Animals and nature
            "octopus": ["octopus", "sea", "ocean", "marine", "animal", "creature", "tentacle"],
            "animal": ["animal", "creature", "wildlife", "nature", "beast", "pet"],
            "sea": ["sea", "ocean", "water", "marine", "aquatic", "underwater", "deep"],
            
            # Relationships and social
            "kiss": ["kiss", "romance", "love", "affection", "relationship", "intimate", "passion"],
            "relationship": ["relationship", "love", "romance", "partner", "couple", "dating"],
        }
        
        keywords = []
        for word in words:
            clean_word = word.strip(".,!?;:\"'()[]{}")
            if len(clean_word) > 2 and clean_word not in stop_words:
                keywords.append(clean_word)
                
                # Add semantic mappings
                if clean_word in semantic_mappings:
                    keywords.extend(semantic_mappings[clean_word])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords[:20]  # Top 20 keywords
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar chunks with intelligent scoring"""
        query_keywords = self.extract_keywords(query)
        
        # Define high-value semantic categories
        high_value_categories = {
            "food": ["food", "eat", "eating", "meal", "hunger", "hungry", "starving", "appetite", "cook", "cooking", "recipe", "ingredient", "taste", "flavor", "delicious", "yummy", "tasty", "sweet", "dessert", "cake", "cookie", "candy", "chocolate", "treat", "snack"],
            "technology": ["computer", "pc", "laptop", "desktop", "machine", "device", "technology", "phone", "mobile", "cell", "smartphone", "drive", "storage", "hard drive", "ssd", "memory", "disk", "usb", "flash"],
            "animals": ["octopus", "sea", "ocean", "marine", "animal", "creature", "tentacle", "wildlife", "nature", "beast", "pet"],
            "relationships": ["kiss", "romance", "love", "affection", "relationship", "intimate", "passion", "partner", "couple", "dating"]
        }
        
        chunk_scores = {}
        
        for chunk in self.chunks:
            chunk_keywords = self.extract_keywords(chunk["content"])
            
            # Calculate base similarity score
            common_keywords = set(query_keywords) & set(chunk_keywords)
            base_score = len(common_keywords)
            
            if base_score > 0:
                # Calculate semantic bonus
                semantic_bonus = 0
                
                # Check for high-value category matches
                for category, category_keywords in high_value_categories.items():
                    query_in_category = any(keyword in category_keywords for keyword in query_keywords)
                    chunk_in_category = any(keyword in category_keywords for keyword in chunk_keywords)
                    
                    if query_in_category and chunk_in_category:
                        semantic_bonus += 3  # High bonus for category matches
                
                # Bonus for exact word matches
                exact_matches = sum(1 for word in query_keywords if word in chunk_keywords)
                exact_bonus = exact_matches * 2
                
                # Bonus for title relevance
                title_keywords = self.extract_keywords(chunk["title"])
                title_matches = len(set(query_keywords) & set(title_keywords))
                title_bonus = title_matches * 1.5
                
                # Calculate final score
                final_score = base_score + semantic_bonus + exact_bonus + title_bonus
                
                # Boost score for food-related queries
                if any(word in query.lower() for word in ["hungry", "eat", "food", "sweet", "treat", "cook", "recipe", "meal"]):
                    if any(word in chunk_keywords for word in ["food", "eat", "eating", "meal", "hunger", "cook", "cooking", "recipe", "ingredient", "taste", "flavor", "delicious", "yummy", "tasty", "sweet", "dessert", "cake", "cookie", "candy", "chocolate", "treat", "snack"]):
                        final_score += 5  # High boost for food relevance
                
                chunk_scores[chunk["chunk_id"]] = {
                    "chunk": chunk,
                    "score": final_score,
                    "common_keywords": list(common_keywords),
                    "semantic_bonus": semantic_bonus,
                    "exact_bonus": exact_bonus,
                    "title_bonus": title_bonus
                }
        
        # Sort by final score and return top k
        sorted_chunks = sorted(chunk_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_chunks[:k]
    
    def generate_answer(self, query: str, context_chunks: List[Dict]) -> str:
        """Generate intelligent answer based on context"""
        if not context_chunks:
            return "I couldn't find relevant information to answer your question based on the scraped content."
        
        # Analyze query intent
        query_lower = query.lower()
        
        # Food-related responses
        if any(word in query_lower for word in ["hungry", "eat", "food", "sweet", "treat", "cook", "recipe", "meal"]):
            # Find food-related chunks first
            food_chunks = []
            other_chunks = []
            
            for chunk_info in context_chunks:
                chunk = chunk_info["chunk"]
                chunk_keywords = self.extract_keywords(chunk["content"])
                
                if any(word in chunk_keywords for word in ["food", "eat", "eating", "meal", "hunger", "cook", "cooking", "recipe", "ingredient", "taste", "flavor", "delicious", "yummy", "tasty", "sweet", "dessert", "cake", "cookie", "candy", "chocolate", "treat", "snack"]):
                    food_chunks.append(chunk_info)
                else:
                    other_chunks.append(chunk_info)
            
            # Prioritize food chunks
            prioritized_chunks = food_chunks + other_chunks
        else:
            prioritized_chunks = context_chunks
        
        # Generate contextual answer
        if any(word in query_lower for word in ["hungry", "eat", "food"]):
            answer = "Here are some food-related options I found:\n\n"
        elif any(word in query_lower for word in ["sweet", "treat", "dessert"]):
            answer = "Here are some sweet treats I found:\n\n"
        elif any(word in query_lower for word in ["cook", "recipe"]):
            answer = "Here are some cooking options I found:\n\n"
        else:
            answer = "Based on the scraped content:\n\n"
        
        # Add context from top chunks
        context_parts = []
        for chunk_info in prioritized_chunks[:3]:  # Limit to top 3 most relevant
            chunk = chunk_info["chunk"]
            context_parts.append(f"From {chunk['title']} ({chunk['url']}):\n{chunk['content'][:400]}...")
        
        context = "\n\n".join(context_parts)
        answer += context
        
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
