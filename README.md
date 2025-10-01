# RAG System Demo

A complete Retrieval-Augmented Generation (RAG) system built with LangChain, demonstrating document processing, vector search, and question-answering capabilities.

## Features

- **Document Processing**: Load and process PDF and text documents
- **Text Chunking**: Intelligent document splitting for better retrieval
- **Vector Search**: ChromaDB-based vector storage with OpenAI embeddings
- **Question Answering**: LangChain-powered RAG with source attribution
- **Web Interface**: Clean Streamlit UI for easy interaction
- **CLI Interface**: Command-line interface for programmatic usage

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Key**
   ```bash
   cp env_example.txt .env
   # Edit .env and add your OpenAI API key
   ```

3. **Run the Web Interface**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Or Run CLI Version**
   ```bash
   python rag_system.py
   ```

## How It Works

The RAG system follows these steps:

1. **Document Loading**: Loads PDF and text files using LangChain loaders
2. **Text Splitting**: Breaks documents into chunks using RecursiveCharacterTextSplitter
3. **Embedding**: Converts text chunks to vectors using OpenAI embeddings
4. **Vector Storage**: Stores embeddings in ChromaDB for fast similarity search
5. **Retrieval**: Finds relevant document chunks for user questions
6. **Generation**: Uses OpenAI LLM to generate answers based on retrieved context

## Architecture

```
Documents → Text Splitter → Embeddings → Vector Store
                                              ↓
User Question → Retriever → Context + Question → LLM → Answer
```

## Key Components

- **RAGSystem**: Main class handling document processing and QA
- **Document Loaders**: Support for PDF and text files
- **Text Splitter**: RecursiveCharacterTextSplitter for optimal chunking
- **Vector Store**: ChromaDB for persistent vector storage
- **Retriever**: Similarity-based document retrieval
- **QA Chain**: LangChain RetrievalQA for question answering

## Usage Examples

### Basic Usage
```python
from rag_system import RAGSystem

# Initialize system
rag = RAGSystem(openai_api_key="your-key")

# Load documents
docs = rag.load_documents(["document.pdf"])
chunks = rag.process_documents(docs)

# Create vector store
rag.create_vectorstore(chunks)
rag.setup_qa_chain()

# Ask questions
result = rag.ask_question("What is machine learning?")
print(result["answer"])
```

### Advanced Features
```python
# Get similar documents
similar_docs = rag.get_similar_documents("AI applications", k=5)

# Custom chunking parameters
rag.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
```

## Configuration

- **Chunk Size**: Default 1000 characters (adjustable)
- **Chunk Overlap**: Default 200 characters (adjustable)
- **Retrieval Count**: Default 4 similar documents
- **Temperature**: Default 0.7 for LLM responses

## File Structure

```
rag_demo/
├── rag_system.py          # Main RAG implementation
├── streamlit_app.py       # Web interface
├── requirements.txt       # Dependencies
├── env_example.txt        # Environment variables template
├── sample_docs/           # Sample documents
└── chroma_db/            # Vector database (created on first run)
```

## Dependencies

- `langchain`: Core RAG framework
- `langchain-openai`: OpenAI integration
- `chromadb`: Vector database
- `pypdf`: PDF processing
- `streamlit`: Web interface
- `python-dotenv`: Environment management

## Notes

- Requires OpenAI API key for embeddings and LLM
- Vector database persists between sessions
- Supports PDF and text file formats
- Includes sample documents for testing
- Web interface provides real-time chat experience

Built with ❤️ using LangChain and Streamlit
