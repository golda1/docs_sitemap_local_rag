# ingest.py
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables (e.g., for Ollama base URL if not default)
load_dotenv()

# --- Configuration ---
DATA_PATH = "data/websphere_liberty_docs.txt" # Path to your large text file
CHROMA_DB_DIR = "chroma_db" # Directory to store ChromaDB persistent data
COLLECTION_NAME = "websphere_liberty_collection" # Name of your ChromaDB collection
OLLAMA_BASE_URL = os.getenv("OLLAMA_API_URL", "http://raspberrypi03.wifi:11434/v1") # Your Ollama server URL
EMBEDDING_MODEL_NAME = "nomic-embed-text" # The embedding model you pulled via Ollama

def ingest_documents():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a ChromaDB vector store.
    """
    print(f"--- Starting document ingestion for {DATA_PATH} ---")

    # 1. Load Documents
    try:
        loader = TextLoader(DATA_PATH)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from {DATA_PATH}.")
    except Exception as e:
        print(f"Error loading document: {e}")
        print("Please ensure 'data/websphere_liberty_docs.txt' exists and is readable.")
        return

    # 2. Split Documents into Chunks
    # RecursiveCharacterTextSplitter is good for maintaining context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Max characters per chunk
        chunk_overlap=200,    # Overlap between chunks to maintain context
        length_function=len,  # Use character length
        add_start_index=True, # Add metadata about chunk's start position
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")
    # Optional: Print a sample chunk to verify
    if chunks:
        print("\n--- Sample Chunk (first 200 chars) ---")
        print(chunks[0].page_content[:200])
        print("------------------------------------")

    # 3. Initialize Embedding Model
    # IMPORTANT: Use the same base_url and model name as your Ollama setup
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )
    print(f"Initialized embedding model: {EMBEDDING_MODEL_NAME} from {OLLAMA_BASE_URL}")

    # 4. Create and Persist ChromaDB
    # This will create the 'chroma_db' folder and store the embeddings
    # If the directory already exists, it will load the existing DB.
    # We use a specific collection name to organize data.
    print(f"Creating/updating ChromaDB at '{CHROMA_DB_DIR}' with collection '{COLLECTION_NAME}'...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    db.persist() # Explicitly persist changes
    print(f"ChromaDB ingestion complete. {db._collection.count()} items in collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    # You can optionally create a .env file in your project root
    # with OLLAMA_API_URL=http://raspberrypi03.wifi:11434/v1 if it's not localhost
    ingest_documents()

