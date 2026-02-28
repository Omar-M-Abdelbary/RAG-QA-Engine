import os
from dotenv import load_dotenv

# ===== LOAD .ENV FILE FIRST =====
# Get the project root directory
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / '.env'

# Load environment variables from .env
load_dotenv(dotenv_path=ENV_PATH)

class Config:
    RAW_DATA_PATH = "Natural-Questions-Base/raw/Natural-Questions-Base.csv"
    PROCESSED_DATA_PATH = "Natural-Questions-Base/processed/processed_dataset.json"
    
    CHUNK_SIZE = 1000
    OVERLAP = 150
    THRESHOLD = 1500

    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION = 384  # Model output dimension
    BATCH_SIZE = 32  # Process 32 documents at a time
    
    FAISS_INDEX_PATH = "Natural-Questions-Base/indexes/faiss_index.bin"
    FAISS_METADATA_PATH = "Natural-Questions-Base/indexes/metadata.json"

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"  # Fast Groq model
    
    LLM_TEMPERATURE = 0.1  # Low = more focused, High = more creative
    LLM_MAX_TOKENS = 500   # Maximum response length
    

    TOP_K_RETRIEVAL = 5