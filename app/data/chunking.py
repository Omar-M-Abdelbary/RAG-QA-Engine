from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import Config

class TextChunker:
    """Handle text chunking with configurable parameters"""
    
    def __init__(self):
        self.config = Config()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def should_chunk(self, text):
        """
        Decide if text should be chunked based on edge case rules
        """
        length = len(text)
        if length < self.config.THRESHOLD:
            return False

        if length >= self.config.THRESHOLD:
            return True

    
    def chunk_text(self, text):
        """
        Chunk text if needed, otherwise return as single chunk
        """
        if self.should_chunk(text):
            return self.text_splitter.split_text(text)
        else:
            return [text]