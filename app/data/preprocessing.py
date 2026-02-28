import pandas as pd
from app.core.config import Config
from app.data.chunking import TextChunker

class DataPreprocessor:
    """Main preprocessing pipeline"""
    
    def __init__(self):
        self.config = Config()
        self.chunker = TextChunker()
    
    def select_answer(self, row):
        """
        Select which answer to use (Strategy A: prefer long)
        Returns: (answer_text, answer_type)
        """
        if pd.notnull(row['long_answers']):
            return row['long_answers'], "long"
        
        elif pd.notnull(row['short_answers']):
            return row['short_answers'], "short"
        
        else:
            return None, None
    
    def process_row(self, row):
        """
        Process a single question-answer pair
        Returns: List of document dictionaries
        """
        question = row['question']
        answer, answer_type = self.select_answer(row)
        
        if answer is None:
            return []  
        
    
        chunks = self.chunker.chunk_text(answer)
        
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = {
                'id': f"{row.name}_{idx}",
                'question': question,
                'answer': answer,
                'chunk': chunk,
                'document': f"{question} {chunk}",  # Question + Answer
                'metadata': {
                    'answer_type': answer_type,
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'chunk_length': len(chunk),
                    'original_length': len(answer)
                }
            }
            documents.append(doc)
        
        return documents
    
    def process_dataset(self, df):
        """
        Process entire dataset
        Returns: List of all processed documents
        """
        print("\n🔄 Processing dataset...")
        all_documents = []
        
        for idx, row in df.iterrows():
            docs = self.process_row(row)
            all_documents.extend(docs)
            
            if (idx + 1) % 10000 == 0:
                print(f"   Processed {idx + 1}/{len(df)} rows → {len(all_documents)} documents created")   # Progress update
        
        print(f"Processing complete: {len(all_documents)} total documents from {len(df)} rows")
        return all_documents