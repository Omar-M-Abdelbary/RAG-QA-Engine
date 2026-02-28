import json
import os
from app.data.dataset_loader import DatasetLoader
from app.data.preprocessing import DataPreprocessor
from app.core.config import Config

def main():
    """Main preprocessing pipeline"""
    
    print("=" * 60)
    print("🚀 RAG SYSTEM - DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    config = Config()
    
    # Step 1: Load data
    print("\n📊 STEP 1: Loading Raw Data")
    loader = DatasetLoader()
    df = loader.load_raw_data()
    loader.validate_data(df)
    
    # Step 2: Preprocess
    print("\n🔧 STEP 2: Preprocessing")
    preprocessor = DataPreprocessor()
    documents = preprocessor.process_dataset(df)
    
    # Step 3: Save processed data
    print("\n💾 STEP 3: Saving Processed Data")
    
    # Create directory if doesn't exist
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    
    print(f"   Saving to {config.PROCESSED_DATA_PATH}...")
    with open(config.PROCESSED_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(documents)} documents")
    
    # Step 4: Print statistics
    print("\n" + "=" * 60)
    print("📈 PREPROCESSING STATISTICS")
    print("=" * 60)
    print(f"Total input rows:        {len(df):,}")
    print(f"Total output documents:  {len(documents):,}")
    print(f"Avg docs per question:   {len(documents) / len(df):.2f}")
    
    # Count by answer type
    long_count = sum(1 for d in documents if d['metadata']['answer_type'] == 'long')
    short_count = sum(1 for d in documents if d['metadata']['answer_type'] == 'short')
    print(f"\nDocuments from long answers:  {long_count:,}")
    print(f"Documents from short answers: {short_count:,}")
    
    # Chunking stats
    chunked = sum(1 for d in documents if d['metadata']['total_chunks'] > 1)
    print(f"\nQuestions that were chunked:  {chunked:,}")
    print(f"Questions kept whole:         {len(df) - chunked:,}")
    
    print("\n" + "=" * 60)
    print("🎉 PREPROCESSING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()