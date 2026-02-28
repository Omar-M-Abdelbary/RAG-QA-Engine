import json
from pathlib import Path
import random
from typing import List, Dict, Any

def load_processed_data(file_path: str, sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Load sample questions from processed dataset
    
    Args:
        file_path: Path to processed_dataset.json
        sample_size: Number of test samples to extract
    
    Returns:
        List of test samples with questions and answers
    """
    print(f" Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f" Loaded {len(data):,} total documents")
    
    # Filter for documents with clear answers
    valid_samples = []
    
    for doc in data:
        # Must have question and answer
        if 'question' in doc and 'answer' in doc and doc['answer']:
            valid_samples.append({
                'question': doc['question'],
                'answer': doc['answer'],
                'doc_id': doc.get('id', 'unknown')
            })
    
    print(f" Found {len(valid_samples):,} valid Q&A pairs")
    
    # Randomly sample
    if len(valid_samples) > sample_size:
        test_samples = random.sample(valid_samples, sample_size)
    else:
        test_samples = valid_samples
    
    print(f" Selected {len(test_samples)} test samples")
    
    return test_samples

def save_test_dataset(test_samples: List[Dict[str, Any]], output_path: str):
    """Save test dataset to file"""
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, indent=2, ensure_ascii=False)
    
    print(f" Saved test dataset to: {output_file}")
    print(f" Total samples: {len(test_samples)}")

def main():
    # Paths
    PROCESSED_DATA = "Natural-Questions-Base/processed/processed_dataset.json"
    OUTPUT_PATH = "evaluation/test_dataset.json"
    SAMPLE_SIZE = 100  # Number of test questions
    
    print("\n Creating Test Dataset for Evaluation")
    print("=" * 60)
    
    # Load samples
    test_samples = load_processed_data(PROCESSED_DATA, SAMPLE_SIZE)
    
    # Save test dataset
    save_test_dataset(test_samples, OUTPUT_PATH)
    
    # Show sample
    print("\n Sample Test Question:")
    print("=" * 60)
    sample = test_samples[0]
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer'][:200]}...")
    print("=" * 60)
    
    print("\n Test dataset ready!")
    print(f" Location: {OUTPUT_PATH}")
    print(f" Use this to evaluate your RAG system")

if __name__ == "__main__":
    main()