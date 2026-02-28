# app/scripts/verify.py

import json

def verify_preprocessing():
    """Verify processed data without loading everything into memory"""
    
    filepath = 'Natural-Questions-Base/processed/processed_dataset.json'
    
    print("🔍 Verifying Processed Data\n")
    print("=" * 60)
    
    # Read file line by line (streaming)
    with open(filepath, 'r', encoding='utf-8') as f:
        # Skip the opening bracket [
        first_char = f.read(1)
        if first_char != '[':
            print("❌ Invalid JSON format")
            return
        
        # Read until we find first complete document
        buffer = ""
        documents = []
        depth = 0
        
        for line in f:
            buffer += line
            
            # Count braces to know when we have a complete document
            for char in line:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and len(documents) < 5:
                        # Try to parse the buffer
                        try:
                            # Clean up buffer (remove trailing commas)
                            clean_buffer = buffer.strip().rstrip(',')
                            doc = json.loads(clean_buffer)
                            documents.append(doc)
                            buffer = ""
                        except:
                            pass
            
            # Stop after getting 5 documents
            if len(documents) >= 5:
                break
    
    # Display samples
    print("\n📄 SAMPLE DOCUMENT (First Document):")
    print("=" * 60)
    if documents:
        doc = documents[0]
        print(f"ID: {doc['id']}")
        print(f"Question: {doc['question'][:100]}...")
        print(f"Chunk (first 200 chars): {doc['chunk'][:200]}...")
        print(f"Document (first 200 chars): {doc['document'][:200]}...")
        print(f"\nMetadata:")
        for key, value in doc['metadata'].items():
            print(f"  {key}: {value}")
    
    # Find a chunked example
    print("\n📄 SAMPLE CHUNKED DOCUMENT:")
    print("=" * 60)
    chunked = [d for d in documents if d['metadata']['total_chunks'] > 1]
    if chunked:
        doc = chunked[0]
        print(f"ID: {doc['id']}")
        print(f"Question: {doc['question'][:100]}...")
        print(f"Total Chunks: {doc['metadata']['total_chunks']}")
        print(f"Current Chunk Index: {doc['metadata']['chunk_index']}")
        print(f"Chunk Length: {doc['metadata']['chunk_length']}")
        print(f"Original Length: {doc['metadata']['original_length']}")
    else:
        print("No chunked documents found in first 5 samples")
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")

if __name__ == "__main__":
    verify_preprocessing()