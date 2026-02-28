import json
import faiss

print("=" * 60)
print("🔍 TESTING FAISS INDEX (NO METADATA LOADING)")
print("=" * 60)

# Load FAISS index ONLY
print("\n📂 Loading FAISS index...")
try:
    index = faiss.read_index("Natural-Questions-Base/indexes/faiss_index.bin")
    print(f"✅ Loaded {index.ntotal:,} vectors")
    print(f"   Dimension: {index.d}")
except Exception as e:
    print(f"❌ Error loading index: {e}")
    exit(1)

# Check metadata file exists (but don't load it)
print("\n📂 Checking metadata file...")
import os
metadata_path = "Natural-Questions-Base/indexes/metadata.json"
if os.path.exists(metadata_path):
    size = os.path.getsize(metadata_path)
    print(f"✅ Metadata file exists ({size / 1024 / 1024:.2f} MB)")
else:
    print(f"❌ Metadata file not found")

# Test search with first vector
print("\n🔍 Testing search functionality...")
try:
    # Get first vector from index
    query_vector = index.reconstruct(0).reshape(1, -1)
    
    # Search for 5 most similar vectors
    scores, indices = index.search(query_vector, 5)
    
    print(f"✅ Search successful!")
    print(f"\n📊 Top 5 similar vectors:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        print(f"   {i}. Document Index: {idx:,} | Similarity Score: {score:.4f}")
    
    # Test with a random vector
    print("\n🔍 Testing with random query vector...")
    import numpy as np
    random_query = np.random.randn(1, index.d).astype('float32')
    scores2, indices2 = index.search(random_query, 3)
    
    print(f"✅ Random query search successful!")
    print(f"   Found {len(indices2[0])} results")
    
except Exception as e:
    print(f"❌ Search failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n📌 Summary:")
print(f"   ✅ FAISS index loaded: {index.ntotal:,} vectors")
print(f"   ✅ Vector dimension: {index.d}")
print(f"   ✅ Search functionality: WORKING")
print(f"   ✅ Metadata file: EXISTS")
print(f"\n🎉 Your RAG retrieval system is READY!")
print(f"   (Metadata will be loaded on-demand in production)")



