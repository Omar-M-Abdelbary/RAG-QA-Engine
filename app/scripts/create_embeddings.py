import ijson
from app.services.embedding_service import EmbeddingService
from app.infrastructure.vector.faiss_store import VectorStore
from app.core.config import Config

def create_embeddings_batched():
    """
    Create embeddings using streaming to avoid memory issues
    """
    print("=" * 60)
    print(" CREATING EMBEDDINGS & FAISS INDEX")
    print("=" * 60)
    
    config = Config()
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    
    # Create FAISS index
    vector_store.create_index(config.EMBEDDING_DIMENSION)
    
    # Stream documents from JSON
    print(f"\n Streaming documents from {config.PROCESSED_DATA_PATH}")
    
    BATCH_SIZE = 10000
    batch_docs = []
    batch_texts = []
    total_processed = 0
    batch_count = 0
    
    with open(config.PROCESSED_DATA_PATH, 'r', encoding='utf-8') as f:
        # Stream parse the JSON array
        parser = ijson.items(f, 'item')
        
        for doc in parser:
            batch_docs.append(doc)
            batch_texts.append(doc['document'])
            
            # When batch is full, process it
            if len(batch_docs) >= BATCH_SIZE:
                batch_count += 1
                total_processed += len(batch_docs)
                
                print(f"\n Processing batch {batch_count}")
                print(f"   Documents: {total_processed - len(batch_docs)} to {total_processed}")
                
                # Create embeddings
                embeddings = embedding_service.embed_texts(batch_texts, show_progress=True)
                
                # Add to FAISS
                vector_store.add_embeddings(embeddings, batch_docs)
                
                # Clear batch
                batch_docs = []
                batch_texts = []
        
        # Process remaining documents
        if batch_docs:
            batch_count += 1
            total_processed += len(batch_docs)
            
            print(f"\n Processing final batch {batch_count}")
            print(f"   Documents: {total_processed - len(batch_docs)} to {total_processed}")
            
            embeddings = embedding_service.embed_texts(batch_texts, show_progress=True)
            vector_store.add_embeddings(embeddings, batch_docs)
    
    # Save everything
    print("\n Saving FAISS index and metadata...")
    vector_store.save()
    
    print("\n" + "=" * 60)
    print(" EMBEDDING CREATION COMPLETE!")
    print("=" * 60)
    print(f"Total documents processed: {total_processed:,}")
    print(f"Total vectors in index: {vector_store.index.ntotal:,}")
    print(f"Index saved to: {config.FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {config.FAISS_METADATA_PATH}")

if __name__ == "__main__":
    create_embeddings_batched()


