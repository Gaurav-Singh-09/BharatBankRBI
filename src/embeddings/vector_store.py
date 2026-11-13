import pandas as pd
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class RBIVectorStoreManager:
    def __init__(self, 
                 chunked_data_path: str = "data/processed/chunked_rbi_documents.csv",
                 model_name: str = "BAAI/bge-m3",
                 collection_name: str = "bharatbank_rbi",
                 vector_db_path: str = "vector_store"):
        
        self.chunked_path = Path(chunked_data_path)
        self.collection_name = collection_name
        self.vector_db_path = vector_db_path

        # Determine device (CUDA for your RTX 4050)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing embedding model '{model_name}' on device: {self.device.upper()}...")

        # Load local BGE-M3 embedding model
        self.model = SentenceTransformer(model_name, device=self.device)

        # Initialize local persistent Qdrant client
        print(f"Initializing Qdrant vector store at '{self.vector_db_path}'...")
        self.qdrant_client = QdrantClient(path=self.vector_db_path)

    def create_and_store_embeddings(self):
        if not self.chunked_path.exists():
            print(f"Error: Chunked file not found at {self.chunked_path}. Please run chunker.py first!")
            return

        df = pd.read_csv(self.chunked_path)
        print(f"Loaded {len(df)} chunks from {self.chunked_path}. Generating embeddings...")

        texts = df["chunk_text"].fillna("").tolist()
        
        # Generate dense embeddings using BGE-M3 on GPU
        embeddings = self.model.encode(
            texts, 
            batch_size=16, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )

        vector_size = embeddings.shape[1]
        print(f"Generated embeddings with dimension: {vector_size}")

        # Recreate Qdrant collection
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(self.collection_name)

        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

        # Prepare points for Qdrant
        points = []
        for idx, row in df.iterrows():
            points.append(
                PointStruct(
                    id=idx,
                    vector=embeddings[idx].tolist(),
                    payload={
                        "chunk_id": row["chunk_id"],
                        "document_title": row["document_title"],
                        "file_name": row["file_name"],
                        "page_number": int(row["page_number"]),
                        "chunk_index": int(row["chunk_index"]),
                        "chunk_text": row["chunk_text"]
                    }
                )
            )

        # Upsert points into Qdrant in batches
        print(f"Uploading {len(points)} vectors to Qdrant collection '{self.collection_name}'...")
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print("Successfully stored all vectors in Qdrant database!")

if __name__ == "__main__":
    manager = RBIVectorStoreManager()
    manager.create_and_store_embeddings()