import pandas as pd
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

class RBIHybridRetriever:
    def __init__(self, 
                 chunked_data_path: str = "data/processed/chunked_rbi_documents.csv",
                 model_name: str = "BAAI/bge-m3",
                 collection_name: str = "bharatbank_rbi",
                 vector_db_path: str = "vector_store"):
        
        self.chunked_path = Path(chunked_data_path)
        self.collection_name = collection_name
        self.vector_db_path = vector_db_path

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model '{model_name}' for query encoding on {self.device.upper()}...")
        self.model = SentenceTransformer(model_name, device=self.device)

        # Connect to Qdrant
        self.qdrant_client = QdrantClient(path=self.vector_db_path)

        # Load chunks for BM25 keyword search
        print("Initializing BM25 keyword search index...")
        if self.chunked_path.exists():
            self.df_chunks = pd.read_csv(self.chunked_path)
            tokenized_corpus = [str(text).lower().split() for text in self.df_chunks["chunk_text"].fillna("").tolist()]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.df_chunks = pd.DataFrame()
            self.bm25 = None
            print("Warning: Chunked data file not found for BM25 initialization.")

    def dense_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs dense vector similarity search using Qdrant."""
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )

        results = []
        for scored_point in search_result.points:
            payload = scored_point.payload
            results.append({
                "chunk_id": payload["chunk_id"],
                "document_title": payload["document_title"],
                "file_name": payload["file_name"],
                "page_number": payload["page_number"],
                "chunk_text": payload["chunk_text"],
                "score": scored_point.score,
                "retrieval_type": "dense"
            })
        return results

    def keyword_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Performs exact keyword matching using BM25."""
        if not self.bm25 or self.df_chunks.empty:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            row = self.df_chunks.iloc[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "document_title": row["document_title"],
                "file_name": row["file_name"],
                "page_number": int(row["page_number"]),
                "chunk_text": row["chunk_text"],
                "score": float(scores[idx]),
                "retrieval_type": "keyword"
            })
        return results

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Combines dense and keyword search results."""
        dense_results = self.dense_search(query, top_k=top_k)
        keyword_results = self.keyword_search(query, top_k=top_k)

        # Merge and deduplicate based on chunk_id, prioritizing dense results
        seen = set()
        combined = []

        for res in dense_results + keyword_results:
            if res["chunk_id"] not in seen:
                seen.add(res["chunk_id"])
                combined.append(res)

        return combined[:top_k]

if __name__ == "__main__":
    retriever = RBIHybridRetriever()
    test_query = "What are the guidelines for digital lending and KYC?"
    print(f"\nRunning test hybrid search for query: '{test_query}'\n")
    results = retriever.hybrid_search(test_query, top_k=3)
    
    for i, res in enumerate(results, 1):
        print(f"[{i}] Doc: {res['document_title']} (Page {res['page_number']}) | Type: {res['retrieval_type']}")
        print(f"Snippet: {res['chunk_text'][:200]}...\n")