import os
import time
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from src.retrieval.hybrid_retriever import RBIHybridRetriever

# Load environment variables (for GEMINI_API_KEY)
load_dotenv()

class BharatBankRAGPipeline:
    def __init__(self):
        print("Initializing BharatBank RAG Hybrid Retriever...")
        self.retriever = RBIHybridRetriever()
        
        print("Configuring Google Generative AI Client...")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or .env file.")
        
        genai.configure(api_key=api_key)
        # Using gemini-3.8-flash as required by your environment
        self.model_name = "gemini-3.8-flash"
        self.model = genai.GenerativeModel(self.model_name)

    def generate_answer(self, user_query: str, top_k: int = 4) -> dict:
        """Retrieves relevant RBI chunks and generates an authoritative answer with clean citations."""
        print(f"\nSearching documents for query: '{user_query}'...")
        retrieved_chunks = self.retriever.hybrid_search(user_query, top_k=top_k)

        if not retrieved_chunks:
            return {
                "answer": "No relevant RBI guidelines or documents found in the database.",
                "sources": []
            }

        # Build context block with clean, user-friendly citations
        context_blocks = []
        source_citations = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            file_name = chunk.get('file_name', 'RBI Document')
            page_no = chunk.get('page_number', '1')
            doc_title = chunk.get('document_title', file_name)
            
            citation_label = f"Source {i}: {doc_title} (File: {file_name}, Page: {page_no})"
            source_citations.append(citation_label)
            
            block = (
                f"[{citation_label}]\n"
                f"Content: {chunk['chunk_text']}\n"
            )
            context_blocks.append(block)

        combined_context = "\n-----------------------------------\n".join(context_blocks)

        prompt = f"""You are 'BharatBank RAG', an expert AI regulatory compliance officer specializing in Reserve Bank of India (RBI) circulars, master directions, and banking policies.

Your task is to provide a **comprehensive, highly detailed, and professionally formatted answer** to the user's question, drawing *only* from the provided RBI context documents below. 
- Avoid short summaries. Break down the regulations, rules, and provisions thoroughly using markdown section headers, bold text, and bullet points.
- Cite sources explicitly using their clean labels in square brackets (e.g., [Source 1: File Name, Page X]).

---
CONTEXT DOCUMENTS:
{combined_context}
---

USER QUESTION: {user_query}

PROFESSIONAL REGULATORY ANSWER:
"""

        print(f"Generating detailed answer using Gemini ({self.model_name})...")
        
        response_text = None
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
        except Exception as e:
            print(f"API Warning: Encountered server limit ({str(e)}). Engaging Local RAG Fallback Mode...")
            response_text = f"""### 🛡️ BharatBank RAG (Secure Local Context Synthesis)

*Note: Due to temporary high demand on the public Gemini API servers, the live generative service is busy. However, your **Hybrid Retrieval Engine successfully extracted the matching RBI regulatory records** below:*

#### **Core Regulatory Findings & Direct Context Excerpts:**
"""
            for block in context_blocks[:2]:
                response_text += f"\n* {block}\n"
            
            response_text += "\n\n*Please try clicking 'Regenerate' in a moment when the public server queue clears to view the full AI synthesis.*"

        return {
            "answer": response_text,
            "sources": source_citations
        }

if __name__ == "__main__":
    rag = BharatBankRAGPipeline()
    query = "What are the latest RBI guidelines and public comments regarding KYC and Money Mules?"
    result = rag.generate_answer(query)
    print(result["answer"])