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
        self.model_name = "gemini-3.8-flash"
        self.model = genai.GenerativeModel(self.model_name)

    def generate_answer(self, user_query: str, top_k: int = 4) -> dict:
        """Retrieves relevant RBI chunks and generates an authoritative answer or simple explanation with clean citations."""
        
        # Check if the query is a short follow-up button command
        is_followup = user_query.strip() in [
            "Explain this in simple terms", 
            "Give me a practical example", 
            "What are the exceptions?",
            "I still didn't get it"
        ]

        print(f"\nSearching documents for query: '{user_query}'...")
        retrieved_chunks = self.retriever.hybrid_search(user_query, top_k=top_k)

        if not retrieved_chunks:
            # Fallback search if a short phrase doesn't match specific doc keywords
            retrieved_chunks = self.retriever.hybrid_search("RBI Master Directions KYC loans compliance", top_k=top_k)

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

        # Adapt prompt based on whether it's a follow-up simplification request or standard query
        if is_followup or "simple" in user_query.lower() or "example" in user_query.lower():
            behavioral_instruction = (
                "CRITICAL INSTRUCTION: The user clicked a follow-up action or requested a simple explanation/example. "
                "Do NOT dump raw legal text. Break down the provided RBI regulations into simple, easy-to-understand "
                "plain English, using clear analogies, bullet points, or practical examples based on the context below."
            )
        else:
            behavioral_instruction = (
                "Your task is to provide a **comprehensive, highly detailed, and professionally formatted answer** to the user's question, drawing *only* from the provided RBI context documents below.\n"
                "- Avoid short summaries. Break down the regulations, rules, and provisions thoroughly using markdown section headers, bold text, and bullet points.\n"
                "- Cite sources explicitly using their clean labels in square brackets (e.g., [Source 1: File Name, Page X])."
            )

        prompt = f"""You are 'BharatBank RAG', an expert AI regulatory compliance officer specializing in Reserve Bank of India (RBI) circulars, master directions, and banking policies.

{behavioral_instruction}

---
CONTEXT DOCUMENTS:
{combined_context}
---

USER INPUT: {user_query}

ANSWER:
"""

        print(f"Generating answer using Gemini ({self.model_name})...")
        
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
    result = rag.generate_answer("What are the latest RBI guidelines regarding KYC?")
    print(result["answer"])