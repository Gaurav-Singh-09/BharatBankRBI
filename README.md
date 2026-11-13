# 🏦 BharatBank · RBI Regulatory Compliance RAG Assistant

**BharatBank** is an advanced, enterprise-grade AI compliance platform designed to streamline regulatory research for banking professionals. By combining automated web scraping with a robust hybrid retrieval engine, BharatBank bridges the gap between complex Reserve Bank of India (RBI) circulars and precise, grounded AI answers.

## ✨ Key Features

* **🤖 Automated RBI Document Sync:** Employs an integrated BeautifulSoup scraper that automatically scans, purges stale records, and bulk-downloads the latest RBI Master Directions and notifications directly into local storage on startup[cite: 1].
* **🔍 Hybrid Retrieval Engine:** Combines dense vector semantic search (`sentence-transformers`) with keyword-based sparse retrieval (`BM25`) to capture both exact statutory phrases and conceptual policy intent.
* **🛡️ Grounded Compliance Answers:** Eliminates hallucinations by anchoring all LLM responses strictly in verified RBI PDF sources, complete with transparent source citations.
* **🎨 Professional Emerald UI:** Features a custom-styled, production-grade Streamlit interface designed with a forest black, emerald, and gold theme tailored for banking workflows.
* **📥 Session Controls & Export:** Supports multiple answer styles (Concise, Detailed, Plain English), session analytics, and one-click Markdown chat exports.

## 🛠️ Tech Stack

* **Frontend / UI:** Streamlit
* **Web Scraping:** BeautifulSoup4 (`bs4`), Requests
* **AI & Embeddings:** Hugging Face `sentence-transformers`, Transformers[cite: 7], Google Gemini / LLM integration
* **Retrieval & Storage:** Qdrant, PyMuPDF (`fitz`), Rank-BM25[cite: 8]
* **Configuration:** Python-dotenv[cite: 7]

---

## 🚀 Getting Started Locally

### 1. Clone the repository
```bash
git clone [https://github.com/Gaurav-Singh-09/bharatbank.git](https://github.com/Gaurav-Singh-09/bharatbank.git)
cd bharatbank
