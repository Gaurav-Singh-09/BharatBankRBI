import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RBIChunker:
    def __init__(self, processed_data_path: str = "data/processed/extracted_rbi_pages.csv", 
                 chunk_output_path: str = "data/processed/chunked_rbi_documents.csv",
                 chunk_size: int = 800, chunk_overlap: int = 150):
        self.input_path = Path(processed_data_path)
        self.output_path = Path(chunk_output_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain's recursive text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_chunks(self) -> pd.DataFrame:
        if not self.input_path.exists():
            print(f"Error: Ingested file not found at {self.input_path}. Please run loader.py first!")
            return pd.DataFrame()

        df = pd.read_csv(self.input_path)
        print(f"Loaded {len(df)} pages from {self.input_path}. Starting chunking process...")

        chunked_records = []

        for idx, row in df.iterrows():
            text = str(row["text"])
            doc_title = row["document_title"]
            file_name = row["file_name"]
            page_num = row["page_number"]

            # Split text into chunks
            chunks = self.splitter.split_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                chunked_records.append({
                    "chunk_id": f"{doc_title}_p{page_num}_c{chunk_idx}",
                    "document_title": doc_title,
                    "file_name": file_name,
                    "page_number": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_text": chunk
                })

        df_chunks = pd.DataFrame(chunked_records)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df_chunks.to_csv(self.output_path, index=False)

        print(f"Successfully generated {len(df_chunks)} chunks from {len(df)} pages.")
        print(f"Saved chunked data to {self.output_path}")

        return df_chunks

if __name__ == "__main__":
    chunker = RBIChunker()
    df_chunks = chunker.process_chunks()
    print(df_chunks.head())