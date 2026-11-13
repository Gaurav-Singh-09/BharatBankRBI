import os
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd

class RBIPDFLoader:
    def __init__(self, raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed"):
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> list[dict]:
        """Extracts text, page numbers, and metadata from a given RBI PDF."""
        doc = fitz.open(pdf_path)
        extracted_pages = []

        print(f"Processing: {pdf_path.name} ({len(doc)} pages)")

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Clean up whitespace
            cleaned_text = " ".join(text.split())

            if cleaned_text:  # Only store if page has text
                extracted_pages.append({
                    "document_title": pdf_path.stem,
                    "file_name": pdf_path.name,
                    "page_number": page_num + 1,
                    "text": cleaned_text
                })

        return extracted_pages

    def process_all_pdfs(self) -> pd.DataFrame:
        """Processes all PDFs in the raw directory and returns a consolidated DataFrame."""
        all_pages = []
        pdf_files = list(self.raw_dir.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.raw_dir}. Please add some RBI PDFs!")
            return pd.DataFrame()

        for pdf_path in pdf_files:
            pages = self.extract_text_from_pdf(pdf_path)
            all_pages.extend(pages)

        df = pd.DataFrame(all_pages)

        # Save to processed directory for downstream chunking
        output_path = self.processed_dir / "extracted_rbi_pages.csv"
        df.to_csv(output_path, index=False)
        print(f"Successfully processed {len(pdf_files)} documents. Saved to {output_path}")

        return df

if __name__ == "__main__":
    loader = RBIPDFLoader()
    df_result = loader.process_all_pdfs()
    print(df_result.head())