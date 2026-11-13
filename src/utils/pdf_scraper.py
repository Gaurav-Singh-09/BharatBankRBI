import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def sync_rbi_pdfs(target_url: str, save_dir: str = "data/rbi_docs"):
    """
    Scrapes a target webpage using BeautifulSoup, clears existing local PDFs, 
    and downloads all newly found PDFs in bulk.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[{target_url}] Connecting and scanning for fresh PDFs...")
    
    try:
        response = requests.get(target_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"Web scraping connection failed: {e}")
        return False

    # Parse HTML content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all anchor tags containing links
    links = soup.find_all("a")
    pdf_links = []
    
    for link in links:
        href = link.get("href")
        if href and (".pdf" in href.lower() or "pdf" in link.get("class", [])):
            absolute_url = urljoin(target_url, href)
            if absolute_url not in pdf_links:
                pdf_links.append(absolute_url)

    if not pdf_links:
        print("No PDF links found on the target page.")
        return False

    print(f"Found {len(pdf_links)} PDF links. Clearing old local storage files...")

    # Clear/delete existing files in the local storage directory first
    deleted_count = 0
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(".pdf"):
            os.remove(file_path)
            deleted_count += 1
            
    print(f"Cleared {deleted_count} stale PDF files from local storage.")

    # Download all current PDFs fresh
    downloaded_count = 0
    for i, pdf_url in enumerate(pdf_links, 1):
        try:
            parsed_url = urlparse(pdf_url)
            file_name = os.path.basename(parsed_url.path)
            if not file_name.lower().endswith(".pdf"):
                file_name = f"rbi_document_{i}.pdf"
                
            file_path = os.path.join(save_dir, file_name)
            
            # Stream download the PDF content
            pdf_response = requests.get(pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
            if pdf_response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(pdf_response.content)
                downloaded_count += 1
                print(f"Downloaded ({downloaded_count}/{len(pdf_links)}): {file_name}")
        except Exception as e:
            print(f"Failed to download {pdf_url}: {e}")

    print(f"Sync complete! Successfully stored {downloaded_count} fresh PDFs in '{save_dir}'.")
    return True