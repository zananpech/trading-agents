"""
Document ingestion pipeline for RAG.
Parses PDFs and HTML files, extracts charts/tables, transcribes images using Gemini,
chunks contents contextually, and stores in local ChromaDB.
"""
from __future__ import annotations

import os
import re
import glob
from typing import List
from PIL import Image

import pymupdf4llm
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai import types

from trading_agents.config import (
    GOOGLE_API_KEY,
    CHROMA_DB_PATH,
    RAG_REPORTS_DIR,
    RAG_IMAGES_DIR,
)

def extract_ticker_from_filename(filepath: str) -> str:
    """
    Extracts stock ticker from report filename.
    Examples:
        'data/reports/AAPL_10Q.pdf' -> 'AAPL'
        'data/reports/TSLA-10K.html' -> 'TSLA'
    """
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    # Split by common delimiters: underscores, hyphens, spaces
    parts = re.split(r'[-_\s]', name)
    if parts:
        ticker = parts[0].upper()
        # Tickers are typically 1-5 letters/numbers
        if re.match(r'^[A-Z0-9]{1,5}$', ticker):
            return ticker
    return "UNKNOWN"

def describe_chart_image(image_path: str, api_key: str) -> str:
    """
    Uses Google Gemini multimodal API to describe a financial chart or table image.
    """
    if not api_key:
        return "[Visual chart: API key missing, transcription skipped]"

    try:
        # Load the image
        img = Image.open(image_path)
        
        # Initialize the official Google Gen AI Client
        client = genai.Client(api_key=api_key)
        
        prompt = (
            "Analyze this financial chart/image from a company's quarterly/annual report. "
            "Provide a detailed textual description of the visual data: list the key metrics, labels, numbers, "
            "trends, axis descriptions, and any relevant qualitative findings shown in the graphic. "
            "Be extremely precise, factual, and quantitative. Do not speculate."
        )
        
        # Call the lightweight fast multimodal model
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[img, prompt]
        )
        
        return response.text.strip()
    except Exception as e:
        print(f"Error describing chart image {image_path}: {e}")
        return f"[Visual chart in report: {os.path.basename(image_path)}. Ingestion was unable to analyze due to error: {e}]"

def process_pdf_report(pdf_path: str, api_key: str) -> str:
    """
    Extracts markdown text, tables, and transcribes visual charts from a PDF.
    """
    print(f"Extracting markdown, tables, and images from PDF: {pdf_path}...")
    
    # Run pymupdf4llm extraction
    md_content = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=RAG_IMAGES_DIR
    )
    
    # Locate all markdown image references: e.g. ![](image_path) or ![alt](image_path)
    image_pattern = r'!\[(.*?)\]\((.*?)\)'
    matches = re.findall(image_pattern, md_content)
    
    if matches:
        print(f"Found {len(matches)} image/chart references. Generating captions using Gemini...")
        for alt_text, img_rel_path in matches:
            # Resolve physical image path
            full_img_path = img_rel_path
            if not os.path.isabs(full_img_path):
                # pymupdf4llm output paths are typically relative to where the tool is executed
                full_img_path = os.path.join(os.getcwd(), img_rel_path)
                
            if os.path.exists(full_img_path):
                print(f"  Describing chart: {os.path.basename(full_img_path)}...")
                chart_description = describe_chart_image(full_img_path, api_key)
                
                # Replace the original image reference with the reference + description block
                original_tag = f"![{alt_text}]({img_rel_path})"
                enriched_block = (
                    f"{original_tag}\n\n"
                    f"> **[Chart Analysis - {os.path.basename(full_img_path)}]:** {chart_description}\n"
                )
                md_content = md_content.replace(original_tag, enriched_block)
            else:
                print(f"  Warning: Extracted image file not found at {full_img_path}")
                
    return md_content

def process_html_report(html_path: str) -> str:
    """
    Extracts text contents from HTML reports.
    """
    print(f"Extracting text from HTML: {html_path}...")
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
        
    # Remove script, style, head, header, footer elements to clean up text
    for element in soup(["script", "style", "head", "header", "footer", "nav"]):
        element.decompose()
        
    # Extract structural text
    text = soup.get_text(separator="\n")
    # Normalize lines
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    cleaned_text = "\n".join(chunk for chunk in chunks if chunk)
    return cleaned_text

def ingest_document(filepath: str) -> int:
    """
    Ingests a single report into the local vector DB.
    Returns the number of chunks added.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return 0
        
    ticker = extract_ticker_from_filename(filepath)
    _, ext = os.path.splitext(filepath.lower())
    
    os.makedirs(RAG_IMAGES_DIR, exist_ok=True)
    
    # Extract text content
    if ext == ".pdf":
        raw_text = process_pdf_report(filepath, GOOGLE_API_KEY)
    elif ext in [".html", ".htm"]:
        raw_text = process_html_report(filepath)
    else:
        print(f"Unsupported file type: {ext} for {filepath}")
        return 0

    if not raw_text.strip():
        print(f"No content extracted from {filepath}")
        return 0

    # Split documents.
    # If it's PDF, it's rich markdown; split by header first, then recursively by character
    if ext == ".pdf":
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(raw_text)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.split_documents(md_header_splits)
    else:
        # Standard recursive character splitting for text/HTML
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        chunks = text_splitter.create_documents([raw_text])

    # Assign metadata to each chunk
    for chunk in chunks:
        chunk.metadata["ticker"] = ticker
        chunk.metadata["source"] = os.path.basename(filepath)

    print(f"Split {os.path.basename(filepath)} into {len(chunks)} chunks. Storing in ChromaDB...")
    
    # Store in ChromaDB
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    db.add_documents(chunks)
    
    if hasattr(db, "persist"):
        db.persist()
        
    print(f"Ingested {len(chunks)} chunks for {ticker} from {os.path.basename(filepath)}.")
    return len(chunks)

def ingest_directory(directory: str) -> int:
    """
    Scans directory for PDF and HTML reports and ingests them.
    Returns total chunks ingested.
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0

    pattern_pdf = os.path.join(directory, "*.pdf")
    pattern_html = os.path.join(directory, "*.html")
    pattern_htm = os.path.join(directory, "*.htm")
    
    files = glob.glob(pattern_pdf) + glob.glob(pattern_html) + glob.glob(pattern_htm)
    
    if not files:
        print(f"No PDF or HTML reports found in {directory}")
        return 0

    print(f"Found {len(files)} report(s) to ingest in {directory}.")
    total_chunks = 0
    for file in files:
        try:
            chunks_added = ingest_document(file)
            total_chunks += chunks_added
        except Exception as e:
            print(f"Error ingesting file {file}: {e}")
            
    print(f"Finished directory ingestion. Added {total_chunks} total chunks to ChromaDB.")
    return total_chunks
