import csv
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

from app import embedder, vectorstore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".xml",
    ".json",
    ".csv",
    ".html",
    ".htm",
    ".pdf",
    ".docx",
}

SAMPLE_DATA_DIR = Path(__file__).parent.parent / "sample_data"
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"


def _read_plain(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_yaml(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _flatten_to_text(data)


def _read_json(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _flatten_to_text(data)


def _read_csv(path: Path) -> str:
    parts = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parts.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return "\n".join(parts)


def _read_xml(path: Path) -> str:
    tree = ET.parse(str(path))
    root = tree.getroot()
    texts = [
        elem.text.strip() for elem in root.iter() if elem.text and elem.text.strip()
    ]
    return "\n".join(texts)


def _read_html(path: Path) -> str:
    content = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _flatten_to_text(obj, depth: int = 0) -> str:
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            parts.append(f"{k}: {_flatten_to_text(v, depth+1)}")
        return "\n".join(parts)
    elif isinstance(obj, list):
        return "\n".join(_flatten_to_text(item, depth + 1) for item in obj)
    else:
        return str(obj)


def read_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md", ".rst"):
        return _read_plain(path)
    elif ext in (".yaml", ".yml"):
        return _read_yaml(path)
    elif ext == ".json":
        return _read_json(path)
    elif ext == ".csv":
        return _read_csv(path)
    elif ext == ".xml":
        return _read_xml(path)
    elif ext in (".html", ".htm"):
        return _read_html(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    elif ext == ".docx":
        return _read_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _split_oversized_paragraph(
    para: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    if len(para) <= chunk_size:
        return [para]
    parts = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(para):
        parts.append(para[start : start + chunk_size])
        start += step
    return parts


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Sliding window chunker with paragraph boundary preference.
    Splits `text` on `\n\n` into paragraphs, then consolidates paragraphs into
    a chunk until adding the next one would exceed `chunk_size` characters.
    Each new chunk starts with trailing paragraphs from the previous chunk,
    keeping their combined length within `chunk_overlap` characters.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Files with only single newlines (no blank lines) become one giant paragraph.
    if len(paragraphs) == 1 and len(paragraphs[0]) > chunk_size:
        paragraphs = [p.strip() for p in paragraphs[0].split("\n") if p.strip()]

    expanded: list[str] = []
    for para in paragraphs:
        expanded.extend(_split_oversized_paragraph(para, chunk_size, chunk_overlap))
    paragraphs = expanded

    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Keep overlap: walk back until we're within overlap budget
            overlap_len = 0
            overlap_paras = []
            for p in reversed(current):
                if overlap_len + len(p) <= chunk_overlap:
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                else:
                    break
            current = overlap_paras
            current_len = overlap_len

        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def scan_and_index(chunk_size: int, chunk_overlap: int) -> None:
    """
    Scans for files with `SUPPORTED_EXTENSIONS` in `SAMPLE_DATA_DIR` and `DOCUMENTS_DIR`.
    For each file, compares its mtime against the value stored in the vector store.
    If the file has not been indexed before, or its mtime differs by 1 second or more,
    the file is re-chunked, re-embedded, and its entries in the vector store are replaced.
    """
    dirs = [SAMPLE_DATA_DIR, DOCUMENTS_DIR]
    for dir_path in dirs:
        if not dir_path.exists():
            continue
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            source = file_path.name
            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                logger.warning("Could not stat '%s', skipping", source, exc_info=True)
                continue

            stored_mtime = vectorstore.get_document_last_modified(source)
            if stored_mtime is not None and abs(stored_mtime - mtime) < 1.0:
                logger.info(f"Skipping '{source}' (unchanged)")
                continue

            logger.info(f"Indexing '{source}'...")
            try:
                text = read_file(file_path)
            except Exception as e:
                logger.warning(f"Failed to read '{source}': {e}")
                continue

            chunks = chunk_text(text, chunk_size, chunk_overlap)
            if not chunks:
                logger.warning(f"No content extracted from '{source}'")
                continue

            try:
                embeddings = embedder.encode(chunks)
                vectorstore.delete_document(source)
                vectorstore.upsert_chunks(
                    chunks=chunks,
                    embeddings=embeddings,
                    source=source,
                    chunk_indices=list(range(len(chunks))),
                    last_modified=mtime,
                )
            except Exception:
                logger.error(
                    "Failed to embed/store '%s', skipping", source, exc_info=True
                )
                continue
            logger.info(f"Indexed '{source}' → {len(chunks)} chunks")
