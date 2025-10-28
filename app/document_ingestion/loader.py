# app/vector_store/loader.py
import os
import asyncio
import docx2txt
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any

# Thread pool for blocking I/O
executor = ThreadPoolExecutor(max_workers=5)


def clean_text(text: str) -> str:
    """Removes null bytes and unwanted control characters."""
    return text.replace("\x00", "").strip()

def load_pdf(file_path: str) -> Dict[str, Any]:
    """Blocking PDF loader."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return {
        "file_path": file_path,
        "content": clean_text(text),
        "source": os.path.basename(file_path)
    }

def load_docx(file_path: str) -> Dict[str, Any]:
    """Blocking DOCX loader."""
    text = docx2txt.process(file_path) or ""
    return {
        "file_path": file_path,
        "content": clean_text(text),
        "source": os.path.basename(file_path)
    }


async def load_file(file_path: str) -> Dict[str, Any]:
    """Async wrapper for file loading."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return await asyncio.get_event_loop().run_in_executor(executor, load_pdf, file_path)
    elif ext == ".docx":
        return await asyncio.get_event_loop().run_in_executor(executor, load_docx, file_path)
    else:
        return {"file_path": file_path, "content": "", "source": os.path.basename(file_path)}


async def load_documents_from_dir(directory: str) -> List[Dict[str, Any]]:
    """Loads all supported documents from directory asynchronously."""
    tasks = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".pdf", ".docx")):
                file_path = os.path.join(root, file)
                tasks.append(load_file(file_path))

    results = await asyncio.gather(*tasks)
    return [doc for doc in results if doc["content"]]  # Filter out empty docs