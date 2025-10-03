import os, re, hashlib
from typing import List, Dict, Tuple
from .settings import settings

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def _md_sections(text: str) -> List[Tuple[str, str]]:
    """
    Improved section splitter that handles nested headers and malformed markdown.
    Returns sections with hierarchical titles and proper content extraction.
    """
    if not text.strip():
        return [("Body", text)]
    
    # Split by any markdown header (1-6 levels)
    parts = re.split(r"\n(?=#{1,6}\s)", text)
    out = []
    
    for p in parts:
        p = p.strip()
        if not p:
            continue
            
        lines = p.splitlines()
        first_line = lines[0] if lines else ""
        
        # Extract header level and title
        header_match = re.match(r"^(#{1,6})\s+(.+)", first_line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            # Create hierarchical title with level indicator
            title = f"{'  ' * (level - 1)}{title}"
        else:
            # Handle content without proper headers
            title = "Body"
            
        # Extract content (remove the header line if it exists)
        if header_match and len(lines) > 1:
            content = "\n".join(lines[1:]).strip()
        else:
            content = p
            
        # Only add non-empty sections
        if content.strip():
            out.append((title, content))
        elif header_match:
            # Keep headers even if they have no content (for structure)
            out.append((title, f"# {header_match.group(2)}"))
    
    return out or [("Body", text)]

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(tokens): break
        i += chunk_size - overlap
    return chunks

def load_documents(data_dir: str) -> List[Dict]:
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        text = _read_text_file(path)
        for section, body in _md_sections(text):
            docs.append({
                "title": fname,
                "section": section,
                "text": body
            })
    return docs

def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
