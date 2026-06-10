"""
file_handler.py
Extracts plain text from: PDF, DOCX, TXT, and 20+ source-code formats.
"""

import io

CODE_EXTENSIONS = {
    "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "cs", "go",
    "rs", "rb", "php", "swift", "kt", "scala", "r", "sh", "bash",
    "html", "css", "json", "xml", "yaml", "yml", "toml", "sql",
    "md", "ipynb", "lua", "pl", "dart", "vue",
}


def extract_from_pdf(file_bytes: bytes) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {e}")


def extract_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        raise ValueError(f"DOCX extraction failed: {e}")


def extract_from_text(file_bytes: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode file — unsupported encoding.")


def process_file(filename: str, file_bytes: bytes) -> tuple[str, str]:
    """Returns (text, file_type_label). Raises ValueError on failure."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return extract_from_pdf(file_bytes), "PDF Document"
    elif ext == "docx":
        return extract_from_docx(file_bytes), "Word Document"
    elif ext == "txt":
        return extract_from_text(file_bytes), "Plain Text"
    elif ext in CODE_EXTENSIONS:
        return extract_from_text(file_bytes), f"Source Code (.{ext})"
    else:
        try:
            return extract_from_text(file_bytes), f"Text File (.{ext})"
        except Exception:
            raise ValueError(
                f"Unsupported file type: .{ext}\n"
                f"Supported: pdf, docx, txt, and code files ({', '.join(sorted(CODE_EXTENSIONS))})"
            )