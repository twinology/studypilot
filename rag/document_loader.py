"""Load documents from various formats into structured text with heading markers."""

import re
from pathlib import Path
from typing import Optional

import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

# Heading marker format used to track section structure
HEADING_MARKER = "[SECTIE: {}]"


def load_document(file_path: Path) -> Optional[str]:
    """Load a document and return its text content with heading markers."""
    suffix = file_path.suffix.lower()
    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".doc": _load_docx,
        ".txt": _load_text,
        ".md": _load_markdown,
        ".html": _load_html,
        ".htm": _load_html,
    }
    loader = loaders.get(suffix)
    if not loader:
        raise ValueError(f"Niet-ondersteund bestandsformaat: {suffix}")
    return loader(file_path)


def _load_pdf(file_path: Path) -> str:
    """Load PDF - detect headings by heuristics (numbered sections)."""
    text_parts = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    raw_text = "\n\n".join(text_parts)

    # Try to detect headings: numbered sections like "1.2 Title"
    lines = raw_text.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result_lines.append("")
            continue
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', stripped) and len(stripped) < 100:
            result_lines.append(HEADING_MARKER.format(stripped))
        else:
            result_lines.append(stripped)
    return "\n".join(result_lines)


def _load_docx(file_path: Path) -> str:
    """Load DOCX with heading style detection."""
    doc = docx.Document(str(file_path))
    parts = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style_name = p.style.name if p.style else ""
        if style_name.startswith("Heading") or style_name.startswith("Title"):
            parts.append(HEADING_MARKER.format(text))
        else:
            parts.append(text)
    return "\n\n".join(parts)


def _load_text(file_path: Path) -> str:
    """Load plain text - detect headings by heuristics."""
    raw_text = file_path.read_text(encoding="utf-8")
    lines = raw_text.split("\n")
    result_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result_lines.append("")
            continue
        if re.match(r'^\d+(\.\d+)*\s+[A-Z]', stripped) and len(stripped) < 100:
            result_lines.append(HEADING_MARKER.format(stripped))
        else:
            result_lines.append(stripped)
    return "\n".join(result_lines)


def _load_markdown(file_path: Path) -> str:
    """Load Markdown - convert # headings to markers."""
    md_text = file_path.read_text(encoding="utf-8")
    lines = md_text.split("\n")
    result_lines = []
    for line in lines:
        heading_match = re.match(r'^#{1,6}\s+(.+)', line)
        if heading_match:
            result_lines.append(HEADING_MARKER.format(heading_match.group(1).strip()))
        else:
            result_lines.append(line)
    text = "\n".join(result_lines)
    # Preserve our markers, convert the rest
    parts = re.split(r'(\[SECTIE: [^\]]+\])', text)
    final_parts = []
    for part in parts:
        if part.startswith("[SECTIE:"):
            final_parts.append(part)
        else:
            html = markdown.markdown(part)
            final_parts.append(BeautifulSoup(html, "html.parser").get_text(separator="\n"))
    return "\n".join(final_parts)


def _load_html(file_path: Path) -> str:
    """Load HTML - convert h1-h6 tags to markers."""
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading_text = heading.get_text(strip=True)
            heading.replace_with(f"\n{HEADING_MARKER.format(heading_text)}\n")
    return soup.get_text(separator="\n", strip=True)
