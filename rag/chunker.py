"""Split text into overlapping chunks with section context for embedding."""

import re
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Matches heading markers inserted by document_loader
_HEADING_PATTERN = re.compile(r'^\[SECTIE:\s*(.+?)\]$')


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, each prefixed with its section heading.

    This preserves document structure so that each chunk knows which section
    it belongs to, dramatically improving retrieval accuracy.
    """
    segments = _split_into_segments(text)
    chunks = []
    current_heading = ""
    current_chunk: list[str] = []
    current_length = 0

    for segment in segments:
        # Check if this segment is a heading marker
        heading_match = _HEADING_PATTERN.match(segment.strip())
        if heading_match:
            current_heading = heading_match.group(1)
            continue

        segment_length = len(segment)

        # If adding this segment exceeds chunk_size, finalize current chunk
        if current_length + segment_length > chunk_size and current_chunk:
            chunk_str = _build_chunk(current_heading, current_chunk)
            chunks.append(chunk_str)

            # Keep overlap by retaining last segments
            overlap_chunk: list[str] = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s)
            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(segment)
        current_length += segment_length

    # Don't forget the last chunk
    if current_chunk:
        chunk_str = _build_chunk(current_heading, current_chunk)
        chunks.append(chunk_str)

    return chunks


def _build_chunk(heading: str, segments: list[str]) -> str:
    """Build a chunk string, prefixing with heading context if available."""
    body = " ".join(segments)
    if heading:
        return f"[{heading}] {body}"
    return body


def _split_into_segments(text: str) -> List[str]:
    """Split text into segments: headings and sentences."""
    # First split on double newlines (paragraph breaks)
    paragraphs = text.split("\n\n")
    segments = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if this is a heading marker (preserve as-is)
        if _HEADING_PATTERN.match(para):
            segments.append(para)
            continue

        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sentence in sentences:
            # Also handle single newlines within paragraphs
            sub_parts = sentence.split("\n")
            for sp in sub_parts:
                sp = sp.strip()
                if sp:
                    segments.append(sp)

    return segments
