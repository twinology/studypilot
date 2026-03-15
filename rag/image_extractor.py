"""Extract images from PDF and DOCX files for multimodal RAG."""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from PIL import Image

import config

logger = logging.getLogger("tutor")


@dataclass
class ExtractedImage:
    """Represents an image extracted from a document."""
    file_path: str       # relative path from project root
    page_number: int     # page (PDF) or position index (DOCX)
    image_index: int     # index within that page/section
    width: int
    height: int


def _resize_image(img: Image.Image) -> Image.Image:
    """Resize image so longest side is at most IMAGE_MAX_DIMENSION."""
    max_dim = config.IMAGE_MAX_DIMENSION
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if w >= h:
        new_w = max_dim
        new_h = int(h * max_dim / w)
    else:
        new_h = max_dim
        new_w = int(w * max_dim / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def extract_images_from_pdf(file_path: Path, output_dir: Path) -> List[ExtractedImage]:
    """Extract images from a PDF file using PyMuPDF (fitz)."""
    import fitz

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = fitz.open(str(file_path))
    except Exception as e:
        logger.warning(f"Kan PDF niet openen voor image extractie: {e}")
        return results

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))

                # Convert to RGB if necessary (e.g., CMYK, palette)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGB")

                w, h = img.size

                # Filter out tiny images (icons, bullets, decorations)
                if w < config.IMAGE_MIN_SIZE or h < config.IMAGE_MIN_SIZE:
                    continue

                # Resize for vision API efficiency
                img = _resize_image(img)

                # Save as PNG
                filename = f"page{page_num + 1}_img{img_idx + 1}.png"
                save_path = output_dir / filename
                img.save(str(save_path), "PNG", optimize=True)

                # Store relative path from project root
                rel_path = str(save_path.relative_to(config.BASE_DIR)).replace("\\", "/")

                results.append(ExtractedImage(
                    file_path=rel_path,
                    page_number=page_num + 1,
                    image_index=img_idx + 1,
                    width=img.size[0],
                    height=img.size[1],
                ))
            except Exception as e:
                logger.warning(f"Fout bij extraheren afbeelding p{page_num + 1} img{img_idx + 1}: {e}")
                continue

    doc.close()
    logger.info(f"PDF image extractie: {len(results)} afbeeldingen uit {file_path.name}")
    return results


def extract_images_from_docx(file_path: Path, output_dir: Path) -> List[ExtractedImage]:
    """Extract images from a DOCX file."""
    from docx import Document
    from docx.opc.constants import RELATIONSHIP_TYPE as RT

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        doc = Document(str(file_path))
    except Exception as e:
        logger.warning(f"Kan DOCX niet openen voor image extractie: {e}")
        return results

    img_idx = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue

        try:
            image_bytes = rel.target_part.blob
            img = Image.open(io.BytesIO(image_bytes))

            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            w, h = img.size
            if w < config.IMAGE_MIN_SIZE or h < config.IMAGE_MIN_SIZE:
                continue

            img = _resize_image(img)
            img_idx += 1

            # Determine extension from original or default to png
            filename = f"img{img_idx}.png"
            save_path = output_dir / filename
            img.save(str(save_path), "PNG", optimize=True)

            rel_path = str(save_path.relative_to(config.BASE_DIR)).replace("\\", "/")

            results.append(ExtractedImage(
                file_path=rel_path,
                page_number=0,  # DOCX doesn't have page numbers easily
                image_index=img_idx,
                width=img.size[0],
                height=img.size[1],
            ))
        except Exception as e:
            logger.warning(f"Fout bij extraheren DOCX afbeelding {img_idx}: {e}")
            continue

    logger.info(f"DOCX image extractie: {len(results)} afbeeldingen uit {file_path.name}")
    return results


def extract_images_from_html(html_content: str, base_url: str, output_dir: Path) -> List[ExtractedImage]:
    """Extract images from HTML content by downloading <img> src URLs."""
    from urllib.parse import urljoin, urlparse
    import urllib.request
    import hashlib

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse HTML to find img tags
    from html.parser import HTMLParser

    img_urls = []

    class ImgParser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "img":
                attrs_dict = dict(attrs)
                src = attrs_dict.get("src", "")
                if src:
                    img_urls.append(src)

    try:
        ImgParser().feed(html_content)
    except Exception as e:
        logger.warning(f"HTML parsing fout bij image extractie: {e}")
        return results

    # Deduplicate
    seen = set()
    unique_urls = []
    for src in img_urls:
        full_url = urljoin(base_url, src)
        if full_url not in seen:
            seen.add(full_url)
            unique_urls.append(full_url)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    img_idx = 0
    for img_url in unique_urls:
        try:
            # Skip data URIs, SVGs, tiny icons
            parsed = urlparse(img_url)
            if not parsed.scheme.startswith("http"):
                continue
            lower_path = parsed.path.lower()
            if lower_path.endswith(".svg") or lower_path.endswith(".gif") or lower_path.endswith(".ico"):
                continue

            req = urllib.request.Request(img_url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                image_bytes = resp.read()

            img = Image.open(io.BytesIO(image_bytes))

            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            w, h = img.size
            if w < config.IMAGE_MIN_SIZE or h < config.IMAGE_MIN_SIZE:
                continue

            img = _resize_image(img)
            img_idx += 1

            # Use hash of URL for unique filename
            url_hash = hashlib.md5(img_url.encode()).hexdigest()[:8]
            filename = f"web_img{img_idx}_{url_hash}.png"
            save_path = output_dir / filename
            img.save(str(save_path), "PNG", optimize=True)

            rel_path = str(save_path.relative_to(config.BASE_DIR)).replace("\\", "/")

            results.append(ExtractedImage(
                file_path=rel_path,
                page_number=0,
                image_index=img_idx,
                width=img.size[0],
                height=img.size[1],
            ))
        except Exception as e:
            logger.debug(f"URL afbeelding overgeslagen ({img_url}): {e}")
            continue

    logger.info(f"HTML image extractie: {len(results)} afbeeldingen van {base_url}")
    return results


def extract_document_images(file_path: Path, doc_name: str) -> List[ExtractedImage]:
    """Extract images from a document (PDF or DOCX). Returns empty list for unsupported types."""
    if not config.EXTRACT_IMAGES:
        return []

    suffix = file_path.suffix.lower()
    stem = Path(doc_name).stem
    output_dir = config.IMAGES_DIR / stem

    if suffix == ".pdf":
        return extract_images_from_pdf(file_path, output_dir)
    elif suffix in (".docx", ".doc"):
        return extract_images_from_docx(file_path, output_dir)
    else:
        return []
