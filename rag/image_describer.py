"""Generate text descriptions of extracted images using the vision API."""

import logging
from pathlib import Path
from typing import List, Tuple

import config
from ai_provider import create_vision_completion
from rag.image_extractor import ExtractedImage

logger = logging.getLogger("tutor")

IMAGE_DESCRIPTION_PROMPT = (
    "Beschrijf deze afbeelding gedetailleerd in het Nederlands. "
    "Focus op: wat er te zien is, eventuele tekst in de afbeelding, "
    "data of getallen in tabellen/grafieken, en de educatieve betekenis. "
    "Geef een beschrijving van 2-5 zinnen."
)


def describe_images(
    images: List[ExtractedImage],
    doc_name: str,
) -> Tuple[List[str], List[dict]]:
    """Generate text descriptions for a list of extracted images.

    Returns:
        Tuple of (chunks, extra_metadatas) where:
        - chunks: list of description texts to be embedded
        - extra_metadatas: list of metadata dicts with content_type and image_path
    """
    chunks: List[str] = []
    metadatas: List[dict] = []

    for img in images:
        image_path = config.BASE_DIR / img.file_path
        if not image_path.exists():
            logger.warning(f"Afbeelding niet gevonden: {img.file_path}")
            continue

        try:
            image_bytes = image_path.read_bytes()

            # Determine MIME type
            suffix = image_path.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
            mime_type = mime_map.get(suffix, "image/png")

            description = create_vision_completion(
                prompt=IMAGE_DESCRIPTION_PROMPT,
                image_bytes=image_bytes,
                mime_type=mime_type,
                max_tokens=512,
            )

            page_info = f"pagina {img.page_number}" if img.page_number > 0 else f"positie {img.image_index}"
            chunk_text = f"[Afbeelding uit {doc_name}, {page_info}] {description}"

            chunks.append(chunk_text)
            metadatas.append({
                "content_type": "image_description",
                "image_path": img.file_path,
            })

            logger.info(f"Afbeelding beschreven: {img.file_path} ({len(description)} tekens)")

        except Exception as e:
            logger.warning(f"Kon afbeelding niet beschrijven ({img.file_path}): {e}")
            continue

    logger.info(f"Image describer: {len(chunks)}/{len(images)} afbeeldingen beschreven voor {doc_name}")
    return chunks, metadatas


def describe_single_image(img: ExtractedImage, doc_name: str) -> Tuple[str, dict, dict]:
    """Describe a single image. Returns (chunk_text, metadata, usage) or (None, None, None) on failure."""
    image_path = config.BASE_DIR / img.file_path
    if not image_path.exists():
        logger.warning(f"Afbeelding niet gevonden: {img.file_path}")
        return None, None, None

    try:
        image_bytes = image_path.read_bytes()
        suffix = image_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
        mime_type = mime_map.get(suffix, "image/png")

        result = create_vision_completion(
            prompt=IMAGE_DESCRIPTION_PROMPT,
            image_bytes=image_bytes,
            mime_type=mime_type,
            max_tokens=512,
            return_usage=True,
        )
        description = result["text"]
        usage = result["usage"]

        page_info = f"pagina {img.page_number}" if img.page_number > 0 else f"positie {img.image_index}"
        chunk_text = f"[Afbeelding uit {doc_name}, {page_info}] {description}"

        logger.info(f"Afbeelding beschreven: {img.file_path} ({len(description)} tekens)")

        return chunk_text, {"content_type": "image_description", "image_path": img.file_path}, usage
    except Exception as e:
        logger.warning(f"Kon afbeelding niet beschrijven ({img.file_path}): {e}")
        return None, None, None
