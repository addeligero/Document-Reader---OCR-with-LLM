from __future__ import annotations

from pathlib import Path

from services.document_processor import process_file_bytes


def run_ocr(file_path: Path, display_name: str) -> dict:
    return process_file_bytes(display_name, file_path.read_bytes())


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).lower()
