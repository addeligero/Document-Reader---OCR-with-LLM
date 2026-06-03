from __future__ import annotations

import io
import os
from pathlib import Path

import cv2
import numpy as np
import pdfplumber
from docx import Document
from pdf2image import convert_from_bytes


ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx"}

if os.name == "nt":
    POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"
else:
    POPPLER_PATH = os.getenv("POPPLER_PATH")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def classify_text(text: str) -> dict:
    from services.classifier_process import svm_classify
    from services.llm_process import classify_document

    svm_candidates = svm_classify(text)
    return classify_document(text, svm_candidates)


def process_file_bytes(filename: str, uploaded_bytes: bytes) -> dict:
    if not filename or not allowed_file(filename):
        raise ValueError("File type not allowed")
    if not uploaded_bytes:
        raise ValueError("Uploaded file is empty")

    extension = Path(filename).suffix.lower().lstrip(".")

    if extension == "pdf":
        extracted_text = _extract_pdf_text(filename, uploaded_bytes)
        result_type = "pdf"
    elif extension == "docx":
        extracted_text = _extract_docx_text(uploaded_bytes)
        result_type = "docx"
    else:
        extracted_text = _extract_image_text(filename, uploaded_bytes)
        result_type = "image"

    llm_result = classify_text(extracted_text)
    return {
        "filename": filename,
        "type": result_type,
        "text": extracted_text,
        "primary_category": llm_result.get("primary_category"),
        "secondary_category": llm_result.get("secondary_category"),
        "tags": llm_result.get("tags", []),
    }


def _extract_pdf_text(filename: str, uploaded_bytes: bytes) -> str:
    from services.ocr_process import ocr_image

    final_text_parts = []

    with pdfplumber.open(io.BytesIO(uploaded_bytes)) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            text = (page.extract_text() or "").strip()

            if text:
                final_text_parts.append(f"--- Page {page_num}/{total_pages} (text) ---\n{text}\n")
                continue

            poppler_kwargs = {"poppler_path": POPPLER_PATH} if POPPLER_PATH else {}
            img = convert_from_bytes(
                uploaded_bytes,
                dpi=400,
                first_page=page_num,
                last_page=page_num,
                **poppler_kwargs,
            )[0]

            if img.mode != "RGB":
                img = img.convert("RGB")

            img_np = np.array(img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            ocr_txt = ocr_image(img_bgr, f"{filename}_page{page_num}", is_pdf=True).strip()
            final_text_parts.append(f"--- Page {page_num}/{total_pages} (ocr) ---\n{ocr_txt}\n")

    return "\n".join(final_text_parts)


def _extract_docx_text(uploaded_bytes: bytes) -> str:
    doc = Document(io.BytesIO(uploaded_bytes))
    return "\n".join(para.text for para in doc.paragraphs)


def _extract_image_text(filename: str, uploaded_bytes: bytes) -> str:
    from services.ocr_process import ocr_image

    img_arr = np.frombuffer(uploaded_bytes, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file")
    return ocr_image(img, filename, is_pdf=False)
