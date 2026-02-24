# ===============================
# 📌 Flask OCR API (PDF + DOCX + Images) — FIXED for scanned tables
# - Fixes PDF loop bug (OCR runs per page)
# - Adds deskew (reduces scan rotation errors)
# - Removes table lines before line detection (prevents giant boxes)
# - Switches to connected-components line grouping (more robust than projection for tables)
# - Adds hard filters to reject absurd regions
# ===============================

import os
import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from flask_cors import CORS
from docx import Document
from services.ocr_process import ocr_image
from services.llm_process import classify_document

# ===============================
# Initialize Flask
# ===============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Poppler (Windows)
# ===============================
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

# ===============================
# Allowed file types
# ===============================
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# Main Upload Endpoint
# ===============================

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename

    if not filename or not allowed_file(filename):
        return jsonify({"error": "File type not allowed"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    extracted_text = ""
    ext = filename.lower().rsplit(".", 1)[1]

    # ===============================
    # PDF
    # ===============================
    if ext == "pdf":
        try:
            final_text_parts = []

            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1

                    # 1) Try text-based extraction
                    text = (page.extract_text() or "").strip()

                    if text:
                        final_text_parts.append(
                            f"--- Page {page_num}/{total_pages} (text) ---\n{text}\n"
                        )
                        continue

                    # 2) Lazy rendering: render ONLY this page (because it needs OCR)
                    img = convert_from_path(
                        filepath,
                        dpi=400,
                        first_page=page_num,
                        last_page=page_num,
                        poppler_path=POPPLER_PATH
                    )[0]

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    img_np = np.array(img)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    ocr_txt = ocr_image(
                        img_bgr,
                        f"{filename}_page{page_num}",
                        is_pdf=True
                    ).strip()

                    final_text_parts.append(
                        f"--- Page {page_num}/{total_pages} (ocr) ---\n{ocr_txt}\n"
                    )

            full_text = "\n".join(final_text_parts)
            llm_result = classify_document(full_text)

            return jsonify({
                "type": "hybrid-pdf-lazy",
                "text": full_text,
                "primary_category": llm_result.get("primary_category"),
                "secondary_category": llm_result.get("secondary_category"),
                "tags": llm_result.get("tags", []),
            })

        except Exception as e:
            print("PDF processing error:", e)
            return jsonify({"error": "Failed to process PDF"}), 500

    # ===============================
    # DOCX
    # ===============================
    elif ext == "docx":
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"

            llm_result = classify_document(extracted_text)

            return jsonify({
                "type": "docx",
                "text": extracted_text,
                "primary_category": llm_result.get("primary_category"),
                "secondary_category": llm_result.get("secondary_category"),
                "tags": llm_result.get("tags", []),
            })

        except Exception as e:
            print("DOCX error:", e)
            return jsonify({"error": "Failed to process DOCX"}), 500

    # ===============================
    # Images
    # ===============================
    else:
        try:
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({"error": "Invalid image file"}), 400

            extracted_text = ocr_image(img, filename, is_pdf=False)
            llm_result = classify_document(extracted_text)

            return jsonify({
                "type": "image",
                "text": extracted_text,
                "primary_category": llm_result.get("primary_category"),
                "secondary_category": llm_result.get("secondary_category"),
                "tags": llm_result.get("tags", []),
            })

        except Exception as e:
            print("Image OCR error:", e)
            return jsonify({"error": "Failed to process image"}), 500


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
