
# ===============================
# 📌 Flask OCR API (PDF + DOCX + Images)
# ===============================

import os
import cv2
import numpy as np
import pdfplumber
from pdf2image import convert_from_path
from flask import Flask, request, jsonify
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from flask_cors import CORS
from docx import Document
import torch

# ===============================
# Initialize Flask
# ===============================
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Allowed file types
# ===============================
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================
# Load TrOCR Model (Load once!)
# ===============================
device = torch.device("cpu")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

model.to(device)
model.eval()
# ===============================
# OCR Preprocessing Function
# ===============================
def ocr_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cropped = gray[50:h-50, 50:w-50]

    _, thresh = cv2.threshold(
        cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    dilated = cv2.dilate(closed, kernel_d, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])

    extracted_lines = []

    for (x, y, w, h) in boxes:
        if w < 40 or h < 15:
            continue
        line_img = cropped[y:y+h, x:x+w]
        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
          generated_ids = model.generate(pixel_values)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        extracted_lines.append(text)

    return "\n".join(extracted_lines)

# ===============================
# 📌 Main API Endpoint
# ===============================
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename

    if not allowed_file(filename):
        return jsonify({"error": f"File type not allowed: {filename}"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    extracted_text = ""

    ext = filename.lower().rsplit(".", 1)[1]

    # ===============================
    # PDF
    # ===============================
    if ext == "pdf":
        # Try text extraction first
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if extracted_text.strip():  # text-based PDF
            return jsonify({
                "type": "text-based-pdf",
                "text": extracted_text
            })

        # scanned PDF → OCR
        pages = convert_from_path(filepath, dpi=300)
        for page in pages:
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            extracted_text += ocr_image(img) + "\n"

        return jsonify({
            "type": "scanned-pdf",
            "text": extracted_text
        })

    # ===============================
    # DOCX
    # ===============================
    elif ext == "docx":
        doc = Document(filepath)
        for para in doc.paragraphs:
            extracted_text += para.text + "\n"

        return jsonify({
            "type": "docx",
            "text": extracted_text
        })

    # ===============================
    # Images
    # ===============================
    else:  # jpg, jpeg, png
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400

        extracted_text = ocr_image(img)
        return jsonify({
            "type": "image",
            "text": extracted_text
        })

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
