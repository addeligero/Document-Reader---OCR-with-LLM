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
from datetime import datetime

# ===============================
# Initialize Flask
# ===============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OCR_PROCESS_FOLDER = "user-ocr-process"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OCR_PROCESS_FOLDER, exist_ok=True)

# ===============================
# Poppler (Windows)
# ===============================
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

# ===============================
# Allowed file types
# ===============================
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "docx"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ===============================
# Load TrOCR Model (Load once)
# ===============================
device = torch.device("cpu")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")

model.to(device)
model.eval()

# ===============================
# Line-by-Line OCR Function
# ===============================
def detect_text_lines(cropped):
    """
    Detect text line bounding boxes using adaptive threshold + horizontal projection.
    Works reliably on both clean PDF-rendered images and noisy scanned/photo images.
    """
    h, w = cropped.shape

    blurred = cv2.GaussianBlur(cropped, (5, 5), 0)

    # Adaptive threshold handles varying image conditions (clean PDFs, noisy scans)
    block_size = max(15, (min(h, w) // 100) | 1)
    if block_size % 2 == 0:
        block_size += 1
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 10
    )

    # Remove long vertical/horizontal ruling lines (common in scanned table PDFs)
    # so bounding boxes are formed from text components rather than table borders.
    line_scale = max(20, min(h, w) // 30)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_scale))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_scale, 1))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    text_only = cv2.subtract(thresh, cv2.bitwise_or(vertical_lines, horizontal_lines))

    # Connect letters inside each row without aggressively merging nearby rows.
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 140), 1))
    text_for_projection = cv2.morphologyEx(text_only, cv2.MORPH_CLOSE, connect_kernel)

    # Horizontal projection: count white pixels per row
    proj = np.sum(text_for_projection, axis=1) / 255

    # Row has text if white pixel count exceeds threshold
    text_threshold = w * 0.003
    text_rows = proj > text_threshold

    # Merge small vertical gaps between lines (e.g. space between baseline and next ascender)
    gap_merge = max(2, h // 350)
    text_mask = text_rows.astype(np.uint8)
    merge_kernel = np.ones(gap_merge, np.uint8)
    text_mask = cv2.dilate(text_mask.reshape(-1, 1), merge_kernel).flatten()

    # Find start/end of each text region
    transitions = np.diff(text_mask.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    if text_mask[0]:
        starts = np.insert(starts, 0, 0)
    if text_mask[-1]:
        ends = np.append(ends, len(text_mask))

    boxes = []
    for s, e in zip(starts, ends):
        line_height = e - s
        if line_height < h * 0.005:
            continue

        # Find horizontal extent of text in this row
        line_strip = text_for_projection[s:e, :]
        col_proj = np.sum(line_strip, axis=0) / 255
        cols_with_text = np.where(col_proj > 0)[0]
        if len(cols_with_text) == 0:
            continue

        x = int(cols_with_text[0])
        x_end = int(cols_with_text[-1])
        box_w = x_end - x + 1
        if box_w < w * 0.02:
            continue

        # Add small padding around the detected region
        pad_y = max(2, line_height // 10)
        pad_x = max(2, box_w // 50)
        y1 = max(0, s - pad_y)
        y2 = min(h, e + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(w, x_end + pad_x)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes, thresh, text_for_projection


def ocr_image(img, filename="image", is_pdf=False):


    # 🔥 Resize large images ONLY if not PDF
    if not is_pdf and img.shape[0] > 2000:
        scale = 2000 / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    process_name = f"{filename}_{timestamp}"
    process_folder = os.path.join(OCR_PROCESS_FOLDER, process_name)
    os.makedirs(process_folder, exist_ok=True)

    cv2.imwrite(os.path.join(process_folder, "01_original.png"), img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(process_folder, "02_grayscale.png"), gray)

    #  crop borders
    h, w = gray.shape
    if is_pdf:
     cropped = gray  # no border crop for PDF
    else:
     cropped = gray[30:h-30, 30:w-30]

    cv2.imwrite(os.path.join(process_folder, "03_cropped.png"), cropped)

    # Detect text lines using adaptive threshold + horizontal projection
    boxes, thresh, text_projection = detect_text_lines(cropped)
    cv2.imwrite(os.path.join(process_folder, "04_threshold.png"), thresh)
    cv2.imwrite(os.path.join(process_folder, "05_text_projection.png"), text_projection)

    bbox_img = cv2.cvtColor(cropped.copy(), cv2.COLOR_GRAY2BGR)

    extracted_lines = []
    line_details = []

    for idx, (x, y, w, h) in enumerate(boxes):

        cv2.rectangle(bbox_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        line_img = cropped[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(process_folder, f"line_{idx:03d}.png"), line_img)

        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=512,
                num_beams=4,
                early_stopping=True
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        extracted_lines.append(text)
        line_details.append(f"Line {idx}: [{x}, {y}, {w}, {h}] -> {text}")

    cv2.imwrite(os.path.join(process_folder, "07_bounding_boxes.png"), bbox_img)

    with open(os.path.join(process_folder, "ocr_results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(line_details))
        f.write("\n\nFinal Extracted Text:\n")
        f.write("\n".join(extracted_lines))

    return "\n".join(extracted_lines)

# ===============================
# Main Upload Endpoint
# ===============================
@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename

    if not allowed_file(filename):
        return jsonify({"error": "File type not allowed"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    extracted_text = ""
    ext = filename.lower().rsplit(".", 1)[1]

    # ===============================
    # PDF
    # ===============================
    if ext == "pdf":

        # 1️⃣ Try text-based extraction first
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"

            if extracted_text.strip():
                return jsonify({
                    "type": "text-based-pdf",
                    "text": extracted_text
                })

        except Exception as e:
            print("pdfplumber error:", e)

        # 2️⃣ Scanned PDF → Convert page to image → Line-by-line OCR
        try:
            pages = convert_from_path(
                filepath,
                dpi=400,
                poppler_path=POPPLER_PATH
            )

            for page_num, page in enumerate(pages, 1):
                # Ensure RGB mode (handles RGBA, grayscale, palette PDFs)
                if page.mode != "RGB":
                    page = page.convert("RGB")
                img = np.array(page)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                extracted_text += ocr_image(
                    img,
                    f"{filename}_page{page_num}",
                    is_pdf=True
                ) + "\n"

            return jsonify({
                "type": "scanned-pdf",
                "text": extracted_text
            })

        except Exception as e:
            print("PDF OCR error:", e)
            return jsonify({"error": "Failed to process scanned PDF"}), 500

    # ===============================
    # DOCX
    # ===============================
    elif ext == "docx":
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                extracted_text += para.text + "\n"

            return jsonify({
                "type": "docx",
                "text": extracted_text
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

            extracted_text = ocr_image(img, filename)

            return jsonify({
                "type": "image",
                "text": extracted_text
            })

        except Exception as e:
            print("Image OCR error:", e)
            return jsonify({"error": "Failed to process image"}), 500

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=False)
