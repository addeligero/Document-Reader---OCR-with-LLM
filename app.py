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


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ===============================
# Load TrOCR Model (Load once)
# ===============================
device = torch.device("cpu")  # change to "cuda" if you want & have GPU
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.to(device)
model.eval()


# ===============================
# Image Preprocessing Helpers
# ===============================
def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Deskew an image using minimum-area rectangle of foreground pixels.
    Works well for small rotation in scans/photos.
    """
    if gray is None or gray.size == 0:
        return gray

    # Otsu binarization for angle estimate
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 1500:
        return gray

    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in [-90, 0)
    angle = -(90 + angle) if angle < -45 else -angle

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def make_binary_inv(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive threshold (binary inverse): foreground is white (255), background is black (0).
    More robust for uneven lighting than plain Otsu for scans/photos.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # adaptive block size scales with image size (must be odd)
    block_size = max(31, (min(h, w) // 40) | 1)
    if block_size % 2 == 0:
        block_size += 1

    bin_inv = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        10,
    )
    return bin_inv


def remove_table_lines(bin_inv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove horizontal/vertical table lines using morphological opening.
    Returns:
      - text_only: bin_inv with lines removed
      - lines: detected line mask
    """
    h, w = bin_inv.shape

    # Kernels tuned for typical tables; scale with image size
    hor_len = max(30, w // 35)
    ver_len = max(30, h // 35)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))

    hor = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    ver = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, ver_kernel, iterations=1)

    lines = cv2.bitwise_or(hor, ver)
    text_only = cv2.bitwise_and(bin_inv, cv2.bitwise_not(lines))
    return text_only, lines


# ===============================
# Robust Line Detection (Connected Components + Y grouping)
# ===============================
def detect_text_lines_cc(text_only: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect text lines in a table-like document:
    - Slight dilation to connect characters into words
    - Find connected components
    - Filter small/noisy components
    - Group components by y-center into lines
    - Return line bounding boxes (x, y, w, h)
    """
    h, w = text_only.shape

    # Connect nearby letters; keep it light
    kx = max(2, w // 250)
    ky = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dil = cv2.dilate(text_only, kernel, iterations=1)

    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    comps = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        # noise filters (tune if needed)
        if bw < max(10, int(0.01 * w)):
            continue
        if bh < max(8, int(0.006 * h)):
            continue
        if bw * bh < int(0.00003 * w * h):
            continue
        if bw > int(0.95 * w) and bh > int(0.10 * h):
            # reject absurd huge region
            continue

        cy = y + bh / 2.0
        comps.append((cy, x, y, bw, bh))

    if not comps:
        return []

    comps.sort(key=lambda t: t[0])  # sort by center y

    # Group by y-center proximity (line band)
    line_band = max(10, h // 120)  # tune for your scans
    lines = []
    current = [comps[0]]
    for item in comps[1:]:
        if abs(item[0] - current[-1][0]) <= line_band:
            current.append(item)
        else:
            lines.append(current)
            current = [item]
    lines.append(current)

    boxes = []
    for group in lines:
        xs = [g[1] for g in group]
        ys = [g[2] for g in group]
        x2s = [g[1] + g[3] for g in group]
        y2s = [g[2] + g[4] for g in group]

        x1 = max(0, min(xs) - 2)
        y1 = max(0, min(ys) - 2)
        x2 = min(w, max(x2s) + 2)
        y2 = min(h, max(y2s) + 2)

        bw = x2 - x1
        bh = y2 - y1

        # Final sanity filters
        if bw < int(0.05 * w):
            continue
        if bh > int(0.20 * h):
            continue
        if bw * bh > int(0.30 * w * h):
            continue

        boxes.append((x1, y1, bw, bh))

    # Sort top-to-bottom, then left-to-right (stable)
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


# ===============================
# OCR Function
# ===============================
def ocr_image(img_bgr: np.ndarray, filename="image", is_pdf=False) -> str:
    """
    OCR an image:
    - grayscale + deskew
    - (optional) crop borders for photos
    - binarize + remove table lines
    - detect text lines (connected component grouping)
    - TrOCR line-by-line
    """

    # Resize very large non-PDF images (keep PDFs at DPI quality)
    if not is_pdf and img_bgr.shape[0] > 2200:
        scale = 2200 / img_bgr.shape[0]
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    process_name = f"{filename}_{timestamp}"
    process_folder = os.path.join(OCR_PROCESS_FOLDER, process_name)
    os.makedirs(process_folder, exist_ok=True)

    cv2.imwrite(os.path.join(process_folder, "01_original.png"), img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    cv2.imwrite(os.path.join(process_folder, "02_grayscale_deskew.png"), gray)

    # Light border crop for photos (keeps PDF pages intact)
    h, w = gray.shape
    if is_pdf:
        cropped = gray
    else:
        pad = 25
        cropped = gray[pad : max(pad + 1, h - pad), pad : max(pad + 1, w - pad)]

    cv2.imwrite(os.path.join(process_folder, "03_cropped.png"), cropped)

    bin_inv = make_binary_inv(cropped)
    cv2.imwrite(os.path.join(process_folder, "04_threshold_bin_inv.png"), bin_inv)

    text_only, lines_mask = remove_table_lines(bin_inv)
    cv2.imwrite(os.path.join(process_folder, "05_lines_mask.png"), lines_mask)
    cv2.imwrite(os.path.join(process_folder, "06_text_only.png"), text_only)

    # Detect line boxes
    boxes = detect_text_lines_cc(text_only)

    bbox_img = cv2.cvtColor(cropped.copy(), cv2.COLOR_GRAY2BGR)

    extracted_lines = []
    line_details = []

    # If no lines detected, fallback: OCR the whole page (last resort)
    if not boxes:
        boxes = [(0, 0, cropped.shape[1], cropped.shape[0])]

    for idx, (x, y, bw, bh) in enumerate(boxes):
        cv2.rectangle(bbox_img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        line_img = cropped[y : y + bh, x : x + bw]

        # Skip nearly blank crops
        # (count dark pixels; adjust threshold for your scans)
        dark_ratio = (line_img < 200).mean()
        if dark_ratio < 0.01:
            continue

        # Improve readability for TrOCR
        line_img = cv2.resize(line_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        line_img = cv2.GaussianBlur(line_img, (3, 3), 0)

        cv2.imwrite(os.path.join(process_folder, f"line_{idx:03d}.png"), line_img)

        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)

        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Basic cleanup: ignore extremely short garbage tokens
        if len(text) <= 1:
            continue

        extracted_lines.append(text)
        line_details.append(f"Line {idx}: [{x}, {y}, {bw}, {bh}] -> {text}")

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

            # Open PDF once
            with pdfplumber.open(filepath) as pdf:
                total_pages = len(pdf.pages)

                # Render all pages once (only used when a page needs OCR)
                # If you want faster: we can render lazily per missing page
                rendered_pages = convert_from_path(
                    filepath,
                    dpi=400,
                    poppler_path=POPPLER_PATH
                )

                for i, page in enumerate(pdf.pages):
                    page_num = i + 1

                    # 1) Try text-based extraction
                    text = page.extract_text() or ""
                    text = text.strip()

                    if text:
                        final_text_parts.append(f"--- Page {page_num}/{total_pages} (text) ---\n{text}\n")
                        continue

                    # 2) If no text, do OCR on the rendered image page
                    img = rendered_pages[i]
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img_np = np.array(img)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    ocr_txt = ocr_image(img_bgr, f"{filename}_page{page_num}", is_pdf=True).strip()
                    final_text_parts.append(f"--- Page {page_num}/{total_pages} (ocr) ---\n{ocr_txt}\n")

            return jsonify({
                "type": "hybrid-pdf",
                "text": "\n".join(final_text_parts)
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

            return jsonify({"type": "docx", "text": extracted_text})

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

            return jsonify({"type": "image", "text": extracted_text})

        except Exception as e:
            print("Image OCR error:", e)
            return jsonify({"error": "Failed to process image"}), 500


# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
