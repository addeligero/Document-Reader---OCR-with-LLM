import os
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

print("Loading TrOCR model...")
# ===============================
# OCR Process Folder
# ===============================
# OCR_PROCESS_FOLDER = "user-ocr-process"
# os.makedirs(OCR_PROCESS_FOLDER, exist_ok=True)

# ===============================
# Load TrOCR Model (Load once)
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.to(device)
model.eval()

# Use fewer beams to reduce memory pressure during generation.
GEN_NUM_BEAMS = 2


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _move_model_to(target_device: torch.device) -> None:
    global device
    if device == target_device:
        return
    model.to(target_device)
    device = target_device
    model.eval()


def _infer_line_text(pil_img: Image.Image) -> str:
    """
    Run OCR inference for one line. If CUDA runs out of memory, retry on CPU.
    """
    global device
    try:
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=256,
                num_beams=GEN_NUM_BEAMS,
                early_stopping=True,
            )
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        del pixel_values, generated_ids
        return text
    except RuntimeError as e:
        if device.type == "cuda" and _is_cuda_oom_error(e):
            print("CUDA OOM during OCR line inference. Retrying on CPU.")
            torch.cuda.empty_cache()
            _move_model_to(torch.device("cpu"))

            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=256,
                    num_beams=GEN_NUM_BEAMS,
                    early_stopping=True,
                )
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            del pixel_values, generated_ids
            return text
        raise


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

    # Disk debug output disabled to avoid filling local storage.
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # process_name = f"{filename}_{timestamp}"
    # process_folder = os.path.join(OCR_PROCESS_FOLDER, process_name)
    # os.makedirs(process_folder, exist_ok=True)
    # cv2.imwrite(os.path.join(process_folder, "01_original.png"), img_bgr)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)
    # cv2.imwrite(os.path.join(process_folder, "02_grayscale_deskew.png"), gray)

    # Light border crop for photos (keeps PDF pages intact)
    h, w = gray.shape
    if is_pdf:
        cropped = gray
    else:
        pad = 25
        cropped = gray[pad : max(pad + 1, h - pad), pad : max(pad + 1, w - pad)]

    # cv2.imwrite(os.path.join(process_folder, "03_cropped.png"), cropped)

    bin_inv = make_binary_inv(cropped)
    # cv2.imwrite(os.path.join(process_folder, "04_threshold_bin_inv.png"), bin_inv)

    text_only, lines_mask = remove_table_lines(bin_inv)
    # cv2.imwrite(os.path.join(process_folder, "05_lines_mask.png"), lines_mask)
    # cv2.imwrite(os.path.join(process_folder, "06_text_only.png"), text_only)

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

        # cv2.imwrite(os.path.join(process_folder, f"line_{idx:03d}.png"), line_img)

        rgb = cv2.cvtColor(line_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(rgb)
        text = _infer_line_text(pil_img)

        # Basic cleanup: ignore extremely short garbage tokens
        if len(text) <= 1:
            continue

        extracted_lines.append(text)
        line_details.append(f"Line {idx}: [{x}, {y}, {bw}, {bh}] -> {text}")

    # cv2.imwrite(os.path.join(process_folder, "07_bounding_boxes.png"), bbox_img)
    # with open(os.path.join(process_folder, "ocr_results.txt"), "w", encoding="utf-8") as f:
    #     f.write("\n".join(line_details))
    #     f.write("\n\nFinal Extracted Text:\n")
    #     f.write("\n".join(extracted_lines))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return "\n".join(extracted_lines)
