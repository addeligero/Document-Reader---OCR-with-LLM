import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

# Create process folder
PROCESS_FOLDER = "process"
os.makedirs(PROCESS_FOLDER, exist_ok=True)

# Initialize model
device = torch.device("cpu")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
model.to(device)
model.eval()

def create_sample_document():
    """Create a sample document image for visualization"""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add some text lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = [
        "Sample Document Title",
        "This is the first line of text",
        "Here is another line for processing",
        "Third line with more content",
        "Final line of the document"
    ]
    
    y_pos = 150
    for line in lines:
        cv2.putText(img, line, (80, y_pos), font, 0.8, (0, 0, 0), 2)
        y_pos += 100
    
    return img

def visualize_ocr_image_process():
    """Visualize the ocr_image function step by step"""
    print("Generating ocr_image() process visualization...")
    
    # Create sample image
    img = create_sample_document()
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "1_ocr_image_input.png"), img)
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "2_ocr_image_grayscale.png"), gray)
    
    # Step 2: Crop borders
    h, w = gray.shape
    cropped = gray[50:h-50, 50:w-50]
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "3_ocr_image_cropped.png"), cropped)
    
    # Step 3: Threshold (Otsu)
    _, thresh = cv2.threshold(
        cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "4_ocr_image_threshold.png"), thresh)
    
    # Step 4: Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "5_ocr_image_morphology_close.png"), closed)
    
    # Step 5: Dilation
    kernel_d = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    dilated = cv2.dilate(closed, kernel_d, iterations=1)
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "6_ocr_image_dilated.png"), dilated)
    
    # Step 6: Find contours and draw bounding boxes
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = sorted([cv2.boundingRect(c) for c in contours], key=lambda b: b[1])
    
    # Draw boxes on cropped image
    cropped_with_boxes = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        if w >= 40 and h >= 15:
            cv2.rectangle(cropped_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(cropped_with_boxes, f"{w}x{h}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "7_ocr_image_bounding_boxes.png"), cropped_with_boxes)
    
    print(f"✓ Found {len([b for b in boxes if b[2] >= 40 and b[3] >= 15])} text regions")

def visualize_ocr_full_page_process():
    """Visualize the ocr_full_page function"""
    print("\nGenerating ocr_full_page() process visualization...")
    
    # Create sample full page
    img = create_sample_document()
    cv2.imwrite(os.path.join(PROCESS_FOLDER, "8_ocr_fullpage_input.png"), img)
    
    # Convert to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.save(os.path.join(PROCESS_FOLDER, "9_ocr_fullpage_rgb_converted.png"))
    
    print("✓ Full page OCR processes entire image at once")

def create_flow_diagram():
    """Create a flow diagram showing the overall process"""
    print("\nGenerating overall flow diagram...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11, 'OCR Processing Pipeline', fontsize=20, weight='bold', ha='center')
    
    # Define colors
    color_input = '#E3F2FD'
    color_process = '#FFF3E0'
    color_ocr = '#F3E5F5'
    color_output = '#E8F5E9'
    
    def draw_box(x, y, w, h, text, color):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, weight='bold')
    
    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic')
    
    # Input
    draw_box(4, 10, 2, 0.6, 'FILE UPLOAD', color_input)
    draw_arrow(5, 10, 5, 9.4, '')
    
    # File type decision
    draw_box(4, 8.8, 2, 0.6, 'Detect File Type', color_process)
    
    # PDF Branch
    draw_arrow(5, 8.8, 2, 8, '')
    draw_box(0.5, 7.4, 2, 0.6, 'PDF', color_input)
    draw_arrow(1.5, 7.4, 1.5, 6.8, '')
    draw_box(0.5, 6.2, 2, 0.6, 'Try pdfplumber\ntext extraction', color_process)
    draw_arrow(1.5, 6.2, 1.5, 5.6, '')
    draw_box(0.5, 5, 2, 0.6, 'Text found?', color_process)
    draw_arrow(2.5, 5.3, 3.5, 5.3, 'No')
    draw_box(3.5, 5, 2, 0.6, 'Convert to images\n(300 DPI)', color_process)
    draw_arrow(4.5, 5, 4.5, 4.4, '')
    draw_box(3.5, 3.8, 2, 0.6, 'ocr_full_page()\non each page', color_ocr)
    
    # DOCX Branch
    draw_arrow(6, 8.8, 6, 8, '')
    draw_box(5, 7.4, 2, 0.6, 'DOCX', color_input)
    draw_arrow(6, 7.4, 6, 6.8, '')
    draw_box(5, 6.2, 2, 0.6, 'Extract paragraphs\n(python-docx)', color_process)
    
    # Image Branch
    draw_arrow(5, 8.8, 8, 8, '')
    draw_box(7.5, 7.4, 2, 0.6, 'IMAGE\n(PNG/JPG)', color_input)
    draw_arrow(8.5, 7.4, 8.5, 6.8, '')
    draw_box(7.5, 6.2, 2, 0.6, 'Load with\nOpenCV', color_process)
    draw_arrow(8.5, 6.2, 8.5, 5.6, '')
    draw_box(7.5, 5, 2, 0.6, 'ocr_image()\nwith line detection', color_ocr)
    
    # Convergence to output
    draw_arrow(1.5, 5, 5, 3, '')
    draw_arrow(6, 6.2, 5, 3, '')
    draw_arrow(8.5, 5, 5, 3, '')
    draw_arrow(4.5, 3.8, 5, 3, '')
    
    draw_box(4, 2.4, 2, 0.6, 'Extracted Text', color_output)
    draw_arrow(5, 2.4, 5, 1.8, '')
    draw_box(4, 1.2, 2, 0.6, 'JSON Response', color_output)
    
    # Add legend
    ax.text(0.5, 0.5, 'Legend:', fontsize=10, weight='bold')
    draw_box(0.5, 0.1, 0.8, 0.25, 'Input', color_input)
    draw_box(1.5, 0.1, 0.8, 0.25, 'Process', color_process)
    draw_box(2.5, 0.1, 0.8, 0.25, 'OCR', color_ocr)
    draw_box(3.5, 0.1, 0.8, 0.25, 'Output', color_output)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESS_FOLDER, '0_overall_flow_diagram.png'), dpi=300, bbox_inches='tight')
    print("✓ Flow diagram created")

def create_ocr_detail_diagram():
    """Create detailed diagrams for each OCR function"""
    print("\nGenerating detailed OCR function diagrams...")
    
    # OCR Image Function Detail
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    ax.text(5, 13, 'ocr_image() Function - Line Detection', fontsize=18, weight='bold', ha='center')
    
    steps = [
        ('1. Input Image', 'BGR image from OpenCV', 11.5),
        ('2. Convert to Grayscale', 'cv2.cvtColor(img, COLOR_BGR2GRAY)', 10.5),
        ('3. Crop Borders', 'Remove 50px from each side', 9.5),
        ('4. Otsu Threshold', 'Binary inverse + automatic threshold', 8.5),
        ('5. Morphological Close', 'Kernel (25x2) - Connect text horizontally', 7.5),
        ('6. Dilation', 'Kernel (35x1) - Merge text regions', 6.5),
        ('7. Find Contours', 'RETR_EXTERNAL - Get bounding boxes', 5.5),
        ('8. Sort by Y-coordinate', 'Process lines top to bottom', 4.5),
        ('9. Filter Small Boxes', 'Keep only w>=40 and h>=15', 3.5),
        ('10. Extract Each Line', 'Crop grayscale image using bbox', 2.5),
        ('11. TrOCR per Line', 'microsoft/trocr-large-printed', 1.5),
        ('12. Join Results', 'Concatenate with newlines', 0.5),
    ]
    
    for i, (title, desc, y_pos) in enumerate(steps):
        color = '#E3F2FD' if i % 2 == 0 else '#FFF3E0'
        rect = plt.Rectangle((1, y_pos - 0.3), 8, 0.6, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.5, y_pos, title, fontsize=11, weight='bold', va='center')
        ax.text(5, y_pos, desc, fontsize=9, va='center', style='italic')
        
        if i < len(steps) - 1:
            ax.annotate('', xy=(5, y_pos - 0.4), xytext=(5, y_pos - 0.2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESS_FOLDER, '10_ocr_image_function_detail.png'), dpi=300, bbox_inches='tight')
    
    # OCR Full Page Function Detail
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(5, 5.5, 'ocr_full_page() Function - Full Page Recognition', fontsize=18, weight='bold', ha='center')
    
    steps_fp = [
        ('1. Input Image', 'BGR image (usually from PDF page)', 4.2),
        ('2. Convert to RGB', 'cv2.cvtColor(img, COLOR_BGR2RGB)', 3.2),
        ('3. Convert to PIL', 'Image.fromarray(rgb)', 2.2),
        ('4. TrOCR on Full Image', 'Process entire page (max_length=512)', 1.2),
        ('5. Return Text', 'Single decoded string', 0.2),
    ]
    
    for i, (title, desc, y_pos) in enumerate(steps_fp):
        color = '#F3E5F5' if i % 2 == 0 else '#E8F5E9'
        rect = plt.Rectangle((1, y_pos - 0.3), 8, 0.6, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.5, y_pos, title, fontsize=11, weight='bold', va='center')
        ax.text(5, y_pos, desc, fontsize=9, va='center', style='italic')
        
        if i < len(steps_fp) - 1:
            ax.annotate('', xy=(5, y_pos - 0.4), xytext=(5, y_pos - 0.2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROCESS_FOLDER, '11_ocr_fullpage_function_detail.png'), dpi=300, bbox_inches='tight')
    
    print("✓ Detailed function diagrams created")

if __name__ == "__main__":
    print("=" * 60)
    print("Creating OCR Process Visualizations")
    print("=" * 60)
    
    create_flow_diagram()
    create_ocr_detail_diagram()
    visualize_ocr_image_process()
    visualize_ocr_full_page_process()
    
    print("\n" + "=" * 60)
    print(f"✓ All visualizations saved to '{PROCESS_FOLDER}/' folder")
    print("=" * 60)
    print("\nGenerated files:")
    for filename in sorted(os.listdir(PROCESS_FOLDER)):
        print(f"  - {filename}")
