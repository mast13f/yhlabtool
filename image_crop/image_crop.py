import numpy as np
from PIL import Image
import cv2
from pathlib import Path

def auto_crop_maldi(image_path, output_path=None, padding=10):
    """
    Automatically crop MALDI image to tissue boundaries.
    
    Args:
        image_path: Path to input image
        output_path: Path to save cropped image (optional)
        padding: Extra pixels around tissue (default: 10)
    
    Returns:
        Cropped PIL Image
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Automatic thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find bounding box
    coords = cv2.findNonZero(binary)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img_array.shape[1] - x, w + 2*padding)
    h = min(img_array.shape[0] - y, h + 2*padding)
    
    # Crop
    cropped = img.crop((x, y, x+w, y+h))
    
    # Save if output path provided
    if output_path:
        cropped.save(output_path)
        print(f"Saved to {output_path}")
    
    return cropped


def batch_crop(input_folder, output_folder, padding=10):
    """Process all images in a folder"""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Process all common image formats
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
        for img_file in input_path.glob(ext):
            print(f"Processing {img_file.name}...")
            auto_crop_maldi(img_file, output_path / img_file.name, padding=padding)


# USAGE EXAMPLES:

# Single image
auto_crop_maldi('input.jpeg', 'output.png')

# Batch process entire folder
# batch_crop('raw_images/', 'cropped_images/', padding=10)