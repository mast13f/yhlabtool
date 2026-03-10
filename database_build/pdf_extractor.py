"""
PDF MALDI Figure Extractor
Extracts MALDI imaging figures from scientific PDFs using Claude AI.
"""

import anthropic
import base64
import os
from pathlib import Path
from typing import List, Dict, Tuple
import fitz  # PyMuPDF


class MALDIFigureExtractor:
    """Extract MALDI imaging figures from PDF papers using Claude AI."""
    
    def __init__(self, api_key: str = None, output_dpi: int = 300):
        """
        Args:
            api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
            output_dpi: DPI for extracted images (default 300)
        """
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.output_dpi = output_dpi
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[int, bytes, str]]:
        """Extract all embedded images from PDF."""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                images.append((page_num + 1, image_bytes, image_ext))
        
        doc.close()
        return images
    
    def render_pdf_pages_as_images(self, pdf_path: str) -> List[Tuple[int, bytes]]:
        """Render each PDF page as high-res image (captures vector graphics)."""
        doc = fitz.open(pdf_path)
        page_images = []
        
        zoom = self.output_dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            page_images.append((page_num + 1, png_bytes))
        
        doc.close()
        return page_images
    
    def is_maldi_image(self, image_bytes: bytes, image_format: str = "png") -> Tuple[bool, str, float]:
        """
        Use Claude to determine if image contains MALDI data.
        Returns: (is_maldi, explanation, confidence_score)
        """
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        media_type_map = {
            'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'gif': 'image/gif', 'webp': 'image/webp'
        }
        media_type = media_type_map.get(image_format.lower(), 'image/png')
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": """Analyze this image and determine if it contains MALDI (Matrix-Assisted Laser Desorption/Ionization) imaging mass spectrometry data.

MALDI imaging figures typically show:
- Spatial distribution maps of metabolites/molecules in tissue sections
- Color-coded intensity maps overlaid on tissue images
- Mass-to-charge (m/z) value annotations
- Multiple panels showing different m/z values
- Ion intensity heatmaps
- Keywords: "MALDI", "MSI", "IMS", "mass spectrometry imaging", m/z ratios

Answer in this EXACT format:
IS_MALDI: [YES/NO]
CONFIDENCE: [0-100]
REASON: [brief explanation]

Be strict - only answer YES if confident this is MALDI imaging data, not regular microscopy, western blots, or other biochemistry figures."""
                        }
                    ],
                }
            ],
        )
        
        response_text = message.content[0].text
        
        is_maldi = "YES" in response_text.split("IS_MALDI:")[1].split("\n")[0].upper()
        
        try:
            confidence_line = response_text.split("CONFIDENCE:")[1].split("\n")[0].strip()
            confidence = float(''.join(filter(str.isdigit, confidence_line)))
        except:
            confidence = 50.0
        
        try:
            reason = response_text.split("REASON:")[1].strip()
        except:
            reason = "Unable to parse reason"
        
        return is_maldi, reason, confidence
    
    def extract_from_pdf(self, pdf_path: str, output_folder: str = "maldi_figures", 
                        confidence_threshold: float = 70.0) -> List[str]:
        """
        Extract MALDI figures from PDF and save them.
        Returns list of saved image paths.
        """
        print(f"\nProcessing: {pdf_path}")
        pdf_name = Path(pdf_path).stem
        
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_images = []
        
        # Extract embedded images
        print("Extracting embedded images...")
        embedded_images = self.extract_images_from_pdf(pdf_path)
        
        for page_num, img_bytes, img_ext in embedded_images:
            print(f"  Checking embedded image from page {page_num}...")
            
            is_maldi, reason, confidence = self.is_maldi_image(img_bytes, img_ext)
            
            if is_maldi and confidence >= confidence_threshold:
                output_file = output_path / f"{pdf_name}_page{page_num}_embedded.{img_ext}"
                with open(output_file, 'wb') as f:
                    f.write(img_bytes)
                saved_images.append(str(output_file))
                print(f"    ✓ MALDI detected (confidence: {confidence:.1f}%) - Saved: {output_file.name}")
                print(f"      Reason: {reason}")
            else:
                print(f"    ✗ Not MALDI (confidence: {confidence:.1f}%)")
        
        # Render pages as images
        print("\nRendering PDF pages as images...")
        page_images = self.render_pdf_pages_as_images(pdf_path)
        
        for page_num, png_bytes in page_images:
            print(f"  Checking rendered page {page_num}...")
            
            is_maldi, reason, confidence = self.is_maldi_image(png_bytes, "png")
            
            if is_maldi and confidence >= confidence_threshold:
                output_file = output_path / f"{pdf_name}_page{page_num}_full.png"
                with open(output_file, 'wb') as f:
                    f.write(png_bytes)
                saved_images.append(str(output_file))
                print(f"    ✓ MALDI detected (confidence: {confidence:.1f}%) - Saved: {output_file.name}")
                print(f"      Reason: {reason}")
            else:
                print(f"    ✗ Not MALDI (confidence: {confidence:.1f}%)")
        
        print(f"\nExtracted {len(saved_images)} MALDI figures from {pdf_path}")
        return saved_images
    
    def batch_extract(self, pdf_paths: List[str], output_folder: str = "maldi_figures",
                     confidence_threshold: float = 70.0) -> Dict[str, List[str]]:
        """Extract MALDI figures from multiple PDFs."""
        results = {}
        
        for pdf_path in pdf_paths:
            try:
                extracted = self.extract_from_pdf(pdf_path, output_folder, confidence_threshold)
                results[pdf_path] = extracted
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                results[pdf_path] = []
        
        total_extracted = sum(len(imgs) for imgs in results.values())
        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION COMPLETE")
        print(f"Processed {len(pdf_paths)} PDFs")
        print(f"Total MALDI figures extracted: {total_extracted}")
        print(f"{'='*60}")
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <pdf_file1> [pdf_file2 ...]")
        sys.exit(1)
    
    extractor = MALDIFigureExtractor()
    extractor.batch_extract(sys.argv[1:], output_folder="extracted_maldi_figures")
