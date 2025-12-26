"""
Document Processing Module
Handles multi-format blueprint processing, image preprocessing, and text extraction
"""

import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import ezdxf
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Core document processing engine for HVAC blueprints
    """
    
    def __init__(self):
        self.supported_formats = ['pdf', 'dwg', 'dxf', 'png', 'jpg', 'jpeg']
        
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process blueprint file and extract information
        
        Args:
            file_path: Path to the blueprint file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            return self._process_pdf(file_path)
        elif file_ext in ['dwg', 'dxf']:
            return self._process_cad(file_path)
        elif file_ext in ['png', 'jpg', 'jpeg']:
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF blueprint"""
        try:
            doc = fitz.open(file_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for quality
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                
                # Convert to RGB if needed
                if pix.n == 4:  # RGBA
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                
                images.append(img_data)
            
            return {
                'images': images,
                'page_count': len(doc),
                'format': 'pdf',
                'metadata': {
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', '')
                }
            }
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise
    
    def _process_cad(self, file_path: str) -> Dict[str, Any]:
        """Process CAD file (DWG/DXF)"""
        try:
            doc = ezdxf.readfile(file_path)
            modelspace = doc.modelspace()
            
            entities = []
            for entity in modelspace:
                entities.append({
                    'type': entity.dxftype(),
                    'layer': entity.dxf.layer if hasattr(entity.dxf, 'layer') else 'default'
                })
            
            return {
                'entities': entities,
                'entity_count': len(entities),
                'format': 'cad',
                'layers': list(doc.layers)
            }
        except Exception as e:
            logger.error(f"CAD processing error: {e}")
            raise
    
    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """Process image blueprint"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Failed to load image")
            
            # Get image properties
            height, width = img.shape[:2]
            
            return {
                'images': [img],
                'format': 'image',
                'dimensions': {
                    'width': width,
                    'height': height
                }
            }
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Threshold
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def detect_scale(self, image: np.ndarray) -> Optional[float]:
        """
        Detect scale factor from blueprint
        
        Args:
            image: Blueprint image
            
        Returns:
            Scale factor (e.g., 0.25 for 1/4" = 1')
        """
        # This is a simplified version - full implementation would use OCR
        # to find scale notation in the blueprint
        
        # Mock implementation
        return 0.25  # 1/4" = 1'
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from blueprint using OCR
        
        Args:
            image: Blueprint image
            
        Returns:
            Extracted text
        """
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""


def create_document_processor() -> DocumentProcessor:
    """Factory function to create document processor"""
    return DocumentProcessor()
