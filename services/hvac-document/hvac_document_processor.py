"""
HVAC Document Processing Module

This module provides HVAC-specialized document processing capabilities including:
- Quality-preserving format conversion
- HVAC-specific image enhancement
- Multi-page blueprint handling
- Symbol and line work preservation

Optimized for HVAC blueprint formats: PDF, DWG, DXF, PNG, JPG, TIFF
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np


class BlueprintFormat(Enum):
    """Supported HVAC blueprint formats"""
    PDF = "pdf"
    DWG = "dwg"
    DXF = "dxf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"


class PageType(Enum):
    """HVAC blueprint page classifications"""
    PLAN_VIEW = "plan_view"
    SECTION_VIEW = "section_view"
    DETAIL_VIEW = "detail_view"
    EQUIPMENT_SCHEDULE = "equipment_schedule"
    LEGEND = "legend"
    UNKNOWN = "unknown"


@dataclass
class QualityMetrics:
    """Blueprint quality assessment metrics"""
    line_clarity: float  # 0-1 score
    symbol_visibility: float  # 0-1 score
    text_readability: float  # 0-1 score
    overall_quality: float  # 0-1 score
    resolution_dpi: Optional[int] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class HVACDocumentProcessor:
    """
    HVAC-specialized document processing pipeline
    
    Handles format conversion, quality assessment, and enhancement
    specifically optimized for HVAC blueprints.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default HVAC-specific processing configuration"""
        return {
            "target_dpi": 300,
            "min_acceptable_dpi": 150,
            "enhance_ductwork_lines": True,
            "enhance_symbols": True,
            "preserve_layers": True,
            "multi_page_support": True,
            "quality_threshold": 0.6
        }
    
    def process_document(
        self,
        file_path: str,
        format_hint: Optional[BlueprintFormat] = None
    ) -> Dict[str, Any]:
        """
        Process HVAC blueprint document
        
        Args:
            file_path: Path to the blueprint file
            format_hint: Optional format hint if known
            
        Returns:
            Dictionary containing:
            - pages: List of processed page images
            - metadata: Document metadata
            - quality_metrics: Quality assessment results
        """
        # Detect format if not provided
        if format_hint is None:
            format_hint = self._detect_format(file_path)
        
        self.logger.info(f"Processing HVAC document: {file_path} (format: {format_hint.value})")
        
        # Process based on format
        if format_hint == BlueprintFormat.PDF:
            return self._process_pdf(file_path)
        elif format_hint in [BlueprintFormat.DWG, BlueprintFormat.DXF]:
            return self._process_cad(file_path, format_hint)
        else:
            return self._process_raster(file_path)
    
    def _detect_format(self, file_path: str) -> BlueprintFormat:
        """Detect blueprint format from file extension"""
        import os
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        
        try:
            return BlueprintFormat(ext)
        except ValueError:
            self.logger.warning(f"Unknown format: {ext}, defaulting to PNG")
            return BlueprintFormat.PNG
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF blueprint with HVAC-specific optimization and multi-page support"""
        pages = []
        metadata = {"format": "pdf", "page_count": 0}
        
        try:
            # Try to use PyMuPDF for PDF processing
            try:
                import fitz  # PyMuPDF
                
                self.logger.info("PDF processing with PyMuPDF - extracting all pages")
                
                pdf_document = fitz.open(file_path)
                metadata["page_count"] = pdf_document.page_count
                
                for page_num in range(pdf_document.page_count):
                    page = pdf_document[page_num]
                    
                    # Render at high DPI for quality
                    mat = fitz.Matrix(self.config["target_dpi"] / 72, 
                                     self.config["target_dpi"] / 72)
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to numpy array
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
                    img_data = img_data.reshape(pix.height, pix.width, pix.n)
                    
                    # Convert RGBA to RGB if needed
                    if pix.n == 4:
                        img_data = img_data[:, :, :3]
                    
                    # Classify page type
                    page_type = self._classify_page_type(img_data)
                    
                    pages.append({
                        "image": img_data,
                        "page_number": page_num + 1,
                        "page_type": page_type,
                        "dimensions": (pix.width, pix.height)
                    })
                    
                    self.logger.info(
                        f"Extracted page {page_num + 1}/{pdf_document.page_count}: "
                        f"{page_type.value}"
                    )
                
                pdf_document.close()
                
            except ImportError:
                self.logger.warning("PyMuPDF not available, using fallback")
                # Fallback to placeholder
                pages.append({
                    "image": np.zeros((3000, 4000, 3), dtype=np.uint8),
                    "page_number": 1,
                    "page_type": PageType.PLAN_VIEW,
                    "dimensions": (4000, 3000)
                })
                metadata["page_count"] = 1
            
        except Exception as e:
            self.logger.error(f"PDF processing failed: {e}")
            raise
        
        # Assess quality and enhance each page
        for page in pages:
            page["quality_metrics"] = self.assess_quality(page["image"])
            if page["quality_metrics"].overall_quality < self.config["quality_threshold"]:
                page["image"] = self.enhance_blueprint(page["image"])
        
        # Add cross-page analysis metadata
        metadata["page_types"] = [p["page_type"].value for p in pages]
        metadata["plan_views"] = sum(1 for p in pages if p["page_type"] == PageType.PLAN_VIEW)
        metadata["sections"] = sum(1 for p in pages if p["page_type"] == PageType.SECTION_VIEW)
        metadata["details"] = sum(1 for p in pages if p["page_type"] == PageType.DETAIL_VIEW)
        
        return {
            "pages": pages,
            "metadata": metadata,
            "processing_config": self.config
        }
    
    def _process_cad(
        self,
        file_path: str,
        format: BlueprintFormat
    ) -> Dict[str, Any]:
        """Process DWG/DXF files with HVAC layer recognition"""
        self.logger.info(f"CAD processing: {format.value}")
        
        # In production, would use ezdxf or similar
        # Would extract HVAC-specific layers (HVAC, M-, Mechanical, etc.)
        # Would preserve blocks and symbol definitions
        
        pages = [{
            "image": np.zeros((3000, 4000, 3), dtype=np.uint8),
            "page_number": 1,
            "page_type": PageType.PLAN_VIEW,
            "layers": ["HVAC-SUPPLY", "HVAC-RETURN", "HVAC-EQUIPMENT"]
        }]
        
        metadata = {
            "format": format.value,
            "page_count": len(pages),
            "cad_metadata": {
                "hvac_layers_found": True,
                "symbol_blocks_extracted": 0
            }
        }
        
        return {
            "pages": pages,
            "metadata": metadata,
            "processing_config": self.config
        }
    
    def _process_raster(self, file_path: str) -> Dict[str, Any]:
        """Process raster image files (PNG, JPG, TIFF)"""
        try:
            import cv2
            image = cv2.imread(file_path)
            
            if image is None:
                raise ValueError(f"Failed to load image: {file_path}")
            
            # Assess quality
            quality_metrics = self.assess_quality(image)
            
            # Enhance if needed
            if quality_metrics.overall_quality < self.config["quality_threshold"]:
                image = self.enhance_blueprint(image)
            
            pages = [{
                "image": image,
                "page_number": 1,
                "page_type": self._classify_page_type(image),
                "quality_metrics": quality_metrics
            }]
            
            metadata = {
                "format": self._detect_format(file_path).value,
                "page_count": 1,
                "dimensions": image.shape[:2]
            }
            
            return {
                "pages": pages,
                "metadata": metadata,
                "processing_config": self.config
            }
            
        except Exception as e:
            self.logger.error(f"Raster processing failed: {e}")
            raise
    
    def assess_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Assess HVAC blueprint quality
        
        Args:
            image: Blueprint image as numpy array
            
        Returns:
            QualityMetrics object with detailed assessment
        """
        issues = []
        
        try:
            import cv2
            
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Line clarity assessment (edge detection)
            edges = cv2.Canny(gray, 50, 150)
            line_clarity = np.sum(edges > 0) / edges.size
            
            if line_clarity < 0.05:
                issues.append("Low line clarity - ductwork may be faded")
            
            # Symbol visibility (local contrast)
            # Using Laplacian variance as proxy for detail
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            symbol_visibility = min(1.0, laplacian_var / 1000.0)
            
            if symbol_visibility < 0.3:
                issues.append("Low symbol visibility - poor scanning quality")
            
            # Text readability (placeholder - would use OCR confidence)
            text_readability = 0.7  # Placeholder
            
            # Overall quality score
            overall_quality = (line_clarity + symbol_visibility + text_readability) / 3.0
            
            return QualityMetrics(
                line_clarity=line_clarity,
                symbol_visibility=symbol_visibility,
                text_readability=text_readability,
                overall_quality=overall_quality,
                issues=issues
            )
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            return QualityMetrics(
                line_clarity=0.5,
                symbol_visibility=0.5,
                text_readability=0.5,
                overall_quality=0.5,
                issues=["Quality assessment failed"]
            )
    
    def enhance_blueprint(self, image: np.ndarray) -> np.ndarray:
        """
        Apply HVAC-specific enhancements to blueprint
        
        Args:
            image: Input blueprint image
            
        Returns:
            Enhanced image
        """
        try:
            import cv2
            
            enhanced = image.copy()
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            
            # Enhance ductwork lines if enabled
            if self.config.get("enhance_ductwork_lines", True):
                # Apply morphological operations for line enhancement
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast for symbol visibility
            if self.config.get("enhance_symbols", True):
                gray = cv2.equalizeHist(gray)
            
            # Apply adaptive thresholding for better line definition
            # (optional, depends on blueprint quality)
            
            # Convert back to BGR if input was color
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                enhanced = gray
            
            self.logger.info("Blueprint enhancement applied")
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Enhancement failed: {e}")
            return image
    
    def _classify_page_type(self, image: np.ndarray) -> PageType:
        """
        Classify HVAC blueprint page type using text and content analysis
        
        Args:
            image: Blueprint image
            
        Returns:
            PageType classification
        """
        try:
            import pytesseract
            
            # Extract text from image (just from title block area typically top/bottom)
            height, width = image.shape[:2]
            
            # Sample top 10% and bottom 10% for title block text
            top_region = image[:int(height * 0.1), :]
            bottom_region = image[int(height * 0.9):, :]
            
            # Extract text from both regions
            try:
                top_text = pytesseract.image_to_string(top_region).upper()
                bottom_text = pytesseract.image_to_string(bottom_region).upper()
                combined_text = top_text + " " + bottom_text
                
                # Classify based on keywords
                if any(keyword in combined_text for keyword in ["PLAN", "FLOOR", "LAYOUT"]):
                    return PageType.PLAN_VIEW
                elif any(keyword in combined_text for keyword in ["SECTION", "ELEVATION", "CUT"]):
                    return PageType.SECTION_VIEW
                elif any(keyword in combined_text for keyword in ["DETAIL", "ENLARGED", "CALLOUT"]):
                    return PageType.DETAIL_VIEW
                elif any(keyword in combined_text for keyword in ["SCHEDULE", "LIST", "TABLE", "SPEC"]):
                    return PageType.EQUIPMENT_SCHEDULE
                elif any(keyword in combined_text for keyword in ["LEGEND", "SYMBOL", "KEY", "NOTE"]):
                    return PageType.LEGEND
                
            except Exception as e:
                self.logger.debug(f"OCR classification failed: {e}")
            
        except ImportError:
            self.logger.debug("pytesseract not available for page classification")
        
        # Fallback to content-based classification
        height, width = image.shape[:2]
        aspect_ratio = width / height
        
        # Analyze content density
        gray = image if len(image.shape) == 2 else image[:, :, 0]
        edge_density = self._calculate_edge_density(gray)
        
        # Plan views typically have:
        # - Wider aspect ratio
        # - Lower edge density (more white space)
        if aspect_ratio > 1.4 and edge_density < 0.15:
            return PageType.PLAN_VIEW
        
        # Sections typically have:
        # - Taller aspect ratio
        # - Higher edge density (more detail)
        elif aspect_ratio < 0.8 and edge_density > 0.15:
            return PageType.SECTION_VIEW
        
        # Details typically have:
        # - Moderate aspect ratio
        # - Very high edge density
        elif edge_density > 0.20:
            return PageType.DETAIL_VIEW
        
        # Default to plan view
        return PageType.PLAN_VIEW
    
    def _calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density for content analysis"""
        try:
            import cv2
            edges = cv2.Canny(image, 50, 150)
            return np.sum(edges > 0) / edges.size
        except:
            return 0.1  # Default value
    
    def extract_hvac_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract HVAC symbols using template matching
        
        Args:
            image: Blueprint image
            
        Returns:
            List of detected symbols with locations
        """
        # Placeholder for symbol extraction
        # In production, would use template matching with ASHRAE/SMACNA symbol library
        
        symbols = []
        
        self.logger.info("Symbol extraction - would use ASHRAE/SMACNA library")
        
        return symbols
    
    def enhance_line_work(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance ductwork lines using morphological operations
        
        Args:
            image: Blueprint image
            
        Returns:
            Image with enhanced line work
        """
        try:
            import cv2
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Ductwork line thinning
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thinned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Junction detection (for connectivity analysis)
            # Would identify duct junctions, tees, transitions
            
            return thinned
            
        except Exception as e:
            self.logger.error(f"Line work enhancement failed: {e}")
            return image


def create_hvac_document_processor(
    config: Optional[Dict[str, Any]] = None
) -> HVACDocumentProcessor:
    """
    Factory function to create HVAC document processor
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured HVACDocumentProcessor instance
    """
    return HVACDocumentProcessor(config=config)
