"""
HVAC Document Processing Services

HVAC-specialized document processing:
- Format conversion (PDF, DWG, DXF)
- Quality assessment and enhancement
- Multi-page handling
- Enhanced document processing with OCR
- Hybrid processing capabilities
- Table extraction
"""

__version__ = "1.0.0"

# Import key components
from .hvac_document_processor import (
    DocumentProcessor, BlueprintFormat, PageType, QualityMetrics
)
from .hvac_symbol_library import HVACSymbolLibrary
from .enhanced_document_processor import create_enhanced_processor
from .hybrid_document_processor import create_hybrid_processor
from .table_extractor import TableExtractor
from .document_processor import DocumentProcessor as BaseDocumentProcessor

__all__ = [
    "DocumentProcessor",
    "BlueprintFormat",
    "PageType",
    "QualityMetrics",
    "HVACSymbolLibrary",
    "create_enhanced_processor",
    "create_hybrid_processor",
    "TableExtractor",
    "BaseDocumentProcessor",
]
