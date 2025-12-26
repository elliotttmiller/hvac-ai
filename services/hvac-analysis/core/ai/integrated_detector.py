"""
Integrated HVAC Detector (Complete YOLOplan Integration - Weeks 1-16)
Combines YOLOplan symbol detection with existing SAHI and document processing

This module integrates:
- YOLOplan for MEP symbol detection
- SAHI for large component detection  
- Enhanced document processing for text extraction
- BOM generation and connectivity analysis
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

from .yoloplan_detector import create_yoloplan_detector, YOLOplanDetector
from .yoloplan_bom import create_bom_generator, create_connectivity_analyzer
from .detector import HVACDetector
from ..document.enhanced_processor import create_enhanced_processor
from ..document.hybrid_processor import create_hybrid_processor

logger = logging.getLogger(__name__)


class IntegratedHVACDetector:
    """
    Integrated detector combining all analysis capabilities
    
    Complete pipeline:
    1. Document processing (text, metadata, layout)
    2. Symbol detection (YOLOplan - MEP symbols)
    3. Component detection (SAHI - large equipment)
    4. BOM generation (quantity takeoff)
    5. Connectivity analysis (system netlist)
    6. Result fusion and validation
    """
    
    def __init__(self,
                 yoloplan_model: Optional[str] = None,
                 use_sahi: bool = True,
                 use_document_processing: bool = True,
                 use_bom_generation: bool = True,
                 use_connectivity_analysis: bool = True):
        """
        Initialize integrated detector
        
        Args:
            yoloplan_model: Path to YOLOplan model
            use_sahi: Enable SAHI component detection
            use_document_processing: Enable document processing
            use_bom_generation: Enable BOM generation
            use_connectivity_analysis: Enable connectivity analysis
        """
        # YOLOplan symbol detector (Weeks 1-10)
        self.yoloplan = create_yoloplan_detector(
            model_path=yoloplan_model or "models/yoloplan_hvac_v1.pt"
        )
        
        # SAHI component detector (existing)
        self.sahi_detector = None
        if use_sahi:
            try:
                self.sahi_detector = HVACDetector()
                logger.info("SAHI detector loaded")
            except Exception as e:
                logger.warning(f"Could not load SAHI detector: {e}")
        
        # Document processors (existing - enhanced)
        self.doc_processor = None
        self.hybrid_processor = None
        if use_document_processing:
            try:
                self.doc_processor = create_enhanced_processor(use_cache=True)
                self.hybrid_processor = create_hybrid_processor(
                    ocr_engine="easyocr",
                    confidence_threshold=0.6
                )
                logger.info("Document processors loaded")
            except Exception as e:
                logger.warning(f"Could not load document processors: {e}")
        
        # BOM generator (Weeks 11-14)
        self.bom_generator = None
        if use_bom_generation:
            self.bom_generator = create_bom_generator()
            logger.info("BOM generator loaded")
        
        # Connectivity analyzer (Weeks 11-14)
        self.connectivity_analyzer = None
        if use_connectivity_analysis:
            self.connectivity_analyzer = create_connectivity_analyzer()
            logger.info("Connectivity analyzer loaded")
        
        logger.info("Integrated HVAC Detector initialized")
    
    def analyze_blueprint(self,
                         image: np.ndarray,
                         include_symbols: bool = True,
                         include_components: bool = True,
                         include_text: bool = True,
                         include_bom: bool = True,
                         include_connectivity: bool = True) -> Dict[str, Any]:
        """
        Complete blueprint analysis
        
        Args:
            image: Input blueprint image
            include_symbols: Run YOLOplan symbol detection
            include_components: Run SAHI component detection
            include_text: Run document/text processing
            include_bom: Generate BOM
            include_connectivity: Analyze connectivity
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting integrated blueprint analysis")
        
        results = {
            'document': None,
            'text': None,
            'symbols': None,
            'components': None,
            'bom': None,
            'connectivity': None,
            'metadata': {}
        }
        
        # Stage 1: Document Processing
        if include_text and self.doc_processor and self.hybrid_processor:
            logger.info("Stage 1: Document processing")
            try:
                results['document'] = self.doc_processor.process(image)
                results['text'] = self.hybrid_processor.process(image)
                results['metadata']['text_elements'] = len(results['text'].get('results', []))
                logger.info(f"Extracted {results['metadata']['text_elements']} text elements")
            except Exception as e:
                logger.error(f"Document processing failed: {e}")
        
        # Stage 2: Symbol Detection (YOLOplan)
        if include_symbols:
            logger.info("Stage 2: Symbol detection (YOLOplan)")
            try:
                results['symbols'] = self.yoloplan.detect_symbols(image)
                results['metadata']['total_symbols'] = results['symbols']['total_symbols']
                logger.info(f"Detected {results['metadata']['total_symbols']} symbols")
            except Exception as e:
                logger.error(f"Symbol detection failed: {e}")
                results['symbols'] = {'detections': [], 'counts': {}, 'total_symbols': 0}
        
        # Stage 3: Component Detection (SAHI)
        if include_components and self.sahi_detector:
            logger.info("Stage 3: Component detection (SAHI)")
            try:
                results['components'] = self.sahi_detector.detect_with_sahi(image)
                results['metadata']['total_components'] = len(results['components'])
                logger.info(f"Detected {results['metadata']['total_components']} components")
            except Exception as e:
                logger.error(f"Component detection failed: {e}")
                results['components'] = []
        
        # Stage 4: BOM Generation
        if include_bom and self.bom_generator and results['symbols']:
            logger.info("Stage 4: BOM generation")
            try:
                results['bom'] = self.bom_generator.generate_bom(
                    results['symbols']['counts'],
                    results['symbols']['detections']
                )
                results['metadata']['bom_items'] = len(results['bom'])
                results['metadata']['total_quantity'] = sum(item.quantity for item in results['bom'])
                logger.info(f"Generated BOM with {results['metadata']['bom_items']} items")
            except Exception as e:
                logger.error(f"BOM generation failed: {e}")
        
        # Stage 5: Connectivity Analysis
        if include_connectivity and self.connectivity_analyzer and results['symbols']:
            logger.info("Stage 5: Connectivity analysis")
            try:
                results['connectivity'] = self.connectivity_analyzer.generate_netlist(
                    results['symbols']['detections'],
                    image_size=image.shape[:2]
                )
                results['metadata']['connections'] = len(results['connectivity']['connections'])
                results['metadata']['circuits'] = len(results['connectivity']['circuits'])
                logger.info(f"Identified {results['metadata']['connections']} connections and {results['metadata']['circuits']} circuits")
            except Exception as e:
                logger.error(f"Connectivity analysis failed: {e}")
        
        # Stage 6: Result Fusion
        results['summary'] = self._generate_summary(results)
        
        logger.info("Integrated analysis complete")
        return results
    
    def batch_analyze(self,
                     images: List[np.ndarray],
                     parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Batch analyze multiple blueprints
        
        Args:
            images: List of blueprint images
            parallel: Use parallel processing where possible
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(images)} blueprints")
        
        results = []
        for idx, image in enumerate(images, 1):
            logger.info(f"Processing blueprint {idx}/{len(images)}")
            result = self.analyze_blueprint(image)
            result['batch_index'] = idx
            results.append(result)
        
        # Generate batch summary
        batch_summary = self._generate_batch_summary(results)
        
        return {
            'results': results,
            'batch_summary': batch_summary,
            'total_blueprints': len(images)
        }
    
    def export_results(self,
                      results: Dict[str, Any],
                      output_dir: str = 'output',
                      formats: List[str] = ['json', 'csv']) -> Dict[str, str]:
        """
        Export all results to files
        
        Args:
            results: Analysis results
            output_dir: Output directory
            formats: Export formats
            
        Returns:
            Dictionary of exported file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # Export symbols
        if results.get('symbols') and 'json' in formats:
            symbols_file = output_path / 'symbols.json'
            exported_files['symbols_json'] = self.yoloplan.export_results(
                results['symbols'],
                format='json',
                output_path=str(symbols_file)
            )
        
        if results.get('symbols') and 'csv' in formats:
            symbols_csv = output_path / 'symbols.csv'
            exported_files['symbols_csv'] = self.yoloplan.export_results(
                results['symbols'],
                format='csv',
                output_path=str(symbols_csv)
            )
        
        # Export BOM
        if results.get('bom') and 'csv' in formats:
            bom_file = output_path / 'bom.csv'
            exported_files['bom_csv'] = self.bom_generator.export_bom(
                results['bom'],
                format='csv',
                output_path=str(bom_file)
            )
        
        if results.get('bom') and 'json' in formats:
            bom_json = output_path / 'bom.json'
            exported_files['bom_json'] = self.bom_generator.export_bom(
                results['bom'],
                format='json',
                output_path=str(bom_json)
            )
        
        # Export connectivity
        if results.get('connectivity') and 'json' in formats:
            import json
            conn_file = output_path / 'connectivity.json'
            with open(conn_file, 'w') as f:
                json.dump(results['connectivity'], f, indent=2)
            exported_files['connectivity_json'] = str(conn_file)
        
        # Export summary
        if results.get('summary') and 'json' in formats:
            import json
            summary_file = output_path / 'summary.json'
            with open(summary_file, 'w') as f:
                json.dump(results['summary'], f, indent=2)
            exported_files['summary_json'] = str(summary_file)
        
        logger.info(f"Exported {len(exported_files)} files to {output_dir}")
        
        return exported_files
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        summary = {
            'analysis_complete': True,
            'stages_completed': []
        }
        
        if results.get('document'):
            summary['stages_completed'].append('document_processing')
            summary['quality_score'] = results['document'].get('quality_info', {}).get('quality_score', 0)
        
        if results.get('text'):
            summary['stages_completed'].append('text_extraction')
            summary['text_elements'] = len(results['text'].get('results', []))
        
        if results.get('symbols'):
            summary['stages_completed'].append('symbol_detection')
            summary['total_symbols'] = results['symbols']['total_symbols']
            summary['symbol_types'] = list(results['symbols']['counts'].keys())
        
        if results.get('components'):
            summary['stages_completed'].append('component_detection')
            summary['total_components'] = len(results['components'])
        
        if results.get('bom'):
            summary['stages_completed'].append('bom_generation')
            summary['bom_items'] = len(results['bom'])
            summary['total_estimated_cost'] = sum(
                item.estimated_cost for item in results['bom'] 
                if item.estimated_cost is not None
            )
        
        if results.get('connectivity'):
            summary['stages_completed'].append('connectivity_analysis')
            summary['connections'] = len(results['connectivity']['connections'])
            summary['circuits'] = len(results['connectivity']['circuits'])
        
        return summary
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate batch processing summary"""
        total_symbols = sum(r.get('metadata', {}).get('total_symbols', 0) for r in results)
        total_components = sum(r.get('metadata', {}).get('total_components', 0) for r in results)
        total_bom_items = sum(r.get('metadata', {}).get('bom_items', 0) for r in results)
        
        return {
            'total_blueprints': len(results),
            'total_symbols_detected': total_symbols,
            'total_components_detected': total_components,
            'total_bom_items': total_bom_items,
            'average_symbols_per_blueprint': total_symbols / len(results) if results else 0,
            'average_components_per_blueprint': total_components / len(results) if results else 0
        }


def create_integrated_detector(**kwargs) -> IntegratedHVACDetector:
    """
    Factory function to create integrated detector
    
    Args:
        **kwargs: Arguments passed to IntegratedHVACDetector
        
    Returns:
        Configured integrated detector
    """
    return IntegratedHVACDetector(**kwargs)


# Example usage
if __name__ == "__main__":
    """
    Example usage of integrated detector:
    
    from core.ai.integrated_detector import create_integrated_detector
    
    # Initialize
    detector = create_integrated_detector()
    
    # Analyze single blueprint
    image = cv2.imread('hvac_blueprint.png')
    results = detector.analyze_blueprint(image)
    
    # View results
    print(f"Analysis Summary:")
    print(f"  Symbols: {results['metadata']['total_symbols']}")
    print(f"  Components: {results['metadata']['total_components']}")
    print(f"  BOM Items: {results['metadata']['bom_items']}")
    print(f"  Connections: {results['metadata']['connections']}")
    
    # Export results
    exported = detector.export_results(results, output_dir='output')
    print(f"Exported files: {list(exported.keys())}")
    
    # Batch processing
    images = [cv2.imread(f'blueprint_{i}.png') for i in range(10)]
    batch_results = detector.batch_analyze(images)
    print(f"Processed {batch_results['total_blueprints']} blueprints")
    """
    print("Integrated HVAC Detector - Complete YOLOplan Integration (Weeks 1-16)")
    print("See docs/YOLOPLAN_INTEGRATION.md for usage guide")
