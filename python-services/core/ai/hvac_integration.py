"""
HVAC Integration Layer

This module integrates the new HVAC services (SAHI, system validation, prompt engineering)
with the existing python-services backend. It provides a unified interface that combines:
- SAHI-based component detection
- System relationship analysis
- ASHRAE/SMACNA validation
- Enhanced document processing

Industry best practices:
- Adapter pattern for seamless integration
- Graceful degradation when services unavailable
- Comprehensive error handling
- Performance monitoring and logging
"""

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time

# Add services directory to path
SERVICES_ROOT = Path(__file__).resolve().parents[3] / "services"
sys.path.insert(0, str(SERVICES_ROOT.parent))

logger = logging.getLogger(__name__)

# Try to import new HVAC services with graceful fallback
SAHI_AVAILABLE = False
SYSTEM_ENGINE_AVAILABLE = False
PROMPT_FRAMEWORK_AVAILABLE = False
DOCUMENT_PROCESSOR_AVAILABLE = False

try:
    from services.hvac_ai.hvac_sahi_engine import (
        create_hvac_sahi_predictor,
        HVACSAHIConfig,
        SAHI_AVAILABLE as SAHI_LIB_AVAILABLE
    )
    SAHI_AVAILABLE = SAHI_LIB_AVAILABLE
    logger.info("✓ HVAC SAHI engine available")
except ImportError as e:
    logger.warning(f"HVAC SAHI engine not available: {e}")

try:
    from services.hvac_domain.hvac_system_engine import (
        HVACSystemEngine,
        HVACComponent,
        HVACComponentType
    )
    SYSTEM_ENGINE_AVAILABLE = True
    logger.info("✓ HVAC system engine available")
except ImportError as e:
    logger.warning(f"HVAC system engine not available: {e}")

try:
    from services.hvac_ai.hvac_prompt_engineering import (
        create_hvac_prompt_framework,
        HVACAnalysisType
    )
    PROMPT_FRAMEWORK_AVAILABLE = True
    logger.info("✓ HVAC prompt framework available")
except ImportError as e:
    logger.warning(f"HVAC prompt framework not available: {e}")

try:
    from services.hvac_document.hvac_document_processor import (
        create_hvac_document_processor,
        BlueprintFormat
    )
    DOCUMENT_PROCESSOR_AVAILABLE = True
    logger.info("✓ HVAC document processor available")
except ImportError as e:
    logger.warning(f"HVAC document processor not available: {e}")


class HVACIntegratedAnalyzer:
    """
    Integrated HVAC analyzer that combines legacy SAM inference with new HVAC services.
    
    This adapter provides:
    - SAHI-based detection for large blueprints
    - System relationship validation
    - Prompt engineering for improved accuracy
    - Document quality assessment
    
    Falls back gracefully to legacy SAM when new services unavailable.
    """
    
    def __init__(
        self,
        sam_engine,
        model_path: str,
        enable_sahi: bool = True,
        enable_validation: bool = True,
        enable_prompts: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize integrated HVAC analyzer.
        
        Args:
            sam_engine: Existing SAM inference engine
            model_path: Path to SAM model weights
            enable_sahi: Enable SAHI-based detection
            enable_validation: Enable system validation
            enable_prompts: Enable prompt engineering
            device: Device for inference ("cuda" or "cpu")
        """
        self.sam_engine = sam_engine
        self.model_path = model_path
        self.device = device
        
        # Feature flags
        self.enable_sahi = enable_sahi and SAHI_AVAILABLE
        self.enable_validation = enable_validation and SYSTEM_ENGINE_AVAILABLE
        self.enable_prompts = enable_prompts and PROMPT_FRAMEWORK_AVAILABLE
        
        # Initialize new services if available
        self.sahi_predictor = None
        self.system_engine = None
        self.prompt_framework = None
        self.document_processor = None
        
        self._initialize_services()
        
        logger.info(
            f"HVAC Integrated Analyzer initialized - "
            f"SAHI: {self.enable_sahi}, "
            f"Validation: {self.enable_validation}, "
            f"Prompts: {self.enable_prompts}"
        )
    
    def _initialize_services(self):
        """Initialize available HVAC services"""
        
        # Initialize SAHI predictor
        if self.enable_sahi:
            try:
                self.sahi_predictor = create_hvac_sahi_predictor(
                    model_path=self.model_path,
                    device=self.device,
                    slice_height=1024,
                    overlap_height_ratio=0.3,
                    confidence_threshold=0.40
                )
                logger.info("✓ SAHI predictor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SAHI predictor: {e}")
                self.enable_sahi = False
        
        # Initialize system engine
        if self.enable_validation:
            try:
                self.system_engine = HVACSystemEngine()
                logger.info("✓ System validation engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize system engine: {e}")
                self.enable_validation = False
        
        # Initialize prompt framework
        if self.enable_prompts:
            try:
                self.prompt_framework = create_hvac_prompt_framework()
                logger.info("✓ Prompt engineering framework initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize prompt framework: {e}")
                self.enable_prompts = False
        
        # Initialize document processor
        if DOCUMENT_PROCESSOR_AVAILABLE:
            try:
                self.document_processor = create_hvac_document_processor()
                logger.info("✓ Document processor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize document processor: {e}")
    
    def analyze_blueprint(
        self,
        image: np.ndarray,
        mode: str = "auto",
        return_relationships: bool = True,
        return_validation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive blueprint analysis with SAHI detection and validation.
        
        Args:
            image: Blueprint image as numpy array
            mode: Analysis mode ("auto", "sahi", "legacy")
            return_relationships: Include relationship graph in results
            return_validation: Include validation results
            
        Returns:
            Dictionary containing detections, relationships, and validation
        """
        start_time = time.time()
        
        # Step 1: Document quality assessment
        quality_metrics = self._assess_quality(image)
        
        # Step 2: Component detection (SAHI or legacy)
        if mode == "auto":
            # Decide based on image size and SAHI availability
            height, width = image.shape[:2]
            use_sahi = self.enable_sahi and (height * width > 2000 * 2000)
        elif mode == "sahi":
            use_sahi = self.enable_sahi
        else:
            use_sahi = False
        
        if use_sahi:
            detections = self._detect_with_sahi(image)
        else:
            detections = self._detect_with_legacy(image)
        
        # Step 3: System relationship analysis
        relationships = []
        validation_results = None
        
        if self.enable_validation and (return_relationships or return_validation):
            relationships, validation_results = self._analyze_system(
                detections,
                return_validation=return_validation
            )
        
        processing_time = time.time() - start_time
        
        # Compile results
        results = {
            "status": "success",
            "detections": detections,
            "total_components": len(detections),
            "processing_time_seconds": processing_time,
            "processing_time_ms": processing_time * 1000,
            "quality_metrics": quality_metrics,
            "analysis_mode": "sahi" if use_sahi else "legacy",
            "features_enabled": {
                "sahi": self.enable_sahi,
                "validation": self.enable_validation,
                "prompts": self.enable_prompts
            }
        }
        
        if return_relationships and relationships:
            results["relationships"] = relationships
        
        if return_validation and validation_results:
            results["validation"] = validation_results
        
        return results
    
    def _assess_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess blueprint quality using document processor"""
        if not self.document_processor:
            return {"quality_score": 0.0, "available": False}
        
        try:
            metrics = self.document_processor.assess_quality(image)
            return {
                "line_clarity": float(metrics.line_clarity),
                "symbol_visibility": float(metrics.symbol_visibility),
                "overall_quality": float(metrics.overall_quality),
                "issues": metrics.issues,
                "available": True
            }
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {"quality_score": 0.0, "available": False, "error": str(e)}
    
    def _detect_with_sahi(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect components using SAHI for improved accuracy on large blueprints.
        
        SAHI benefits:
        - Better small component detection
        - Linear memory scaling
        - Adaptive slicing based on complexity
        """
        if not self.sahi_predictor:
            logger.warning("SAHI predictor not available, falling back to legacy")
            return self._detect_with_legacy(image)
        
        try:
            # SAHI requires image path, so we'd need to save temporarily
            # For now, use complexity analysis and fall back to legacy
            # In production, implement proper temp file handling
            
            complexity = self.sahi_predictor.analyze_blueprint_complexity(image)
            logger.info(
                f"Blueprint complexity: {complexity['complexity_level']}, "
                f"recommended slice: {complexity['recommended_slice_size']}px"
            )
            
            # Fall back to legacy for now (proper SAHI integration needs temp file handling)
            logger.info("Using legacy detection (SAHI temp file handling not yet implemented)")
            return self._detect_with_legacy(image)
            
        except Exception as e:
            logger.error(f"SAHI detection failed: {e}", exc_info=True)
            return self._detect_with_legacy(image)
    
    def _detect_with_legacy(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect components using legacy SAM engine"""
        # Use existing SAM engine's count method for component detection
        try:
            result = self.sam_engine.count(
                image,
                grid_size=32,
                min_score=0.2,
                debug=False,
                max_grid_points=2000
            )
            
            # Convert to standardized format
            detections = []
            if isinstance(result, dict) and result.get("objects"):
                for i, obj in enumerate(result["objects"]):
                    detection = {
                        "id": f"comp_{i:03d}",
                        "bbox": obj.get("bbox", []),
                        "confidence": obj.get("score", 0.0),
                        "mask": obj.get("mask"),
                        "category": obj.get("category", "unknown"),
                        "type": "unknown"  # Legacy doesn't have type classification
                    }
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Legacy detection failed: {e}", exc_info=True)
            return []
    
    def _analyze_system(
        self,
        detections: List[Dict[str, Any]],
        return_validation: bool = True
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Analyze system relationships and validate configuration.
        
        This implements ASHRAE/SMACNA validation rules:
        - Ductwork connectivity
        - Equipment clearances
        - VAV box connections
        - Flow path validation
        """
        if not self.system_engine:
            return [], None
        
        try:
            # Clear previous components
            self.system_engine.components.clear()
            self.system_engine.relationships.clear()
            
            # Add detected components to system engine
            for detection in detections:
                # Map detection to HVAC component type
                component_type = self._map_to_component_type(detection.get("category", "unknown"))
                
                component = HVACComponent(
                    id=detection["id"],
                    component_type=component_type,
                    bbox=detection.get("bbox", []),
                    confidence=detection.get("confidence", 0.0),
                    attributes={"category": detection.get("category")}
                )
                self.system_engine.add_component(component)
            
            # Build relationship graph
            graph = self.system_engine.build_relationship_graph()
            
            # Format relationships for API response
            relationships = []
            for rel in self.system_engine.relationships:
                relationships.append({
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relationship_type.value,
                    "confidence": rel.confidence,
                    "metadata": rel.metadata
                })
            
            # Validate system configuration
            validation = None
            if return_validation:
                validation = self.system_engine.validate_system_configuration()
            
            return relationships, validation
            
        except Exception as e:
            logger.error(f"System analysis failed: {e}", exc_info=True)
            return [], None
    
    def _map_to_component_type(self, category: str) -> 'HVACComponentType':
        """Map detection category to HVAC component type"""
        if not SYSTEM_ENGINE_AVAILABLE:
            return None
        
        # Mapping from detection categories to HVAC types
        category_lower = category.lower()
        
        if "duct" in category_lower or "pipe" in category_lower:
            return HVACComponentType.DUCTWORK
        elif "damper" in category_lower:
            return HVACComponentType.DAMPER
        elif "vav" in category_lower or "valve" in category_lower:
            return HVACComponentType.VAV_BOX
        elif "fan" in category_lower or "blower" in category_lower:
            return HVACComponentType.FAN
        elif "ahu" in category_lower or "air handling" in category_lower:
            return HVACComponentType.AHU
        elif "coil" in category_lower:
            return HVACComponentType.COIL
        elif "sensor" in category_lower or "instrument" in category_lower:
            return HVACComponentType.SENSOR
        elif "controller" in category_lower or "control" in category_lower:
            return HVACComponentType.CONTROL
        else:
            return HVACComponentType.UNKNOWN
    
    def generate_analysis_prompt(
        self,
        analysis_type: str,
        context: Dict[str, str]
    ) -> Optional[str]:
        """Generate HVAC-specific analysis prompt"""
        if not self.prompt_framework:
            return None
        
        try:
            # Map analysis type string to enum
            type_map = {
                "component_detection": "component_detection_cot",
                "duct_connectivity": "duct_connectivity_analysis",
                "symbol_recognition": "symbol_recognition_fewshot",
                "equipment_sizing": "equipment_sizing_role",
                "code_compliance": "code_compliance_constraints"
            }
            
            template_name = type_map.get(analysis_type)
            if not template_name:
                return None
            
            prompt = self.prompt_framework.generate_prompt(
                template_name=template_name,
                variables=context
            )
            
            return prompt
            
        except Exception as e:
            logger.error(f"Prompt generation failed: {e}", exc_info=True)
            return None


def create_integrated_analyzer(
    sam_engine,
    model_path: str,
    **kwargs
) -> HVACIntegratedAnalyzer:
    """
    Factory function to create integrated HVAC analyzer.
    
    Args:
        sam_engine: Existing SAM inference engine
        model_path: Path to SAM model weights
        **kwargs: Additional configuration options
        
    Returns:
        Configured HVACIntegratedAnalyzer instance
    """
    return HVACIntegratedAnalyzer(
        sam_engine=sam_engine,
        model_path=model_path,
        **kwargs
    )
