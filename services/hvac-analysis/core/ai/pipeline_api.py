"""
FastAPI integration for the HVAC Drawing Analysis Pipeline.
Provides REST API endpoints for end-to-end analysis.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import logging
import io
import numpy as np
from PIL import Image
import tempfile
import os

from .hvac_pipeline import create_hvac_analyzer, HVACDrawingAnalyzer
from .pipeline_models import PipelineConfig, HVACResult

logger = logging.getLogger(__name__)

# Global analyzer instance (initialized on startup)
_analyzer: Optional[HVACDrawingAnalyzer] = None

# Create router
router = APIRouter(prefix="/api/v1/pipeline", tags=["HVAC Pipeline"])


def initialize_pipeline(model_path: str, config: Optional[PipelineConfig] = None):
    """
    Initialize the HVAC pipeline analyzer.
    Should be called during application startup.
    
    Args:
        model_path: Path to YOLOv11-obb model
        config: Optional pipeline configuration
    """
    global _analyzer
    
    try:
        logger.info("🚀 Initializing HVAC Drawing Analysis Pipeline...")
        _analyzer = create_hvac_analyzer(
            model_path=model_path,
            config=config
        )
        logger.info("✅ HVAC Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize HVAC pipeline: {e}", exc_info=True)
        _analyzer = None
        raise


def get_analyzer() -> HVACDrawingAnalyzer:
    """Get the initialized analyzer instance."""
    if _analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="HVAC Pipeline not initialized. Check server logs."
        )
    return _analyzer


@router.get("/health")
async def pipeline_health_check():
    """
    Check pipeline health status.
    
    Returns health information for all pipeline components.
    """
    try:
        analyzer = get_analyzer()
        health = analyzer.health_check()
        return JSONResponse(content=health)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "unhealthy",
                "error": e.detail
            }
        )


@router.get("/stats")
async def pipeline_statistics():
    """
    Get pipeline performance statistics.
    
    Returns cumulative statistics about pipeline performance.
    """
    try:
        analyzer = get_analyzer()
        stats = analyzer.get_statistics()
        return JSONResponse(content=stats)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )


@router.post("/analyze")
async def analyze_drawing(
    image: UploadFile = File(..., description="HVAC drawing image file"),
    confidence_threshold: float = Form(0.7, description="Detection confidence threshold (0.0-1.0)"),
    max_processing_time_ms: float = Form(25.0, description="Maximum processing time in milliseconds")
):
    """
    Analyze HVAC drawing with end-to-end pipeline.
    
    Performs:
    1. Component & text region detection (YOLOv11-obb)
    2. Text recognition (EasyOCR)
    3. HVAC semantic interpretation
    
    Args:
        image: HVAC drawing image file (PNG, JPG, etc.)
        confidence_threshold: Confidence threshold for detections (default: 0.7)
        max_processing_time_ms: Maximum total processing time (default: 25ms)
        
    Returns:
        Complete analysis results including detections, recognized text, and interpretations
    """
    logger.info(f"📡 [Pipeline API] Received analyze request: {image.filename}")
    
    try:
        analyzer = get_analyzer()
    except HTTPException as e:
        raise
    
    # Validate parameters
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=400,
            detail=f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
        )
    
    if max_processing_time_ms <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"max_processing_time_ms must be positive, got {max_processing_time_ms}"
        )
    
    try:
        # Read uploaded image
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Convert to numpy array
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_np = np.array(pil_image)
        except Exception as e:
            logger.error(f"Failed to process image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        logger.info(f"📷 [Pipeline API] Image loaded: {image_np.shape}")
        
        # Save to temporary file (pipeline expects file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_path = tmp_file.name
            pil_image.save(tmp_path)
        
        try:
            # Update pipeline config for this request
            config = PipelineConfig(
                confidence_threshold=confidence_threshold,
                max_processing_time_ms=max_processing_time_ms
            )
            
            # Create analyzer with custom config
            analyzer = create_hvac_analyzer(
                model_path=analyzer.yolo_model_path,
                config=config,
                device=analyzer.device
            )
            
            # Run end-to-end analysis
            result: HVACResult = analyzer.analyze_drawing(tmp_path)
            
            # Convert to dict for JSON response
            result_dict = result.model_dump(mode='json')
            
            logger.info(f"✅ [Pipeline API] Analysis complete: {result.success}")
            logger.info(f"   Total time: {result.total_processing_time_ms:.2f}ms")
            logger.info(f"   Detections: {len(result.detection_result.detections) if result.detection_result else 0}")
            logger.info(f"   Text results: {len(result.text_results)}")
            logger.info(f"   Interpretations: {len(result.interpretation_result.interpretations) if result.interpretation_result else 0}")
            
            return JSONResponse(content={
                "status": "success" if result.success else "partial_success" if result.partial_success else "failed",
                **result_dict
            })
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [Pipeline API] Analysis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/analyze/batch")
async def analyze_drawings_batch(
    images: list[UploadFile] = File(..., description="Multiple HVAC drawing images"),
    confidence_threshold: float = Form(0.7, description="Detection confidence threshold"),
):
    """
    Analyze multiple HVAC drawings in batch.
    
    Processes multiple images with the same configuration.
    Results are returned in the same order as input images.
    
    Args:
        images: List of HVAC drawing image files
        confidence_threshold: Confidence threshold for all images
        
    Returns:
        List of analysis results for each image
    """
    logger.info(f"📡 [Pipeline API] Received batch analyze request: {len(images)} images")
    
    try:
        analyzer = get_analyzer()
    except HTTPException as e:
        raise
    
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")
    
    if len(images) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too many images. Maximum 10 images per batch, got {len(images)}"
        )
    
    results = []
    
    for idx, image in enumerate(images):
        logger.info(f"📷 [Pipeline API] Processing image {idx+1}/{len(images)}: {image.filename}")
        
        try:
            # Read and process image
            contents = await image.read()
            if not contents:
                results.append({
                    "status": "failed",
                    "filename": image.filename,
                    "error": "Empty file"
                })
                continue
            
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_path = tmp_file.name
                pil_image.save(tmp_path)
            
            try:
                # Analyze
                result = analyzer.analyze_drawing(tmp_path)
                result_dict = result.model_dump(mode='json')
                
                results.append({
                    "status": "success" if result.success else "partial_success",
                    "filename": image.filename,
                    **result_dict
                })
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"Failed to process image {image.filename}: {e}")
            results.append({
                "status": "failed",
                "filename": image.filename,
                "error": str(e)
            })
    
    logger.info(f"✅ [Pipeline API] Batch analysis complete: {len(results)} results")
    
    return JSONResponse(content={
        "status": "success",
        "total_images": len(images),
        "results": results
    })
