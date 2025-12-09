"""
HVAC AI Platform - Core Analysis Service
Intelligent blueprint analysis engine for HVAC systems
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
import os
import tempfile
import asyncio
from datetime import datetime
import json
import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HVAC AI Platform",
    description="AI-Powered HVAC Blueprint Analysis Service",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enums for data validation
class FileFormat(str, Enum):
    PDF = "pdf"
    DWG = "dwg"
    DXF = "dxf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"

class ComponentType(str, Enum):
    HVAC_UNIT = "hvac_unit"
    DUCT = "duct"
    VAV_BOX = "vav_box"
    DIFFUSER = "diffuser"
    THERMOSTAT = "thermostat"
    DAMPER = "damper"
    FAN = "fan"
    COIL = "coil"
    FILTER = "filter"
    PIPE = "pipe"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request/Response Models
class BlueprintAnalysisRequest(BaseModel):
    """Request model for blueprint analysis"""
    project_id: str = Field(..., description="Project identifier")
    location: Optional[str] = Field(None, description="Project location for regional rules")
    climate_zone: Optional[str] = Field(None, description="Climate zone identifier")
    
class ComponentDetection(BaseModel):
    """Detected HVAC component"""
    component_id: str
    component_type: ComponentType
    confidence: float = Field(..., ge=0.0, le=1.0)
    bounding_box: Dict[str, float]
    dimensions: Optional[Dict[str, float]] = None
    specifications: Optional[Dict[str, Any]] = None

class BlueprintAnalysisResponse(BaseModel):
    """Response model for blueprint analysis"""
    analysis_id: str
    status: AnalysisStatus
    file_name: str
    file_format: FileFormat
    detected_components: List[ComponentDetection]
    scale_factor: Optional[float] = None
    total_components: int
    processing_time_seconds: float
    metadata: Dict[str, Any]

class EstimationRequest(BaseModel):
    """Request model for cost/labor estimation"""
    analysis_id: str
    location: str
    labor_rate: Optional[float] = Field(None, description="Override labor rate per hour")
    
class EstimationResponse(BaseModel):
    """Response model for cost estimation"""
    estimation_id: str
    analysis_id: str
    material_costs: Dict[str, float]
    labor_hours: Dict[str, float]
    total_cost: float
    regional_adjustments: Dict[str, Any]
    compliance_notes: List[str]

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]

# SAM-specific models
class SegmentRequest(BaseModel):
    """Request model for SAM segmentation"""
    prompt: str = Field(..., description="JSON string with prompt details")

class SegmentationResult(BaseModel):
    """Single segmentation result"""
    label: str
    score: float
    mask: str  # Base64 encoded RLE
    bbox: List[int]  # [x, y, width, height]

class SegmentResponse(BaseModel):
    """Response model for segmentation"""
    status: str
    segments: List[SegmentationResult]

class CountResponse(BaseModel):
    """Response model for component counting"""
    status: str
    total_objects_found: int
    counts_by_category: Dict[str, int]

# In-memory storage for demo (replace with database in production)
analysis_results = {}
estimation_results = {}

# Import analysis modules (to be implemented)
try:
    from .core import document_processor
    from .core import ai_engine
    from .core import location_intelligence
    from .core import estimation_engine
    MODULES_LOADED = True
except ImportError:
    logger.warning("Analysis modules not yet implemented. Using mock responses.")
    MODULES_LOADED = False

# Initialize SAM engine
SAM_ENGINE = None
try:
    from core.ai.sam_inference import create_sam_engine
    SAM_ENGINE = create_sam_engine()
    logger.info("SAM inference engine initialized successfully")
except ImportError as e:
    logger.warning(f"SAM dependencies not installed: {e}. Using mock mode.")
except FileNotFoundError as e:
    logger.warning(f"SAM model file not found: {e}. Using mock mode.")
except Exception as e:
    logger.error(f"SAM engine initialization failed with unexpected error: {e}. Using mock mode.")
    import traceback
    logger.debug(traceback.format_exc())

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "HVAC AI Platform - Blueprint Analysis Service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="0.1.0",
        services={
            "document_processor": "operational" if MODULES_LOADED else "not_loaded",
            "ai_engine": "operational" if MODULES_LOADED else "not_loaded",
            "location_intelligence": "operational" if MODULES_LOADED else "not_loaded",
            "estimation_engine": "operational" if MODULES_LOADED else "not_loaded"
        }
    )

@app.post("/api/analyze/blueprint", response_model=BlueprintAnalysisResponse)
async def analyze_blueprint(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    project_id: str = "default",
    location: Optional[str] = None
):
    """
    Analyze uploaded HVAC blueprint
    
    This endpoint processes blueprint files (PDF, DWG, images) and performs:
    - Document format detection and conversion
    - Scale and dimension detection
    - HVAC component recognition with AI
    - Spatial relationship analysis
    - Specification extraction
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate file format
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in [f.value for f in FileFormat]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Generate analysis ID
        analysis_id = f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Mock analysis (replace with actual implementation)
        logger.info(f"Starting blueprint analysis: {analysis_id}")
        
        # Simulate processing
        await asyncio.sleep(0.5)  # Remove in production
        
        # Mock detected components
        mock_components = [
            ComponentDetection(
                component_id="comp_001",
                component_type=ComponentType.HVAC_UNIT,
                confidence=0.95,
                bounding_box={"x": 100, "y": 200, "width": 150, "height": 200},
                dimensions={"length": 48, "width": 36, "height": 60},
                specifications={"capacity_tons": 5, "efficiency": "16 SEER"}
            ),
            ComponentDetection(
                component_id="comp_002",
                component_type=ComponentType.DUCT,
                confidence=0.88,
                bounding_box={"x": 300, "y": 150, "width": 400, "height": 50},
                dimensions={"length": 240, "diameter": 12}
            ),
        ]
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = BlueprintAnalysisResponse(
            analysis_id=analysis_id,
            status=AnalysisStatus.COMPLETED,
            file_name=file.filename,
            file_format=FileFormat(file_ext),
            detected_components=mock_components,
            scale_factor=0.25,  # 1/4" = 1'
            total_components=len(mock_components),
            processing_time_seconds=processing_time,
            metadata={
                "project_id": project_id,
                "location": location,
                "upload_timestamp": start_time.isoformat(),
                "file_size_bytes": len(content)
            }
        )
        
        # Store results
        analysis_results[analysis_id] = response
        
        # Cleanup
        os.unlink(tmp_path)
        
        return response
        
    except Exception as e:
        logger.error(f"Blueprint analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/api/analyze/{analysis_id}", response_model=BlueprintAnalysisResponse)
async def get_analysis_result(analysis_id: str):
    """Retrieve analysis results by ID"""
    if analysis_id not in analysis_results:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis_results[analysis_id]

@app.post("/api/estimate", response_model=EstimationResponse)
async def create_estimation(request: EstimationRequest):
    """
    Generate cost and labor estimation
    
    This endpoint provides:
    - Material quantity calculations
    - Labor hour estimations
    - Location-adjusted costs
    - Building code compliance checking
    """
    try:
        # Verify analysis exists
        if request.analysis_id not in analysis_results:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        analysis = analysis_results[request.analysis_id]
        
        # Generate estimation ID
        estimation_id = f"est_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Mock estimation (replace with actual implementation)
        logger.info(f"Generating estimation: {estimation_id}")
        
        response = EstimationResponse(
            estimation_id=estimation_id,
            analysis_id=request.analysis_id,
            material_costs={
                "hvac_units": 8500.00,
                "ductwork": 3200.00,
                "vav_boxes": 4500.00,
                "controls": 2100.00,
                "miscellaneous": 1200.00
            },
            labor_hours={
                "installation": 120.0,
                "electrical": 40.0,
                "testing": 16.0,
                "commissioning": 8.0
            },
            total_cost=35720.00,
            regional_adjustments={
                "location": request.location,
                "labor_multiplier": 1.15,
                "material_multiplier": 1.08
            },
            compliance_notes=[
                "System meets ASHRAE 90.1 requirements",
                "Verify local permit requirements",
                "Manual J calculation recommended"
            ]
        )
        
        # Store results
        estimation_results[estimation_id] = response
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Estimation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Estimation failed: {str(e)}")

@app.get("/api/estimate/{estimation_id}", response_model=EstimationResponse)
async def get_estimation(estimation_id: str):
    """Retrieve estimation by ID"""
    if estimation_id not in estimation_results:
        raise HTTPException(status_code=404, detail="Estimation not found")
    return estimation_results[estimation_id]

@app.get("/api/components/types")
async def get_component_types():
    """Get list of supported HVAC component types"""
    return {
        "component_types": [
            {"value": ct.value, "label": ct.value.replace("_", " ").title()}
            for ct in ComponentType
        ]
    }

@app.get("/api/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "formats": [
            {"value": fmt.value, "label": fmt.value.upper()}
            for fmt in FileFormat
        ]
    }

@app.post("/api/v1/segment", response_model=SegmentResponse)
async def segment_component(
    image: UploadFile = File(...),
    prompt: str = Form(...)
):
    """
    Interactive segmentation endpoint
    
    Segments a single component based on user prompt (e.g., point click).
    Returns segmentation mask, label, and bounding box.
    
    Args:
        image: Uploaded P&ID or HVAC diagram image
        prompt: JSON string with prompt details
               Example: {"type": "point", "data": {"coords": [452, 312], "label": 1}}
    
    Returns:
        Segmentation result with mask, label, score, and bbox
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Parse prompt
        try:
            prompt_dict = json.loads(prompt)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid prompt JSON")
        
        # Run segmentation
        if SAM_ENGINE is None:
            raise HTTPException(status_code=503, detail="SAM engine not available")
        
        results = SAM_ENGINE.segment(img, prompt_dict)
        
        # Format response
        segments = [
            SegmentationResult(
                label=r.label,
                score=r.score,
                mask=r.mask,
                bbox=r.bbox
            )
            for r in results
        ]
        
        return SegmentResponse(
            status="success",
            segments=segments
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")

@app.post("/api/v1/count", response_model=CountResponse)
async def count_components(
    image: UploadFile = File(...)
):
    """
    Automated component counting endpoint
    
    Analyzes entire diagram to identify, classify, and count all recognized components.
    Uses grid-based prompting with Non-Maximum Suppression for de-duplication.
    
    Args:
        image: Uploaded P&ID or HVAC diagram image
    
    Returns:
        Total count and breakdown by component category
    """
    try:
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run counting
        if SAM_ENGINE is None:
            raise HTTPException(status_code=503, detail="SAM engine not available")
        
        result = SAM_ENGINE.count(img)
        
        return CountResponse(
            status="success",
            total_objects_found=result.total_objects_found,
            counts_by_category=result.counts_by_category
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Counting failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Counting failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hvac_analysis_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
