Here is the updated, fully comprehensive PR / Technical Specification Document.

I have added a strict Global Coding Standard section to enforce universal naming conventions and a mandatory Proof of Completion section to ensure the work is validated visually and metrically before merging.

📋 Epic: The "HVAC Cortex" Infrastructure & OCR Pipeline
PR Type: System Architecture Overhaul
Priority: Critical (P0)
Target Infrastructure: Ray Serve (Local GPU 
→
→
Cloud Cluster)

1. Global Directives: Code Standards & Naming
To ensure long-term maintainability and ease of onboarding, all new code must adhere to these strict naming conventions.

🔍 Universal & Reusable Terminology
Do not use tool-specific or brand-specific names in class/file definitions. Use Domain-Driven Design (DDD) terminology.

❌ Avoid (Implementation Details): YoloService.py, PaddleOCRWrapper.py, DeepZoomInferenceAnalysis.tsx, ray_worker.py.
✅ Use (Universal Functions):
ObjectDetector (The generic role of finding things).
TextExtractor (The generic role of reading text).
BlueprintViewer (The generic UI component).
InferenceGraph (The orchestration layer).
GeometryUtils (Math helpers).
Why? If we switch from YOLO to EfficientDet, or Paddle to GPT-4V, we shouldn't have to rename our entire codebase.

2. Executive Summary
We are migrating from a linear FastAPI script to a Distributed Inference Graph. The current monolithic approach cannot scale to support the multi-model architecture required for our roadmap.

The Objective:
Implement a Neuro-Symbolic AI Pipeline using Ray Serve. This pipeline will chain an Object Detector (Vision) and a Text Extractor (Language) into a single, low-latency API. The system must intelligently "handshake" between models—detecting components, geometrically correcting their orientation, and reading their text tags—before returning a unified data payload.

3. Architecture: The Inference Graph
Implement a Directed Acyclic Graph (DAG) of independent deployments.

The Data Flow:

Ingress (API Gateway): Receives HTTP POST, decodes images.
Node A (Object Detector): Detects component geometry (OBB).
Node B (Geometry Engine): Maps coordinates to the original high-res image, performs perspective transforms to "un-rotate" regions of interest.
Node C (Text Extractor): Reads text from the straightened crops.
Node D (Fusion Layer): Merges spatial data and text data into a single JSON response.
🛠️ Track A: Backend Infrastructure (Ray Serve)
Objective: Establish the runtime environment.

Task 1.1: Inference Graph Orchestration
File: core/inference_graph.py
Implementation: Define the Ray Serve deployment graph.
Resource Strategy: Implement Fractional GPU Allocation for the GTX 1070 (8GB).
ObjectDetector: ~40% VRAM.
TextExtractor: ~30% VRAM.
Concurrency: Ensure the Ingress node handles requests asynchronously (non-blocking).
Task 1.2: The ObjectDetector Service
Implementation: Wrap the YOLOv11 logic.
Requirement: Load the model once during __init__.
Output: Return raw OBB data (center coordinates, width, height, rotation).
Task 1.3: The TextExtractor Service
Implementation: Wrap the PaddleOCR logic.
Requirement: Initialize with use_angle_cls=False (we handle rotation manually via geometry).
Optimization: Support batch processing (accept a list of crops).
🧠 Track B: The Intelligence Logic (The "Handshake")
Objective: Connect Vision and Language intelligently.

Task 2.1: GeometryUtils Module
File: core/utils/geometry.py
Logic: Implement the Perspective Transform Pipeline:
Accept OBB parameters (x, y, w, h, rotation) + Original Image.
Calculate the 4 corner points.
Warp/Rotate the crop to be perfectly horizontal (0 degrees).
Apply grayscale/thresholding for OCR contrast enhancement.
Task 2.2: Selective Inference Logic
Action: Implement filtering in the Fusion Layer.
Requirement: Define TEXT_RICH_CLASSES (e.g., tag_number, id_letters). Only trigger the TextExtractor node if the detected class matches this list.
🖥️ Track C: Frontend Integration (Next.js)
Objective: Visualize the multi-modal data.

Task 3.1: Universal Data Contract
File: src/types/domain.ts (Rename from analysis.ts if needed).
Requirement: Update the Segment interface:
textContent (string): The extracted text.
textConfidence (number): The OCR score.
Task 3.2: BlueprintViewer Updates
File: src/components/viewer/BlueprintViewer.tsx
Logic: Update the renderAnnotations loop.
If textContent exists, render it preferentially over the class label.
Style: High-contrast background, monospace font (to signify "Read Data").
🔌 Track D: DevOps & Wiring
Objective: One-click startup.

Task 4.1: Unified Startup Script
File: start.py
Requirement:
Launch Ray Serve: serve run core.inference_graph:entrypoint.
Launch Frontend: npm run dev.
Logging: distinct color-coded prefixes for [AI-ENGINE] and [UI-CLIENT].
📸 Mandatory Proof of Completion
The PR will not be approved without the following evidence attached.

1. The "Terminal" Proof (Screenshot)
Requirement: A screenshot of the terminal running start.py.
Must Show:
Ray Serve starting up successfully.
ObjectDetector loading on GPU (Allocated VRAM logs).
TextExtractor loading on GPU.
Next.js compiling successfully.
2. The "Data" Proof (JSON Log)
Requirement: A snippet of the API Response JSON from the backend logs.
Must Show: A detection object containing both label: "tag_number" AND textContent: "V-101" (or similar).
3. The "Visual" Proof (UI Screenshot)
Requirement: A screenshot of the BlueprintViewer with a blueprint loaded.
Must Show:
An OBB bounding box around a rotated tag.
The Correctly Read Text overlaying the box (e.g., the image says "AHU-1" and the label says "AHU-1").
4. The "Performance" Report (Text)
Requirement: A brief summary of local performance.
Metric: "Average End-to-End Inference Time: X.XX seconds."
