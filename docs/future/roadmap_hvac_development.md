# HVAC AI Autonomous Extractor: Strategic Development Roadmap & Technical Audit

**Version:** 1.0  
**Target Architecture:** YOLO11 (Ultralytics)  
**Infrastructure:** Google Colab (T4 GPU)  
**Domain:** MEP/HVAC Technical Drawings & Schematics

---

## 1. Executive Summary
This document outlines the architectural standards, optimization strategies, and future integration logic required to build a state-of-the-art (SOTA) AI system for extracting intelligence from HVAC technical drawings. It synthesizes findings from deep-learning optimization research and industry-standard MEP drafting guidelines.

---

## 2. The Core AI Engine (Current "Golden Ratio" Configuration)
To maximize performance on Google Colab T4 Free Tier while maintaining SOTA accuracy for small objects (valves/instruments), the following configuration is established as the baseline.

### 2.1 Model Architecture
*   **Engine:** YOLO11 (Medium) - `yolo11m.pt`
*   **Reasoning:** Offers the best balance of **C2PSA** (Spatial Attention for small objects) and inference speed/VRAM usage.

### 2.2 Hardware Optimization (T4 GPU Constraints)
| Parameter | Value | Logic |
| :--- | :--- | :--- |
| **Image Size** | `1024` | **Critical.** 640px is insufficient for needle valves. 1024px allows geometry resolution. |
| **Batch Size** | `12` | Optimized to fill T4 VRAM (~14GB) without OOM crashes. |
| **Workers** | `4` | Prevents System RAM bottlenecks on Colab (2 vCPU limit). |
| **Cache** | `False` | Forces disk streaming to prevent RAM overflow. |
| **AMP** | `True` | Mixed Precision enabled for 2x training speed. |

### 2.3 SOTA Hyperparameters (The "Unified" Strategy)
We utilize a two-layer augmentation approach to handle both **Geometry** (Physical) and **Texture** (Visual).

*   **Physical Layer (Native YOLO Args):**
    *   `mixup=0.0`: **DISABLED.** Blending images destroys sharp edges required for line drawings.
    *   `copy_paste=0.3`: **ENABLED.** Artificially increases density of rare small symbols.
    *   `mosaic=1.0`: **ENABLED.** Teaches context and scale invariance.
    *   `degrees=10.0`: Handles scan skew (not orientation).
    *   `fliplr=0.5` / `flipud=0.5`: Handles 90-degree orientation changes.

*   **Visual Layer (Albumentations via Native Args):**
    *   `hsv_s=0.7`: Simulates faded ink or high-saturation digital exports.
    *   `hsv_v=0.4`: Simulates dark scans or overexposed photography.

---

## 3. Industry Standards & Logic Integration (MEP Audit)
Based on the audit of *MEP Academy* and industry drafting standards, the model must evolve to understand the following visual logic.

### 3.1 Visual Hierarchy (The "Alphabet" of Lines)
The model must eventually distinguish objects based on line type, not just shape.
*   **Solid Lines:** Physical object boundaries (Equipment, Duct). -> *Primary Detection Target.*
*   **Hidden Lines (`---`):** Objects obscured or on different elevations. -> *Requires specific class or attribute (`is_hidden`).*
*   **Center Lines (`_ . _`):** Alignment markers. -> *Ignore for detection, use for logic/connection.*
*   **Leader Lines:** Connects Text to Geometry. -> *Critical for OCR association.*

### 3.2 View Logic (Context Separation)
**CRITICAL RISK:** Schematic symbols (bowties) do not look like Plan View symbols (physical footprints).
*   **Action:** Do not mix "Schematic" and "Floor Plan" images in the same dataset unless using a classifier to distinguish them first.
*   **Current Focus:** Schematics (P&ID/Flow Diagrams).

### 3.3 Symbol Taxonomy Expansion
To move beyond basic valves, future datasets must include:
1.  **Ductwork Logic:** Supply (X), Return (/), and Exhaust markers.
2.  **Line Types:** Distinguishing Piping (Single Line) from Ducting (Double Line).
3.  **Instrumentation:** Sensors (T, P, H) inside circles vs. Actuators.

---

## 4. Future Implementation Roadmap

### Phase 1: Data Engineering (Immediate)
*   **Annotation Strategy:** Continue using **Smart Polygons**. They provide the highest quality data and allow for future segmentation usage.
*   **Class Names:** Use specific technical IDs (e.g., `needle_valve`) during training. Map to human-readable labels (`Needle Valve`) during inference/post-processing.
*   **Versioning:** Maintain the `hvac_coco_vX` naming convention to track data lineage.

### Phase 2: Hybrid AI Architecture (Mid-Term)
Object detection finds *where* things are. It does not read *what* they are (specifically).
*   **OCR Integration:** Implement a secondary stage (PaddleOCR or EasyOCR).
    *   *Workflow:* YOLO detects `text_tag` region -> Crop -> OCR reads "VAV-101".
*   **Text-Symbol Association:** Use proximity logic or Leader Line detection to link the text "VAV-101" to the detected "VAV Box" symbol.

### Phase 3: Connectivity & Logic (Long-Term)
*   **Line Segmentation:** Train a YOLO11-Seg model to trace lines (`pipe_run`).
*   **Graph Logic:** Build a node-edge graph where Symbols are Nodes and Pipes are Edges. This allows the system to understand *flow* (e.g., "Water flows from Chiller to Pump").

---

## 5. MLOps Best Practices (Google Colab Specific)
*   **Secrets Management:** Always use `google.colab.userdata` for API keys. Never hardcode credentials.
*   **Smart Resume:** Always implement logic to check for `last.pt` before starting training to handle Colab disconnects gracefully.
*   **Storage:** Save checkpoints (`project=...`) directly to Google Drive to prevent data loss.

---

**End of Report**
*Generated by AI Agent for HVAC Autonomous Extraction Project*