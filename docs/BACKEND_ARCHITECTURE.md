# Backend Pipeline Architecture Review

## Summary: Three Backend Options (Choose ONE)

We have **three different backend implementations**. They serve similar purposes but use different architectures:

| Backend | Location | Architecture | Status | Use Case |
|---------|----------|--------------|--------|----------|
| **inference_graph.py** (NEW) | `services/hvac-ai/` | Ray Serve + FastAPI | ✅ RECOMMENDED | Production/distributed |
| **backend_start.py** | `scripts/` | Uvicorn (standalone FastAPI) | ⚠️ Legacy | Simple/local dev |
| **hvac_unified_service.py** | `services/` | Uvicorn + Pipeline Router | ⚠️ Legacy | Complex analysis pipeline |

---

## 1. inference_graph.py (CURRENT/RECOMMENDED)

**Architecture:** Ray Serve + FastAPI with distributed deployments

**Location:** `services/hvac-ai/inference_graph.py`

**Startup:** Via `scripts/start_ray_serve.py`

**How it works:**
- Ray Serve wraps FastAPI app
- ObjectDetector deployment (40% GPU)
- TextExtractor deployment (30% GPU)
- APIServer deployment (CPU-only router)
- All workers communicate via Ray protocol on port 10001
- HTTP endpoint exposed on port 8000

**Endpoints:**
```
GET  /health                    → Health check
POST /api/hvac/analyze          → Image analysis + text extraction + pricing
```

**Configuration:** Uses `.env.local` variables:
- `MODEL_PATH` - Path to YOLO model
- `BACKEND_PORT` (8000) - Ray Serve HTTP port
- `RAY_PORT` (10001) - Ray cluster communication
- `FORCE_CPU` - CPU-only mode
- `RAY_USE_GPU` - GPU allocation

**Advantages:**
- Distributed/scalable
- Proper separation of concerns (detection, text, routing)
- GPU-optimized with Ray resource allocation
- FastAPI modern framework

---

## 2. backend_start.py (LEGACY)

**Architecture:** Standalone Uvicorn FastAPI server

**Location:** `scripts/backend_start.py`

**Startup:** Via `scripts/start.py`

**How it works:**
- Single FastAPI app running on Uvicorn
- Direct model loading (no Ray deployments)
- All inference in single process
- Blocking I/O (no async optimization)

**Endpoints:**
```
GET  /health                           → Health check
POST /api/hvac/analyze                 → Image analysis
POST /api/v1/quote/generate            → Price quotes
GET  /api/v1/quote/available           → Available quotes
GET  /api/v1/diagnostics               → Diagnostics
```

**Disadvantages:**
- Single-process (not scalable)
- No Ray optimization
- Blocking endpoints
- Limited by single machine resources

---

## 3. hvac_unified_service.py (LEGACY)

**Architecture:** Uvicorn FastAPI + Pipeline Router

**Location:** `services/hvac_unified_service.py`

**Startup:** Not currently used by start_unified.py

**How it works:**
- Attempts to import optional `pipeline_api` router
- Falls back to YOLO inference if pipeline unavailable
- Includes streaming analysis endpoint
- Complex annotation/database logic

**Endpoints:**
```
GET  /health                              → Health check
POST /api/v1/analyze                      → Standard analysis
POST /api/v1/analyze/stream               → Streaming analysis (SSE)
POST /api/v1/count                        → Count detections only
POST /api/v1/annotations/save             → Save annotations
POST /api/v1/quote/generate               → Price quotes
```

**Disadvantages:**
- Unused/deprecated (no active startup script)
- Complex optional dependencies
- Not integrated with Ray
- Streaming logic may conflict with frontend expectations

---

## Current Architecture (What We're Using)

```
start_unified.py
    ↓
start_ray_serve.py
    ↓
inference_graph.py
    ├─ FastAPI app
    ├─ Ray Serve wrapper
    └─ Deployments:
        ├─ ObjectDetectorDeployment (40% GPU)
        ├─ TextExtractorDeployment (30% GPU)
        └─ APIServer (CPU router)
```

**Frontend → Backend Flow:**
```
Browser (port 3000)
    ↓ POST /api/hvac/analyze
Next.js API Route (src/app/api/hvac/analyze/route.ts)
    ↓ POST /api/hvac/analyze
Ray Serve HTTP (port 8000) ← inference_graph.py
    ↓ Ray Protocol (port 10001)
Distributed Workers (GPU inference)
    ↓
Frontend receives JSON with detections
```

---

## Recommendation: Consolidate to inference_graph.py ONLY

**Action Items:**

1. ✅ **Use inference_graph.py** - Already integrated with `start_unified.py`
2. ❌ **Remove or archive backend_start.py** - Legacy, not needed
3. ❌ **Remove or archive hvac_unified_service.py** - Legacy, not needed

**Why:**
- `inference_graph.py` is the only backend using Ray Serve (modern/scalable)
- `start_unified.py` already launches it via `start_ray_serve.py`
- Frontend is configured to post to `/api/hvac/analyze` (which inference_graph.py provides)
- GPU optimization via Ray deployments
- Cleaner architecture with single source of truth

---

## If You Need to Support Multiple Backends

If you want to support switching between backends, update `start_unified.py` to accept a `--backend` flag:

```python
parser.add_argument('--backend', choices=['ray-serve', 'uvicorn-simple', 'uvicorn-pipeline'], 
                   default='ray-serve')
```

Then conditionally launch the appropriate script. But for now, just use **inference_graph.py**.

---

## Files to Clean Up

- `scripts/backend_start.py` - Legacy standalone FastAPI
- `scripts/start.py` - Legacy launcher (uses backend_start.py)
- `services/hvac_unified_service.py` - Legacy unified service

Keep:
- `scripts/start_unified.py` - Modern unified launcher
- `scripts/start_ray_serve.py` - Ray Serve initialization
- `services/hvac-ai/inference_graph.py` - Modern distributed backend
