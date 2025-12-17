# HVAC AI Platform - System Flow and Issue Diagnosis

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Next.js Frontend (http://localhost:3000)                 │  │
│  │  - SAMAnalysis Component                                  │  │
│  │  - Image Upload UI                                        │  │
│  │  - Canvas Visualization                                   │  │
│  │                                                            │  │
│  │  Environment: NEXT_PUBLIC_API_BASE_URL ────────────┐     │  │
│  └────────────────────────────────────────────────────┼─────┘  │
└─────────────────────────────────────────────────────┼┼─────────┘
                                                       ││
                                     HTTP Requests    ││
                                     (fetch API)      ││
                                                       ││
┌──────────────────────────────────────────────────────┼┼─────────┐
│                     Backend Server                   ││         │
│  ┌───────────────────────────────────────────────────┼┼──────┐ │
│  │  FastAPI Service (http://localhost:8000)          ││      │ │
│  │  - hvac_analysis_service.py                       ││      │ │
│  │                                                    ││      │ │
│  │  Endpoints:                                        ▼▼      │ │
│  │  - GET  /                 (service info)      [CORS OK]  │ │
│  │  - GET  /health           (health check)                 │ │
│  │  - POST /api/analyze      (segmentation)                 │ │
│  │  - POST /api/count        (counting)                     │ │
│  │                                                           │ │
│  │  Environment: MODEL_PATH ─────────────┐                  │ │
│  └───────────────────────────────────────┼──────────────────┘ │
│                                           │                    │
│  ┌───────────────────────────────────────┼──────────────────┐ │
│  │  SAM Inference Engine                 │                  │ │
│  │  - core/ai/sam_inference.py           │                  │ │
│  │  - SAMInferenceEngine class           │                  │ │
│  │                                        ▼                  │ │
│  │  Loads model from: ./models/sam_hvac_finetuned.pth      │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Request Flow: Image Upload and Analysis

### 1. User Uploads Image

```
User Action
    │
    ├─> Drag & Drop OR Click to Select
    │
    ├─> SAMAnalysis.tsx: onDrop() handler
    │   - Validates file type (PNG/JPG/JPEG)
    │   - Sets uploadedImage state
    │   - Loads into canvas
    │
    └─> Image displayed in canvas
```

### 2. User Clicks "Analyze & Count All"

```
User Clicks Button
    │
    ├─> SAMAnalysis.tsx: handleCountAll()
    │   
    ├─> Pre-flight checks:
    │   - ✓ Is apiHealthy === true?
    │   - ✓ Is API_BASE_URL configured?
    │   - ✓ Is uploadedImage available?
    │   
    ├─> Create FormData with image
    │   
    ├─> fetch(`${API_BASE_URL}/api/count`, { method: 'POST', body: formData })
    │   
    │   ┌────────── HTTP POST Request ──────────┐
    │   │                                        │
    │   │  URL: http://localhost:8000/api/count │
    │   │  Method: POST                          │
    │   │  Body: multipart/form-data             │
    │   │    - image: File (binary)              │
    │   │    - grid_size: 32                     │
    │   │    - min_score: 0.2                    │
    │   └────────────────────────────────────────┘
    │                       │
    │                       ▼
    │   ┌────────── Backend Receives ───────────┐
    │   │                                        │
    │   │  1. CORS middleware ✓                 │
    │   │  2. Request logging                   │
    │   │  3. Check sam_engine exists?          │
    │   │     - Yes → Process                   │
    │   │     - No  → Return 503                │
    │   │                                        │
    │   │  4. Parse image from form data        │
    │   │  5. Convert to numpy array            │
    │   │  6. Call sam_engine.count()           │
    │   │     - Generate image embeddings       │
    │   │     - Grid sampling                   │
    │   │     - Component detection             │
    │   │     - NMS filtering                   │
    │   │     - Classification                  │
    │   │                                        │
    │   │  7. Return JSON response              │
    │   └────────────────────────────────────────┘
    │                       │
    │                       ▼
    ├─> Response received
    │   
    ├─> Parse JSON
    │   - segments: Array<Segment>
    │   - counts_by_category: Object
    │   - total_objects_found: number
    │   
    ├─> Update UI state
    │   - setCountResult(data)
    │   - Display results
    │   - Draw masks on canvas
    │   
    └─> User sees results!
```

## Common Failure Points and Solutions

### 🔴 Failure Point 1: Environment Not Configured

**Location:** Startup / Component Mount

```
Frontend loads → checks process.env.NEXT_PUBLIC_API_BASE_URL
                 → empty string ''
                 → Health check fails (invalid URL)
                 → Red warning banner shown
```

**Fix:**
1. Create `.env.local` with `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
2. Restart frontend dev server
3. Page should show backend connectivity

---

### 🔴 Failure Point 2: Backend Not Running

**Location:** API Request

```
Frontend → fetch(http://localhost:8000/health)
           → Connection refused / Network error
           → Health check fails
           → Warning banner shown
```

**Fix:**
```bash
cd python-services
python hvac_analysis_service.py
```

---

### 🔴 Failure Point 3: MODEL_PATH Not Set

**Location:** Backend Startup

```
Backend starts → Loads environment
                → MODEL_PATH is None
                → Logs error: "MODEL_PATH not set"
                → ml_models["sam_engine"] = None
                → Server runs in degraded mode
                → /health returns status: "degraded"
```

**Fix:**
1. Create `.env` with `MODEL_PATH=./models/sam_hvac_finetuned.pth`
2. Restart backend

---

### 🔴 Failure Point 4: Model File Missing

**Location:** Backend Startup (SAM Loading)

```
Backend starts → Loads environment
                → MODEL_PATH set to './models/sam_hvac_finetuned.pth'
                → Checks if file exists
                → File NOT found
                → Logs error: "Model file not found"
                → ml_models["sam_engine"] = None
                → Server runs in degraded mode
                → /health returns status: "degraded"
```

**Fix:**
1. Create `models/` directory
2. Place valid SAM model .pth file there
3. Restart backend

---

### 🔴 Failure Point 5: API Call When Backend Degraded

**Location:** Image Analysis

```
User clicks analyze → handleCountAll()
                     → API health check passes (server running)
                     → fetch(/api/count)
                     → Backend receives request
                     → sam_engine is None
                     → Returns 503 with error details
                     → Frontend shows error message
```

**Fix:**
Ensure MODEL_PATH is set and model file exists (see above)

---

## Health Check Flow

```
Component Mounts
    │
    ├─> useEffect() runs health check
    │   
    ├─> fetch(`${API_BASE_URL}/health`)
    │   
    │   Response Cases:
    │   
    │   Case 1: Cannot connect
    │   └─> setApiHealthy(false)
    │       setApiError("Cannot connect...")
    │       Show red warning banner
    │   
    │   Case 2: 200 OK, model_loaded: true
    │   └─> setApiHealthy(true)
    │       No warning banner
    │       Upload/analyze buttons enabled
    │   
    │   Case 3: 503 or model_loaded: false
    │   └─> setApiHealthy(false)
    │       setApiError(response.error)
    │       Show warning banner with details
    │
    └─> User sees current system status
```

## New Features in This Update

### 1. Graceful Degradation

- Backend starts even without model
- Clear status indicators
- Users know what's wrong

### 2. Pre-flight Checks

- Frontend checks backend health before requests
- Prevents confusing error messages
- Guides user to fix issues

### 3. Detailed Error Messages

**Old behavior:**
```
Error: Failed to fetch
```

**New behavior:**
```
Backend Service Issue: Model file not found at: ./models/sam_hvac_finetuned.pth

Quick troubleshooting:
• Ensure the backend server is running at http://localhost:8000
• Check that NEXT_PUBLIC_API_BASE_URL is set in .env.local
• Verify the SAM model is loaded (check backend logs)
• Visit http://localhost:8000/health for detailed status
```

### 4. Setup Validation

```bash
npm run check

Output:
✓ Node.js installed: v18.17.0
✓ Python installed: Python 3.10.0
✓ Environment file found: .env.local
✓ NEXT_PUBLIC_API_BASE_URL configured: http://localhost:8000
✓ MODEL_PATH configured: ./models/sam_hvac_finetuned.pth
✗ Model file not found at: ./models/sam_hvac_finetuned.pth

Found 1 error(s) and 0 warning(s)
Please fix the errors above before starting the platform.
```

## Monitoring System Health

### Backend Logs

Look for these indicators:

```
✅ Good:
   ✅ SAM engine loaded successfully from ./models/sam_hvac_finetuned.pth

⚠️ Warning:
   ❌ Cannot load SAM engine: Model file not found at: ./models/sam_hvac_finetuned.pth
   Server will run in degraded mode. API endpoints will return 503 errors.
```

### Frontend UI

Look for these indicators:

```
✅ Good:
   - No warning banners
   - Upload interface active
   - Buttons enabled

⚠️ Issues:
   - Red warning banner at top
   - Error message with troubleshooting steps
   - Upload/analyze disabled or shows errors
```

## Testing Your Setup

### 1. Backend Test

```bash
# Start backend
cd python-services
python hvac_analysis_service.py

# In another terminal, test health endpoint
curl http://localhost:8000/health | jq

# Expected: "status": "healthy" or "degraded"
```

### 2. Frontend Test

```bash
# Start frontend
npm run dev

# Open browser to http://localhost:3000/sam-analysis
# Look for:
# - No red warning banner (good)
# - Red warning banner (issue - read the message)
```

### 3. End-to-End Test

1. Navigate to SAM Analysis page
2. Upload a test image
3. Click "Analyze & Count All"
4. Should see results within 30-120 seconds
5. Canvas should show detected components

## Conclusion

The system now has comprehensive error handling and user feedback at every level:

- ✅ Clear error messages
- ✅ Health checks
- ✅ Graceful degradation
- ✅ Setup validation
- ✅ Troubleshooting guides

Users can now easily diagnose and fix configuration issues themselves!
