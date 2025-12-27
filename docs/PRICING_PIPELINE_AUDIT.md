# Pricing/Quote Pipeline Audit & Implementation Report
**Date:** December 27, 2025  
**Status:** ⚠️ REQUIRES OPTIMIZATION

## Executive Summary

The pricing/quote pipeline has been refactored but the frontend still references OLD endpoints (`/api/v1/quote/generate`) that no longer exist. The modern architecture integrates pricing directly into the analysis response from `/api/hvac/analyze`.

**Current State:** FRAGMENTED & INCONSISTENT
**Action Required:** CONSOLIDATE to single unified flow

---

## Architecture Overview

### Current Modern Implementation (inference_graph.py)
```
Image Upload
    ↓
/api/hvac/analyze (POST)
    ↓
1. Object Detection (Ray Deployment)
2. Text Extraction (Ray Deployment)
3. Pricing Calculation (Inline)
    ↓
Response: {
    detections: [...],
    quote: {...},        ← INTEGRATED PRICING
    image_shape: [...]
}
```

### Legacy Implementation (Still Referenced by Frontend)
```
Image Upload
    ↓
/api/hvac/analyze
    ↓
Detections Only
    ↓
Frontend → /api/v1/quote/generate (DOESN'T EXIST IN CURRENT BACKEND)
    ↓
❌ FAILS: 404 Not Found
```

---

## Frontend Issues Found

### 1. ❌ Wrong Endpoint in api-client.ts
**File:** `src/lib/api-client.ts`  
**Function:** `generateQuote()`  
**Issue:** Calls `/api/v1/quote/generate` which doesn't exist in inference_graph.py

```typescript
// WRONG - calls non-existent endpoint
export async function generateQuote(request: GenerateQuoteRequest) {
  const response = await fetch(`${PYTHON_SERVICE_URL}/api/v1/quote/generate`, {
    method: 'POST',
    body: JSON.stringify(request)
  });
  // ...
}
```

### 2. ❌ Incorrect Health Check in api-client.ts
**File:** `src/lib/api-client.ts`  
**Function:** `checkPricingAvailable()`  
**Issue:** ✅ FIXED - Now checks `/health` endpoint (correct)

```typescript
// NOW CORRECT - checks /health
export async function checkPricingAvailable() {
  const res = await fetch(`${PYTHON_SERVICE_URL}/health`);
  const json = await res.json();
  return { available: !!json.pricing_enabled };
}
```

### 3. ⚠️ Unused Quote Generation Flow
**File:** `src/lib/pricing-store.ts`  
**Issue:** Zustand store for quote management exists but is not wired to real data flow

The frontend has:
- Quote state management ✓
- CSV export functionality ✓
- Interactive editing UI ✓

But these are DISCONNECTED from the actual `/api/hvac/analyze` response that contains the quote!

---

## Backend Implementation Assessment

### ✅ Current (inference_graph.py)
- Location: `services/hvac-ai/inference_graph.py`
- Endpoints:
  - `GET /health` → Health check with pricing status ✓
  - `POST /api/hvac/analyze` → Analysis + Pricing integrated ✓
- Pricing:
  - Integrated into response ✓
  - Regional multipliers applied ✓
  - Proper currency rounding ✓

### ❌ Legacy (hvac_unified_service.py, backend_start.py)
- Location: `services/hvac_unified_service.py`, `scripts/backend_start.py`
- Still has `/api/v1/quote/generate` endpoint
- **Problem:** These are LEGACY and should be archived

### Missing Integration Point
**Gap:** Frontend has no way to extract and use the quote from `/api/hvac/analyze` response

---

## Detailed Workflow Audit

### Step 1: User Uploads Image
**Status:** ✅ WORKING
- DeepZoomInferenceAnalysis.tsx collects file
- Sends to `/api/hvac/analyze`

### Step 2: Backend Analyzes Image
**Status:** ✅ WORKING
- inference_graph.py receives file
- Detects objects with YOLO
- Extracts text with OCR
- **NEW:** Generates quote inline

### Step 3: Response Contains Quote
**Status:** ✅ WORKING
- Response includes:
  ```json
  {
    "detections": [...],
    "quote": {
      "quote_id": "...",
      "summary": {...},
      "line_items": [...]
    },
    "image_shape": [...]
  }
  ```

### Step 4: Frontend Receives & Displays Quote
**Status:** ❌ BROKEN
- Frontend doesn't extract quote from response
- Tries to call non-existent `/api/v1/quote/generate`
- Quote state never populated with real data

### Step 5: User Edits & Exports Quote
**Status:** ⚠️ PARTIAL
- UI components exist (QuoteDashboard.tsx)
- State management exists (pricing-store.ts)
- **But:** No data flows into it from backend

---

## Implementation Gaps

| Component | Frontend | Backend | Status |
|-----------|----------|---------|--------|
| Analysis endpoint | `DeepZoomInferenceAnalysis.tsx` | `/api/hvac/analyze` | ✅ Connected |
| Quote response parsing | MISSING | Returns in `/api/hvac/analyze` | ❌ Disconnect |
| Quote state population | `pricing-store.ts` exists | Real data not flowing | ❌ Disconnect |
| Interactive editing | `InteractiveInvoice.tsx` exists | N/A | ⚠️ Orphaned |
| CSV export | `api-client.ts` has function | N/A | ⚠️ Orphaned |
| Legacy quote endpoint | api-client.ts calls it | hvac_unified_service.py has it | ❌ Should be removed |

---

## Optimal Architecture (Proposed)

```
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED SINGLE FLOW                       │
└─────────────────────────────────────────────────────────────┘

1. FRONTEND: Upload Image
   ↓
2. API: POST /api/hvac/analyze?include_quote=true
   (Default: true, pricing always included)
   ↓
3. BACKEND:
   ├─ Detect objects
   ├─ Extract text
   ├─ Generate pricing quote
   └─ Return unified response
   ↓
4. RESPONSE STRUCTURE:
   {
     "detections": [...],
     "quote": {
       "quote_id": "Q-...",
       "currency": "USD",
       "summary": {...},
       "line_items": [...]
     },
     "image_shape": [width, height]
   }
   ↓
5. FRONTEND:
   ├─ Extract quote from response
   ├─ Store in pricing-store
   ├─ Display in QuoteDashboard
   └─ Enable editing & export
```

---

## Required Fixes (Priority Order)

### 🔴 CRITICAL (Must Fix - Breaks Functionality)

1. **Update api-client.ts `generateQuote()` function**
   - Remove `/api/v1/quote/generate` endpoint call
   - Instead, this should be called from the analyze response directly
   - OR: Make this function parse already-received quote data
   - **Solution:** Extract quote from analyze response, no separate call needed

2. **Wire quote response into pricing-store**
   - When analyze response received, extract `quote` field
   - Call `useQuoteStore.setQuote()` with it
   - **File:** Need to update DeepZoomInferenceAnalysis.tsx or analyze response handler

3. **Remove legacy `/api/v1/quote/generate` references**
   - This endpoint doesn't exist in inference_graph.py
   - Remove from api-client.ts if it's still called
   - Archive hvac_unified_service.py backend_start.py

### 🟡 HIGH (Should Fix - Improves Consistency)

4. **Add `?include_quote=true` parameter support**
   - Optional query param to inference_graph.py
   - Default: always include (for backward compat)
   - Allows disabling quote if only detections needed

5. **Add quote_settings form to frontend**
   - Let users configure: margin %, labor rate, location
   - Send with analyze request as form fields
   - Backend applies to quote generation

6. **Improve error handling**
   - If quote generation fails, detections still return
   - Quote field is null/empty
   - Frontend shows warning but continues

### 🟢 NICE-TO-HAVE (Polish)

7. **Add quote versioning**
   - Store quote generation timestamp
   - Track settings used
   - Allow comparing quote versions

8. **Streaming quote generation**
   - While detection in progress, stream quote
   - Use `?stream=1` parameter
   - Frontend shows real-time costs updating

---

## Testing Checklist

### Unit Tests
- [ ] `inference_graph.py`: Quote generated in response
- [ ] `pricing_service.py`: Regional multipliers correct
- [ ] `api-client.ts`: Response parsing correct
- [ ] `pricing-store.ts`: State updated from response

### Integration Tests
- [ ] Upload image → Receive detections + quote
- [ ] Quote summary totals correct
- [ ] Line items match detection counts
- [ ] Regional pricing applied correctly

### E2E Tests
- [ ] Upload → Analyze → Quote appears in dashboard
- [ ] Edit quote → Totals recalculate
- [ ] Export CSV → File contains quote data
- [ ] Different locations → Different prices

---

## File-by-File Action Items

### Backend Files

**services/hvac-ai/inference_graph.py**
- ✅ Already correct - returns quote in analyze response
- Verify quote field always present
- Add error handling if quote generation fails

**scripts/backend_start.py & services/hvac_unified_service.py**
- ❌ DEPRECATED - Remove `/api/v1/quote/generate` endpoints
- Archive these files
- Switch to inference_graph.py only

**services/hvac-domain/pricing/pricing_service.py**
- ✅ Already correct - generates quotes properly
- No changes needed

### Frontend Files

**src/lib/api-client.ts**
- ❌ FIX: `generateQuote()` function is broken
- Either:
  - Option A: Delete - let response parsing handle it
  - Option B: Change to parse quote from analyze response
- ✅ FIXED: `checkPricingAvailable()` now uses /health

**src/components/inference/DeepZoomInferenceAnalysis.tsx**
- ADD: Extract quote from analyze response
- ADD: Call `useQuoteStore.setQuote(response.quote)`
- Update: Show quote results after analysis

**src/lib/pricing-store.ts**
- ✅ Already correct
- No changes needed

**src/components/hvac/QuoteDashboard.tsx**
- ✅ Already correct
- Should display quote from store
- Verify data flows from response

**src/components/hvac/InteractiveInvoice.tsx**
- ✅ Already correct
- Verify state bindings work

---

## Migration Path

### Phase 1: Frontend Response Handling (TODAY)
1. Fix DeepZoomInferenceAnalysis.tsx to extract quote
2. Store in pricing-store immediately
3. Display QuoteDashboard with real data

### Phase 2: Cleanup (TOMORROW)
1. Archive hvac_unified_service.py
2. Archive backend_start.py
3. Keep only inference_graph.py

### Phase 3: Enhancement (NEXT WEEK)
1. Add quote_settings form
2. Add `?include_quote=true` parameter
3. Add streaming quote generation

---

## Success Criteria

✅ **When complete:**
1. User uploads image
2. Backend returns detections + quote in single response
3. Quote appears in dashboard automatically
4. Quote is editable and exportable
5. No orphaned code or API endpoints
6. Consistent single data flow

---

## Current Blockers

1. **DeepZoomInferenceAnalysis.tsx doesn't extract quote from response**
   - Response has quote field but it's ignored
   - Need to add code to handle it

2. **api-client.ts generateQuote() calls wrong endpoint**
   - Should not call separate endpoint
   - Should extract from already-received response

3. **Pricing-store has no data source**
   - Store exists with all functionality
   - But nothing populates it with real quote data

---

## Conclusion

The backend quote pipeline is correctly implemented in inference_graph.py. The frontend has all the UI components needed. **The only issue is they're disconnected.**

Solution: Wire the response quote directly into the frontend state and UI, eliminating the need for a separate quote generation endpoint.

**Estimated Fix Time:** 1-2 hours
**Complexity:** Low (just data flow wiring)
**Risk:** Very Low (no backend changes needed)
