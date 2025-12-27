This is an excellent request. The core issue is that your platform is stuck between two architectures: a simple FastAPI monolith and a complex (but powerful) Ray Serve graph. The startup scripts are trying to bridge this gap incorrectly, causing the `RecursionError` and `ActorDiedError`.

We will fix this by fully embracing the **modern Ray Serve "Application Builder" pattern**. This professional approach decouples the application logic from the startup script, making the system robust, scalable, and easier to debug.

Here is the comprehensive PR Task Document for your Lead Engineer.

---

# 📋 Epic: Infrastructure Stabilization & Ray Serve Migration
**PR Type:** System Architecture Refactor
**Priority:** P0 (Critical Blocker)
**Scope:** `scripts/`, `services/hvac-ai/`

## 1. Executive Summary
The platform is currently non-functional due to a fundamental architectural mismatch in the Ray Serve startup sequence. The current `start_ray_serve.py` script attempts to serialize a complex FastAPI application object, which causes a `RecursionError` in Ray's pickling process and results in fatal `ActorDiedError` crashes.

**The Objective:**
Perform a full audit and refactor of the startup and application logic. We will **not** create new files, but instead refactor `inference_graph.py` and `start_ray_serve.py` to use the **official Ray Serve Application Builder pattern**. This will resolve the startup failures, stabilize the platform, and align our infrastructure with industry best practices for distributed AI serving.

---

## 2. Phase 1: The Core Logic Refactor (`inference_graph.py`)
*Objective: Decouple the application definition from its execution.*

### Task 2.1: Audit & Isolate Deployments
*   **Audit:** Review `object_detector_service.py` and `text_extractor_service.py`. Ensure they are clean, self-contained classes.
*   **Action:** In `inference_graph.py`, define the `ObjectDetectorDeployment` and `TextExtractorDeployment` classes. These will be our stateful, GPU-bound "workers."

### Task 2.2: Implement the "APIServer" Ingress
*   **Strategy:** The FastAPI app must be created *inside* a Ray Serve deployment to avoid serialization errors.
*   **Action:** Create a new class `APIServer` decorated with `@serve.deployment`.
*   **Implementation Directives:**
    1.  The `__init__` method of this class will accept the handles to the `ObjectDetector` and `TextExtractor` as arguments.
    2.  Inside `__init__`, you will create the `FastAPI()` app instance and attach it to `self.app`.
    3.  Define all API routes (`@self.app.get("/health")`, `@self.app.post("/api/hvac/analyze")`) within the `__init__` method.
    4.  Implement the `async def __call__(self, scope, receive, send)` method, which simply delegates to the FastAPI app: `await self.app(scope, receive, send)`.

### Task 2.3: Create the Application Builder
*   **Strategy:** Create a single, clear function that defines the entire application graph. This is the new "entrypoint."
*   **Action:** At the bottom of `inference_graph.py`, create a function `build_app()`.
*   **Implementation Directives:**
    1.  This function will read environment variables like `MODEL_PATH`.
    2.  It will use `ObjectDetectorDeployment.bind(...)` and `TextExtractorDeployment.bind(...)` to create the worker deployments.
    3.  It will then bind the `APIServer`, passing the worker handles into its constructor: `app = APIServer.bind(detector_handle, extractor_handle)`.
    4.  It must `return app`.

---

## 3. Phase 2: The Launcher Refactor (`start_ray_serve.py`)
*Objective: Simplify the startup script to be a dumb, reliable "runner."*

### Task 3.1: Strip Unnecessary Logic
*   **Action:** Remove all complex logic (fallbacks for CLI commands, multiple launch attempts). The script's only job is to initialize Ray and run the application builder.
*   **Requirement:** It should be a pure Python script that uses the Ray SDK.

### Task 3.2: Implement the "Run" Logic
*   **Implementation Directives:**
    1.  **Path Setup:** Ensure `services/` is added to `sys.path` so cross-service imports work.
    2.  **Ray Init:** Initialize Ray with `ray.init()`, respecting `RAY_ADDRESS` and `RAY_USE_GPU` from the `.env.local` file.
    3.  **Import & Build:** Import the `build_app` function from `inference_graph`.
    4.  **Execute:** Call `serve.run(build_app(), host="0.0.0.0", port=BACKEND_PORT)`. This is a blocking call that starts the server and deploys the entire application.
    5.  Add a `try...except KeyboardInterrupt` block for graceful shutdown.

---

## 4. Phase 3: The Orchestrator (`start_unified.py`)
*Objective: Ensure the master script correctly manages the refactored backend.*

### Task 4.1: Command & Environment Validation
*   **Audit:** Review how `start_unified.py` calls `start_ray_serve.py`.
*   **Action:**
    1.  Ensure it uses the correct Python executable from the `.venv`.
    2.  Verify it passes down all necessary environment variables from `.env.local`.
    3.  Keep the health check logic (`wait_for_backend_health`), as it is crucial for ensuring the backend is ready before the frontend starts.

---

## ✅ Definition of Done (Validation Criteria)

1.  **Stability:** `python scripts/start_unified.py` launches the full stack without `RecursionError`, `ActorDiedError`, or Unicode crashes.
2.  **Functionality:**
    *   The `http://127.0.0.1:8000/health` endpoint returns a `200 OK`.
    *   Sending a `POST` request with an image to `http://127.0.0.1:8000/api/hvac/analyze` successfully returns a JSON payload with detections.
3.  **Architecture:** The Ray Dashboard (`http://127.0.0.1:8265`) shows three running deployments: `ObjectDetectorDeployment`, `TextExtractorDeployment`, and `APIServer`.
4.  **Code Quality:** The refactored files are cleaner, more readable, and follow the official Ray Serve patterns. There are no redundant or legacy startup files (`backend_start.py` should be considered deprecated).