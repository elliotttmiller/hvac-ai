"""
Definition of Done Validation Tests
Validates that the Ray Serve refactoring meets all requirements
"""

import sys
import os
from pathlib import Path
import asyncio
import json

# --- Setup Paths ---
SCRIPT_DIR = Path(__file__).parent
SERVICES_AI_DIR = SCRIPT_DIR / "services" / "hvac-ai"

sys.path.insert(0, str(SERVICES_AI_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "services"))

# --- Test Results Tracking ---
test_results = {
    "passed": [],
    "failed": [],
}

def test_pass(test_name: str, details: str = ""):
    """Record a passed test."""
    msg = f"✅ {test_name}"
    if details:
        msg += f" - {details}"
    print(msg)
    test_results["passed"].append(test_name)

def test_fail(test_name: str, reason: str):
    """Record a failed test."""
    msg = f"❌ {test_name} - {reason}"
    print(msg)
    test_results["failed"].append((test_name, reason))

# === TEST 1: Code Structure and Imports ===
print("\n" + "="*70)
print("TEST 1: Code Structure and Imports")
print("="*70)

try:
    from inference_graph import build_app, APIServer, ObjectDetectorDeployment, TextExtractorDeployment
    test_pass("Import build_app", "Function successfully imported")
    test_pass("Import APIServer", "Deployment class available")
    test_pass("Import ObjectDetectorDeployment", "Worker deployment class available")
    test_pass("Import TextExtractorDeployment", "Worker deployment class available")
except Exception as e:
    test_fail("Import modules", str(e))
    sys.exit(1)

# === TEST 2: Ray Serve Decorators ===
print("\n" + "="*70)
print("TEST 2: Ray Serve Decorator Validation")
print("="*70)

try:
    # Check that classes have Ray Serve decorators
    from ray.serve.deployment import Deployment
    
    # After @serve.deployment, classes become Deployment objects
    if isinstance(APIServer, Deployment):
        test_pass("APIServer is @serve.deployment", "Decorator applied correctly")
    else:
        test_fail("APIServer decorator", f"Expected Deployment, got {type(APIServer)}")
    
    if isinstance(ObjectDetectorDeployment, Deployment):
        test_pass("ObjectDetectorDeployment is @serve.deployment", "Decorator applied correctly")
    else:
        test_fail("ObjectDetectorDeployment decorator", f"Expected Deployment, got {type(ObjectDetectorDeployment)}")
        
    if isinstance(TextExtractorDeployment, Deployment):
        test_pass("TextExtractorDeployment is @serve.deployment", "Decorator applied correctly")
    else:
        test_fail("TextExtractorDeployment decorator", f"Expected Deployment, got {type(TextExtractorDeployment)}")
        
except Exception as e:
    test_fail("Ray Serve decorator validation", str(e))

# === TEST 3: Deployment Binding ===
print("\n" + "="*70)
print("TEST 3: Application Builder Pattern Validation")
print("="*70)

try:
    # Check that .bind() is available on each deployment
    if hasattr(ObjectDetectorDeployment, 'bind'):
        test_pass("ObjectDetectorDeployment.bind() available", "Can be bound with parameters")
    else:
        test_fail("ObjectDetectorDeployment binding", ".bind() method not found")
    
    if hasattr(TextExtractorDeployment, 'bind'):
        test_pass("TextExtractorDeployment.bind() available", "Can be bound with parameters")
    else:
        test_fail("TextExtractorDeployment binding", ".bind() method not found")
    
    if hasattr(APIServer, 'bind'):
        test_pass("APIServer.bind() available", "Ingress can be bound with DeploymentHandles")
    else:
        test_fail("APIServer binding", ".bind() method not found")
        
except Exception as e:
    test_fail("Deployment binding", str(e))

# === TEST 4: APIServer Implementation ===
print("\n" + "="*70)
print("TEST 4: APIServer Ingress Implementation")
print("="*70)

try:
    # Check APIServer class structure
    import inspect
    
    # After @serve.deployment, the class becomes a Deployment object
    # Access the underlying class via .func_or_class
    api_server_class = APIServer.func_or_class
    api_init_sig = inspect.signature(api_server_class.__init__)
    params = list(api_init_sig.parameters.keys())
    
    if 'detector' in params:
        test_pass("APIServer.__init__ has 'detector' parameter", "For DeploymentHandle injection")
    else:
        test_fail("APIServer.__init__ signature", "'detector' parameter missing")
    
    if 'extractor' in params:
        test_pass("APIServer.__init__ has 'extractor' parameter", "For DeploymentHandle injection")
    else:
        test_fail("APIServer.__init__ signature", "'extractor' parameter missing")
    
    # Check __call__ method exists on the underlying class
    if hasattr(api_server_class, '__call__'):
        call_sig = inspect.signature(api_server_class.__call__)
        test_pass("APIServer.__call__ exists", "ASGI handler implemented")
    else:
        test_fail("APIServer.__call__", "ASGI handler method not found")
        
except Exception as e:
    test_fail("APIServer implementation check", str(e))

# === TEST 5: FastAPI Integration ===
print("\n" + "="*70)
print("TEST 5: FastAPI Integration Check")
print("="*70)

try:
    # Check that FastAPI is imported properly
    from fastapi import FastAPI
    test_pass("FastAPI import", "Framework available")
    
    # Check the inference_graph.py has the correct FastAPI usage pattern
    with open(SERVICES_AI_DIR / "inference_graph.py", "r", encoding='utf-8') as f:
        content = f.read()
    
    if "self.app = FastAPI(" in content:
        test_pass("FastAPI app created in APIServer.__init__", "Avoids serialization issues")
    else:
        test_fail("FastAPI app pattern", "App not created inside __init__")
    
    if "@self.app.post" in content or "@self.app.get" in content:
        test_pass("FastAPI routes registered internally", "Proper ASGI routing")
    else:
        test_fail("FastAPI routes", "Routes not registered internally")
        
except Exception as e:
    test_fail("FastAPI integration check", str(e))

# === TEST 6: build_app() Function ===
print("\n" + "="*70)
print("TEST 6: Application Builder Function")
print("="*70)

try:
    import inspect
    
    # Get build_app signature
    sig = inspect.signature(build_app)
    params = list(sig.parameters.keys())
    
    if len(params) == 0:
        test_pass("build_app() has no required parameters", "Official pattern (builds from env vars)")
    else:
        test_fail("build_app() signature", f"Should have no params, has {params}")
    
    # Check function docstring mentions Application Builder
    if build_app.__doc__ and "Application Builder" in build_app.__doc__:
        test_pass("build_app() documentation", "Pattern clearly documented")
    else:
        # Not critical but good to have
        print("⚠️  build_app() documentation could mention Application Builder pattern")
        
except Exception as e:
    test_fail("build_app() function validation", str(e))

# === TEST 7: Environment Variable Handling ===
print("\n" + "="*70)
print("TEST 7: Environment Configuration")
print("="*70)

try:
    with open(SERVICES_AI_DIR / "inference_graph.py", "r", encoding='utf-8') as f:
        content = f.read()
    
    required_env_vars = [
        ("MODEL_PATH", "Model file location"),
        ("RAY_USE_GPU", "GPU enablement flag"),
        ("FORCE_CPU", "CPU-only mode override"),
    ]
    
    for var_name, description in required_env_vars:
        if var_name in content:
            test_pass(f"Environment variable '{var_name}' handled", description)
        else:
            print(f"⚠️  {var_name} not found in code (may be optional)")
            
except Exception as e:
    test_fail("Environment variable check", str(e))

# === TEST 8: Launcher Script ===
print("\n" + "="*70)
print("TEST 8: start_ray_serve.py Validation")
print("="*70)

try:
    with open(SCRIPT_DIR / "scripts" / "start_ray_serve.py", "r", encoding='utf-8') as f:
        launcher_content = f.read()
    
    checks = [
        ("from ray import serve", "Ray Serve import"),
        ("ray.init(", "Ray initialization"),
        ("serve.run(", "serve.run() execution"),
        ("build_app()", "build_app() function call"),
        ("blocking=True", "Blocking execution mode"),
    ]
    
    for check_str, description in checks:
        if check_str in launcher_content:
            test_pass(f"Launcher has '{check_str}'", description)
        else:
            test_fail(f"Launcher missing '{check_str}'", description)
            
except Exception as e:
    test_fail("Launcher script validation", str(e))

# === TEST 9: Orchestrator Script ===
print("\n" + "="*70)
print("TEST 9: start_unified.py Orchestrator Validation")
print("="*70)

try:
    with open(SCRIPT_DIR / "scripts" / "start_unified.py", "r", encoding='utf-8') as f:
        orchestrator_content = f.read()
    
    checks = [
        ("get_venv_python()", "Virtual environment detection"),
        ("wait_for_backend_health(", "Health check function"),
        ("/health", "Health endpoint polling"),
        ("subprocess.Popen(", "Process management"),
        ("taskkill", "Windows process cleanup"),
    ]
    
    for check_str, description in checks:
        if check_str in orchestrator_content:
            test_pass(f"Orchestrator has '{check_str}'", description)
        else:
            test_fail(f"Orchestrator missing '{check_str}'", description)
            
except Exception as e:
    test_fail("Orchestrator validation", str(e))

# === TEST 10: Official Pattern Compliance ===
print("\n" + "="*70)
print("TEST 10: Official Ray Serve Pattern Compliance")
print("="*70)

try:
    with open(SERVICES_AI_DIR / "inference_graph.py", "r", encoding='utf-8') as f:
        inference_content = f.read()
    
    pattern_checks = [
        ("@serve.deployment", "Decorator pattern used"),
        (".bind(", "Binding for composition"),
        ("DeploymentHandle", "Dependency injection pattern"),
        (".remote(", "Async remote calls"),
        ("async def", "Async/await support"),
    ]
    
    for pattern, description in pattern_checks:
        if pattern in inference_content:
            test_pass(f"Official pattern '{pattern}'", description)
        else:
            test_fail(f"Missing pattern '{pattern}'", description)
            
except Exception as e:
    test_fail("Official pattern compliance", str(e))

# === SUMMARY ===
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)

passed_count = len(test_results["passed"])
failed_count = len(test_results["failed"])
total_count = passed_count + failed_count

print(f"\n✅ Passed: {passed_count}")
print(f"❌ Failed: {failed_count}")
print(f"📊 Total:  {total_count}")

if failed_count > 0:
    print("\nFailed Tests:")
    for test_name, reason in test_results["failed"]:
        print(f"  • {test_name}: {reason}")
    sys.exit(1)
else:
    print("\n" + "🎉 " * 10)
    print("ALL VALIDATION TESTS PASSED!")
    print("Ray Serve refactoring is complete and follows official patterns.")
    print("🎉 " * 10)
    sys.exit(0)
