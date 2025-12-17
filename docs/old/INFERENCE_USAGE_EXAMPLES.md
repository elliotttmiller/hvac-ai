# AI Inference Pipeline - Usage Examples

This document provides practical examples of using the enhanced AI inference pipeline.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Advanced Features](#advanced-features)
3. [Performance Optimization](#performance-optimization)
4. [Integration Examples](#integration-examples)
5. [Troubleshooting](#troubleshooting)

## Basic Usage

### Example 1: Simple Interactive Segmentation

```python
import cv2
from core.ai.sam_inference import create_sam_engine

# Initialize engine
engine = create_sam_engine()

# Load image
image = cv2.imread("hvac_diagram.png")

# Define a point prompt (user clicked at x=450, y=300)
prompt = {
    "type": "point",
    "data": {
        "coords": [450, 300],
        "label": 1  # Positive point (1) vs negative point (0)
    }
}

# Run segmentation
results = engine.segment(image, prompt)

# Access result
segment = results[0]
print(f"Detected: {segment.label}")
print(f"Confidence: {segment.score:.3f}")
print(f"Bounding box: {segment.bbox}")
```

### Example 2: Automated Component Counting

```python
import cv2
from core.ai.sam_inference import create_sam_engine

# Initialize engine
engine = create_sam_engine()

# Load image
image = cv2.imread("pid_diagram.png")

# Run automated counting
result = engine.count(image)

# Display results
print(f"Total objects found: {result.total_objects_found}")
print("\nBreakdown by category:")
for category, count in result.counts_by_category.items():
    print(f"  {category}: {count}")
```

### Example 3: Using the REST API

```bash
# Segment a component
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@hvac_diagram.png" \
  -F 'prompt={"type":"point","data":{"coords":[450,300],"label":1}}'

# Count all components
curl -X POST http://localhost:8000/api/v1/count \
  -F "image=@pid_diagram.png"
```

## Advanced Features

### Example 4: Top-K Predictions

Get multiple segmentation predictions for uncertain cases:

```python
import cv2
from core.ai.sam_inference import create_sam_engine

engine = create_sam_engine()
image = cv2.imread("complex_diagram.png")

prompt = {
    "type": "point",
    "data": {"coords": [450, 300], "label": 1}
}

# Get top 3 predictions
results = engine.segment(
    image, 
    prompt, 
    return_top_k=3,
    enable_refinement=True
)

# Review all predictions
for i, segment in enumerate(results, 1):
    print(f"\nPrediction {i}:")
    print(f"  Label: {segment.label}")
    print(f"  Score: {segment.score:.3f}")
    print(f"  Confidence breakdown:")
    for key, value in segment.confidence_breakdown.items():
        print(f"    {key}: {value:.3f}")
    
    print(f"  Alternative labels:")
    for alt_label, alt_score in segment.alternative_labels[:3]:
        print(f"    {alt_label}: {alt_score:.3f}")
```

**Example Output**:
```
Prediction 1:
  Label: Valve-Ball
  Score: 0.967
  Confidence breakdown:
    geometric: 0.920
    visual: 0.880
    combined: 0.902
  Alternative labels:
    Valve-Gate: 0.850
    Valve-Control: 0.780
    Fitting-Generic: 0.650

Prediction 2:
  Label: Valve-Control
  Score: 0.923
  ...
```

### Example 5: Adaptive Grid Counting

Optimize counting for different image sizes:

```python
import cv2
from core.ai.sam_inference import create_sam_engine

engine = create_sam_engine()
image = cv2.imread("large_pid_diagram.png")

# Let the system adapt grid size automatically
result = engine.count(
    image,
    grid_size=32,           # Base grid size
    use_adaptive_grid=True,  # Enable adaptation
    confidence_threshold=0.85
)

print(f"Found {result.total_objects_found} objects")
print(f"Processing time: {result.processing_time_ms:.1f}ms")
print(f"\nConfidence statistics:")
print(f"  Mean: {result.confidence_stats['mean']:.3f}")
print(f"  Std Dev: {result.confidence_stats['std']:.3f}")
print(f"  Range: {result.confidence_stats['min']:.3f} - {result.confidence_stats['max']:.3f}")
print(f"  Before NMS: {result.confidence_stats['above_threshold']}")
print(f"  After NMS: {result.confidence_stats['after_nms']}")
```

### Example 6: Using the Cache

Optimize performance for repeated operations:

```python
import cv2
import time
from core.ai.sam_inference import create_sam_engine

# Initialize with larger cache
engine = create_sam_engine(
    enable_cache=True,
    cache_size=100
)

image = cv2.imread("hvac_diagram.png")

# First run - cold cache
start = time.time()
result1 = engine.count(image)
time1 = time.time() - start

# Second run - warm cache
start = time.time()
result2 = engine.count(image)
time2 = time.time() - start

print(f"First run: {time1:.3f}s")
print(f"Second run: {time2:.3f}s")
print(f"Speedup: {time1/time2:.1f}x")

# Check cache statistics
metrics = engine.get_metrics()
print(f"\nCache hit rate: {metrics['cache_hit_rate']:.1%}")
print(f"Cache utilization: {metrics['cache_size']}/{metrics['cache_max_size']}")
```

## Performance Optimization

### Example 7: Batch Processing with Cache

Process multiple images efficiently:

```python
import cv2
import glob
from core.ai.sam_inference import create_sam_engine

# Initialize with cache enabled
engine = create_sam_engine(enable_cache=True, cache_size=50)

# Get all images
image_paths = glob.glob("diagrams/*.png")

results = []
for path in image_paths:
    image = cv2.imread(path)
    
    # Count components
    result = engine.count(image, use_adaptive_grid=True)
    
    results.append({
        'file': path,
        'count': result.total_objects_found,
        'time_ms': result.processing_time_ms,
        'categories': result.counts_by_category
    })

# Summary statistics
total_objects = sum(r['count'] for r in results)
avg_time = sum(r['time_ms'] for r in results) / len(results)

print(f"Processed {len(results)} images")
print(f"Total objects: {total_objects}")
print(f"Average processing time: {avg_time:.1f}ms")

# Get cache statistics
metrics = engine.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
```

### Example 8: Monitoring Performance

Track and optimize inference performance:

```python
import cv2
from core.ai.sam_inference import create_sam_engine

engine = create_sam_engine()

# Process some images
for i in range(10):
    image = cv2.imread(f"diagram_{i}.png")
    result = engine.count(image)

# Get comprehensive metrics
metrics = engine.get_metrics()

print("Performance Metrics:")
print(f"  Total inferences: {metrics['total_inferences']}")
print(f"  Cache hits: {metrics['cache_hits']}")
print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
print(f"  Avg inference time: {metrics['avg_inference_time_ms']:.1f}ms")
print(f"  Cache utilization: {metrics['cache_size']}/{metrics['cache_max_size']}")

# Clear cache if needed
if metrics['cache_size'] > 45:
    print("\nCache nearly full, clearing...")
    engine.clear_cache()
```

### Example 9: API Performance Monitoring

Monitor performance through the REST API:

```bash
# Run some operations
for i in {1..5}; do
  curl -X POST http://localhost:8000/api/v1/count \
    -F "image=@diagram_$i.png" \
    -s > /dev/null
done

# Check metrics
curl http://localhost:8000/api/v1/metrics | jq .

# Output:
# {
#   "status": "success",
#   "metrics": {
#     "total_inferences": 25,
#     "cache_hits": 12,
#     "cache_hit_rate": 0.48,
#     "avg_inference_time_ms": 287.3,
#     "cache_size": 15,
#     "cache_max_size": 50
#   }
# }

# Clear cache if needed
curl -X POST http://localhost:8000/api/v1/cache/clear
```

## Integration Examples

### Example 10: Python Client for API

```python
import requests
import json
from pathlib import Path

class HVACAnalysisClient:
    """Client for HVAC AI Platform API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def segment_component(self, image_path, coords, return_top_k=1, 
                         enable_refinement=True):
        """Segment a component interactively"""
        url = f"{self.base_url}/api/v1/segment"
        
        prompt = {
            "type": "point",
            "data": {"coords": coords, "label": 1}
        }
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'prompt': json.dumps(prompt),
                'return_top_k': return_top_k,
                'enable_refinement': enable_refinement
            }
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def count_components(self, image_path, grid_size=32, 
                        confidence_threshold=0.85, 
                        use_adaptive_grid=True):
        """Count all components in diagram"""
        url = f"{self.base_url}/api/v1/count"
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {
                'grid_size': grid_size,
                'confidence_threshold': confidence_threshold,
                'use_adaptive_grid': use_adaptive_grid
            }
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def get_metrics(self):
        """Get performance metrics"""
        url = f"{self.base_url}/api/v1/metrics"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def clear_cache(self):
        """Clear the inference cache"""
        url = f"{self.base_url}/api/v1/cache/clear"
        response = requests.post(url)
        response.raise_for_status()
        return response.json()

# Usage
client = HVACAnalysisClient()

# Segment a component
result = client.segment_component(
    "hvac_diagram.png",
    coords=[450, 300],
    return_top_k=3
)

print(f"Top prediction: {result['segments'][0]['label']}")
print(f"Confidence: {result['segments'][0]['score']:.3f}")

# Count components
count_result = client.count_components("pid_diagram.png")
print(f"\nTotal objects: {count_result['total_objects_found']}")
print(f"Processing time: {count_result['processing_time_ms']:.1f}ms")

# Get metrics
metrics = client.get_metrics()
print(f"\nCache hit rate: {metrics['metrics']['cache_hit_rate']:.1%}")
```

### Example 11: Frontend Integration

JavaScript/TypeScript example for React:

```typescript
// api/hvacAnalysis.ts
export interface SegmentResult {
  label: string;
  score: number;
  mask: string;
  bbox: number[];
  confidence_breakdown?: {
    geometric: number;
    visual: number;
    combined: number;
  };
  alternative_labels?: [string, number][];
}

export interface CountResult {
  total_objects_found: number;
  counts_by_category: Record<string, number>;
  processing_time_ms?: number;
  confidence_stats?: {
    mean: number;
    std: number;
    min: number;
    max: number;
    above_threshold: number;
    after_nms: number;
  };
}

export async function segmentComponent(
  imageFile: File,
  coords: [number, number],
  returnTopK: number = 1,
  enableRefinement: boolean = true
): Promise<SegmentResult[]> {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('prompt', JSON.stringify({
    type: 'point',
    data: { coords, label: 1 }
  }));
  formData.append('return_top_k', returnTopK.toString());
  formData.append('enable_refinement', enableRefinement.toString());

  const response = await fetch('/api/v1/segment', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Segmentation failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.segments;
}

export async function countComponents(
  imageFile: File,
  options?: {
    gridSize?: number;
    confidenceThreshold?: number;
    useAdaptiveGrid?: boolean;
  }
): Promise<CountResult> {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  if (options?.gridSize) {
    formData.append('grid_size', options.gridSize.toString());
  }
  if (options?.confidenceThreshold) {
    formData.append('confidence_threshold', options.confidenceThreshold.toString());
  }
  if (options?.useAdaptiveGrid !== undefined) {
    formData.append('use_adaptive_grid', options.useAdaptiveGrid.toString());
  }

  const response = await fetch('/api/v1/count', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Counting failed: ${response.statusText}`);
  }

  return await response.json();
}

// Usage in React component
import { useState } from 'react';
import { segmentComponent, countComponents } from './api/hvacAnalysis';

function DiagramAnalysis() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleCanvasClick = async (e, imageFile) => {
    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setLoading(true);
    try {
      const segments = await segmentComponent(
        imageFile,
        [x, y],
        3, // Get top 3 predictions
        true // Enable refinement
      );
      
      setResults(segments);
      console.log('Top prediction:', segments[0].label);
      console.log('Confidence:', segments[0].score);
      console.log('Alternatives:', segments[0].alternative_labels);
    } catch (error) {
      console.error('Segmentation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCount = async (imageFile) => {
    setLoading(true);
    try {
      const result = await countComponents(imageFile, {
        useAdaptiveGrid: true,
        confidenceThreshold: 0.85
      });
      
      console.log('Found:', result.total_objects_found, 'objects');
      console.log('Time:', result.processing_time_ms, 'ms');
      console.log('Categories:', result.counts_by_category);
      console.log('Confidence:', result.confidence_stats);
    } catch (error) {
      console.error('Counting failed:', error);
    } finally {
      setLoading(false);
    }
  };

  // ... component JSX
}
```

## Troubleshooting

### Example 12: Debugging Low Confidence

```python
import cv2
from core.ai.sam_inference import create_sam_engine

engine = create_sam_engine()
image = cv2.imread("problematic_diagram.png")

# Get detailed predictions
results = engine.segment(
    image,
    {"type": "point", "data": {"coords": [450, 300], "label": 1}},
    return_top_k=5  # Get more predictions
)

# Analyze confidence breakdown
for i, segment in enumerate(results, 1):
    print(f"\nPrediction {i}: {segment.label}")
    print(f"Overall score: {segment.score:.3f}")
    print(f"Confidence breakdown:")
    print(f"  Geometric: {segment.confidence_breakdown['geometric']:.3f}")
    print(f"  Visual: {segment.confidence_breakdown['visual']:.3f}")
    print(f"  Combined: {segment.confidence_breakdown['combined']:.3f}")
    
    # Check if geometric or visual features are weak
    if segment.confidence_breakdown['geometric'] < 0.7:
        print("  ⚠️  Low geometric confidence - shape may be ambiguous")
    if segment.confidence_breakdown['visual'] < 0.7:
        print("  ⚠️  Low visual confidence - image quality may be poor")
```

### Example 13: Cache Optimization

```python
import cv2
import glob
from core.ai.sam_inference import create_sam_engine

# Start with default cache
engine = create_sam_engine(enable_cache=True, cache_size=50)

images = [cv2.imread(p) for p in glob.glob("diagrams/*.png")]

# First pass - measure performance
for image in images:
    engine.count(image)

metrics = engine.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")

# If hit rate is low, increase cache size
if metrics['cache_hit_rate'] < 0.3:
    print("Low hit rate, increasing cache size...")
    engine = create_sam_engine(enable_cache=True, cache_size=100)
    
    # Process again
    for image in images:
        engine.count(image)
    
    new_metrics = engine.get_metrics()
    print(f"New cache hit rate: {new_metrics['cache_hit_rate']:.1%}")
```

## Best Practices Summary

1. **Enable caching** for repeated operations on the same images
2. **Use adaptive grid** for counting unless you know the optimal grid size
3. **Request top-k predictions** for uncertain or quality-critical cases
4. **Monitor metrics** periodically to optimize performance
5. **Clear cache** when switching between different projects or datasets
6. **Use refinement** for better boundary segmentation
7. **Adjust confidence threshold** based on precision/recall requirements
8. **Check confidence breakdown** when debugging low-quality predictions

## Additional Resources

- [API Documentation](../docs/SAM_INTEGRATION_GUIDE.md)
- [Enhancement Details](../docs/AI_INFERENCE_ENHANCEMENTS.md)
- [GitHub Repository](https://github.com/elliotttmiller/hvac-ai)
