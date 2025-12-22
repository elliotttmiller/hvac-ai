# Deep-Zoom Viewport Examples

This directory contains practical examples of using the Deep-Zoom Inference & Analysis Viewport.

## Examples

### 1. Basic Integration

The simplest way to use the deep-zoom viewport:

```typescript
// pages/analysis.tsx
'use client';

import dynamic from 'next/dynamic';

const DeepZoomAnalysis = dynamic(
  () => import('@/components/inference/DeepZoomInferenceAnalysis'),
  { 
    ssr: false,
    loading: () => <div>Loading...</div>
  }
);

export default function AnalysisPage() {
  return <DeepZoomAnalysis />;
}
```

### 2. Custom Annotation Store

Building a custom analysis component with full control:

```typescript
'use client';

import { useState } from 'react';
import { useAnnotationStore } from '@/lib/annotation-store';
import DeepZoomViewer from '@/components/viewer/DeepZoomViewer';
import AnnotationSidebar from '@/components/inference/AnnotationSidebar';

export default function CustomAnalysis() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [renderConfig, setRenderConfig] = useState({
    showLabels: true,
    showFill: true,
    showGrid: false,
    opacity: 0.2,
    lodLevel: 'detail' as const,
  });

  const {
    annotations,
    selectedId,
    hoveredId,
    confidenceThreshold,
    filteredAnnotations,
    hasUnsavedChanges,
    initializeFromSegments,
    updateAnnotation,
    deleteAnnotation,
    reclassifyAnnotation,
    setConfidenceThreshold,
    setSelectedId,
    setHoveredId,
    computeDelta,
    clearDirtyFlags,
    spatialIndex,
  } = useAnnotationStore();

  const handleAnalyze = async (file: File) => {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch('/api/v1/analyze', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    initializeFromSegments(data.segments);
    setImageUrl(URL.createObjectURL(file));
  };

  const handleSave = async () => {
    const delta = computeDelta();
    
    await fetch('/api/v1/annotations/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(delta),
    });
    
    clearDirtyFlags();
  };

  return (
    <div className="grid grid-cols-4 gap-4 h-screen">
      <div className="col-span-3">
        {imageUrl && (
          <DeepZoomViewer
            imageUrl={imageUrl}
            annotations={filteredAnnotations}
            selectedId={selectedId}
            hoveredId={hoveredId}
            renderConfig={renderConfig}
            onSelect={setSelectedId}
            onHover={setHoveredId}
            onAnnotationUpdate={updateAnnotation}
            spatialIndex={spatialIndex}
          />
        )}
      </div>
      <div className="col-span-1">
        <AnnotationSidebar
          annotations={annotations}
          filteredAnnotations={filteredAnnotations}
          selectedId={selectedId}
          confidenceThreshold={confidenceThreshold}
          onConfidenceChange={setConfidenceThreshold}
          onSelect={setSelectedId}
          onDelete={deleteAnnotation}
          onReclassify={reclassifyAnnotation}
          hasUnsavedChanges={hasUnsavedChanges}
          onSave={handleSave}
        />
      </div>
    </div>
  );
}
```

## Running the Examples

1. **Start the development servers:**
   ```bash
   # Terminal 1: Frontend
   npm run dev

   # Terminal 2: Backend
   cd python-services
   python hvac_analysis_service.py
   ```

2. **Navigate to the deep-zoom analysis page:**
   ```
   http://localhost:3000/deep-zoom-analysis
   ```

3. **Upload a test image and run analysis**

## Best Practices

1. **Always use dynamic imports** for OpenSeadragon components to avoid SSR issues
2. **Initialize spatial index** with `bulkLoad()` for better performance
3. **Use viewport culling** to only render visible annotations
4. **Implement debouncing** for expensive operations like filtering
5. **Clear dirty flags** after successful save operations
6. **Validate bboxes** to ensure they stay within image bounds
7. **Monitor performance** in production with FPS counters

## See Also

- [Full Documentation](../../docs/DEEP_ZOOM_VIEWPORT.md)
- [API Reference](../../docs/API.md)
- [Troubleshooting](../../docs/DEEP_ZOOM_VIEWPORT.md#troubleshooting)
