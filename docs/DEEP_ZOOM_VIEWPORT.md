# Deep-Zoom Inference & Analysis Viewport

## Overview

The Deep-Zoom Inference & Analysis Viewport is a Google Maps-style navigation system for HVAC blueprint analysis, designed to handle massive raster images (10,000px+ width) with thousands of AI detections without browser latency or memory leaks.

## Features

### Core Capabilities
- **Tile-Based Rendering**: Seamless pan and zoom using OpenSeadragon
- **Viewport Culling**: Only visible annotations are rendered for optimal performance
- **Spatial Indexing**: O(log n) hit detection using R-Tree data structure
- **Canvas Overlay**: Hardware-accelerated rendering of thousands of bounding boxes
- **Human-in-the-Loop Editing**: Reclassify, delete, and annotate detections
- **Delta-Based Saving**: Minimal network overhead by only sending changes
- **Real-Time Filtering**: Confidence threshold slider with instant updates
- **Bi-Directional Sync**: Canvas selection ↔ Sidebar highlighting

### Performance Optimizations
- Spatial indexing for fast hit-testing
- Viewport culling renders only visible items
- Off-screen canvas caching
- RequestAnimationFrame-based rendering
- Dirty tracking for efficient state updates
- Optimized for 10,000+ annotations

## Architecture

### Frontend Components

#### DeepZoomViewer
Main viewport component using OpenSeadragon for tile-based rendering.

```typescript
import DeepZoomViewer from '@/components/viewer/DeepZoomViewer';

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
```

**Features:**
- Smooth pan/zoom with inertial drag
- Canvas overlay for annotations
- Polygon and bounding box rendering
- Level of Detail (LOD) management
- SAHI grid visualization overlay

#### AnnotationSidebar
Filterable sidebar for annotation management.

```typescript
import AnnotationSidebar from '@/components/inference/AnnotationSidebar';

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
```

**Features:**
- Search and filter annotations
- Sort by confidence, label, or position
- Category summary with counts
- Confidence threshold slider
- Edit and delete actions
- Unsaved changes indicator

#### DeepZoomInferenceAnalysis
Complete integrated component.

```typescript
import DeepZoomInferenceAnalysis from '@/components/inference/DeepZoomInferenceAnalysis';

export default function MyPage() {
  return <DeepZoomInferenceAnalysis />;
}
```

### State Management

#### Annotation Store
Centralized state with dirty tracking and delta computation.

```typescript
import { useAnnotationStore } from '@/lib/annotation-store';

const {
  annotations,
  selectedId,
  hoveredId,
  confidenceThreshold,
  filteredAnnotations,
  hasUnsavedChanges,
  
  initializeFromSegments,
  addAnnotation,
  updateAnnotation,
  deleteAnnotation,
  reclassifyAnnotation,
  setConfidenceThreshold,
  setSelectedId,
  setHoveredId,
  computeDelta,
  clearDirtyFlags,
  
  spatialIndex,
  editHistory,
} = useAnnotationStore();
```

**Key Methods:**
- `initializeFromSegments(segments)` - Load YOLO results
- `addAnnotation(annotation)` - Create new annotation (phantom)
- `updateAnnotation(id, updates)` - Modify existing annotation
- `deleteAnnotation(id)` - Remove annotation
- `reclassifyAnnotation(id, newLabel)` - Change class label
- `computeDelta()` - Generate save payload
- `clearDirtyFlags()` - Reset after successful save

### Spatial Indexing

Efficient viewport culling and hit-testing using R-Tree.

```typescript
import { SpatialAnnotationIndex } from '@/lib/spatial-index';

const spatialIndex = new SpatialAnnotationIndex();

// Insert annotations
spatialIndex.bulkLoad(annotations);

// Search visible annotations
const visible = spatialIndex.search({
  x: viewportX,
  y: viewportY,
  width: viewportWidth,
  height: viewportHeight,
});

// Find annotation at point
const hit = spatialIndex.findAtPoint(mouseX, mouseY);
```

## API Endpoints

### POST /api/v1/analyze
Standard YOLO inference endpoint.

```typescript
const formData = new FormData();
formData.append('image', file);
formData.append('conf_threshold', '0.50');
formData.append('nms_threshold', '0.45');

const response = await fetch('/api/v1/analyze', {
  method: 'POST',
  body: formData,
});

const data = await response.json();
// Returns: { segments: [...], total_objects_found: N, ... }
```

### POST /api/v1/annotations/save
Delta-based annotation save endpoint.

```typescript
const delta = {
  added: [...],      // New annotations
  modified: [...],   // Changed annotations
  deleted: [...],    // Deleted annotation IDs
  verification_status: 'verified'
};

const response = await fetch('/api/v1/annotations/save', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(delta),
});
```

**Response:**
```json
{
  "status": "success",
  "save_id": "abc123",
  "added_count": 5,
  "modified_count": 3,
  "deleted_count": 2,
  "verification_status": "verified",
  "timestamp": 1234567890
}
```

## Type Definitions

### EditableAnnotation
```typescript
interface EditableAnnotation {
  id: string;
  label: string;
  score: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  polygon?: number[][];
  isDirty: boolean;
  isNew: boolean;
}
```

### RenderConfig
```typescript
interface RenderConfig {
  showLabels: boolean;
  showFill: boolean;
  showGrid: boolean;      // SAHI grid overlay
  opacity: number;        // Fill opacity (0-1)
  lodLevel: LODLevel;     // 'overview' | 'medium' | 'detail'
}
```

### DeltaSavePayload
```typescript
interface DeltaSavePayload {
  added: EditableAnnotation[];
  modified: EditableAnnotation[];
  deleted: string[];
  verification_status?: 'pending' | 'verified' | 'rejected';
}
```

## Usage Guide

### Basic Setup

1. **Import the Component**
```typescript
import DeepZoomInferenceAnalysis from '@/components/inference/DeepZoomInferenceAnalysis';
```

2. **Add to Your Page**
```typescript
// For SSR compatibility, use dynamic import
import dynamic from 'next/dynamic';

const DeepZoomAnalysis = dynamic(
  () => import('@/components/inference/DeepZoomInferenceAnalysis'),
  { ssr: false }
);

export default function AnalysisPage() {
  return <DeepZoomAnalysis />;
}
```

### Navigation Controls

**Mouse:**
- **Drag** - Pan viewport
- **Scroll** - Zoom in/out
- **Double-click** - Zoom in
- **Click annotation** - Select

**Keyboard:**
- **+/=** - Zoom in
- **-** - Zoom out
- **0** - Reset view
- **Arrow keys** - Pan viewport

### Editing Workflow

1. **Upload Blueprint**: Drag and drop image file
2. **Run Analysis**: Click "Analyze Blueprint" button
3. **Review Detections**: Browse results in sidebar
4. **Filter Results**: Adjust confidence threshold slider
5. **Edit Annotations**:
   - Click Edit icon to reclassify
   - Click Delete icon to remove
   - Select annotation to view details
6. **Save Changes**: Click "Save Changes" button when ready

### Confidence Filtering

The confidence threshold slider allows real-time filtering:
- **Low threshold (0-40%)**: Show all detections (more false positives)
- **Medium threshold (40-60%)**: Balanced precision/recall
- **High threshold (60-100%)**: Only high-confidence detections

### Class Color Coding

Each HVAC category has a distinct color:
- **Valve**: Red (#ef4444)
- **Instrument**: Blue (#3b82f6)
- **Sensor**: Green (#10b981)
- **Duct**: Orange (#f59e0b)
- **Vent**: Purple (#8b5cf6)

## Performance Benchmarks

Expected performance on typical hardware:

- **10,000 annotations**: 60fps rendering
- **Viewport culling**: <5ms per frame
- **Spatial indexing**: O(log n) hit detection
- **Memory usage**: <500MB for 10k annotations
- **Initial load**: <2s for 5000px image

## Future Enhancements

Planned features (not yet implemented):
- [ ] Deep Zoom Image (DZI) pyramid generation
- [ ] Tile-based image serving
- [ ] WebSocket streaming for progressive results
- [ ] Bounding box resize handles
- [ ] Phantom annotation drawing tool
- [ ] Undo/redo system
- [ ] Heatmap visualization
- [ ] Batch annotation operations
- [ ] Export to various formats

## Troubleshooting

### Common Issues

**Issue: "document is not defined" error**
- **Solution**: Use dynamic import with `{ ssr: false }` in Next.js pages

**Issue: Annotations not rendering**
- **Solution**: Check that `initializeFromSegments()` was called after analysis

**Issue: Poor performance with many annotations**
- **Solution**: Verify viewport culling is enabled and spatial index is populated

**Issue: Canvas not updating**
- **Solution**: Ensure `renderConfig` changes trigger re-render

## Technical Details

### OpenSeadragon Configuration
```javascript
{
  showNavigationControl: true,
  showNavigator: true,
  navigatorPosition: 'BOTTOM_RIGHT',
  animationTime: 0.5,
  springStiffness: 10,
  maxZoomPixelRatio: 3,
  minZoomLevel: 0.5,
  visibilityRatio: 0.5,
}
```

### Canvas Rendering Context
```javascript
{
  alpha: true,
  desynchronized: true,  // Off-main-thread rendering
  willReadFrequently: false,
}
```

## References

- [OpenSeadragon Documentation](https://openseadragon.github.io/)
- [R-Tree Spatial Index](https://github.com/mourner/rbush)
- [SAHI - Sliced Aided Hyper Inference](https://github.com/obss/sahi)
- [YOLO Ultralytics](https://docs.ultralytics.com/)

## License

Part of the HVAC AI Platform - See repository LICENSE for details.
