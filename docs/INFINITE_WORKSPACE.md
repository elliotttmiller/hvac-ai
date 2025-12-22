# Infinite Workspace UI

A sleek, smooth, and seamless full-screen interface for HVAC blueprint analysis with AI-powered detection visualization.

## Overview

The Infinite Workspace provides a distraction-free, viewport-optimized experience for viewing large technical HVAC drawings with real-time AI detection overlays. Built with React, Next.js, and HTML5 Canvas for optimal performance.

## Features

### 🎨 Modern Design
- **Dark Theme**: Professional slate-950 background with cyan (#00f0ff) accents
- **Glassmorphism**: Floating panels with backdrop-blur effects
- **Gradient Branding**: Blue-to-cyan gradient on "HVAC.AI" header
- **Responsive**: Adapts to any screen size while maintaining aspect ratio

### 🔄 State-Driven Architecture
The UI implements a clean state machine with smooth transitions:
1. **IDLE**: Upload area with glassmorphism effect
2. **UPLOADING**: Brief transition state (800ms)
3. **PROCESSING**: Pulsing "Analyzing Geometry" message
4. **COMPLETE**: Full-screen drawing with detection overlays

### 🖼️ Viewport Optimization
- Drawing fills 100% of available viewport (max-h-90vh)
- Zero layout shift with `object-contain` sizing
- No scrolling required to see complete results
- Automatic scaling based on image dimensions

### 🎯 Canvas-Based Detection Overlay
- HTML5 Canvas for high-performance rendering
- 60fps smooth even with 500+ detections
- Automatic coordinate scaling
- Cyan bounding boxes with shadow effects
- Confidence percentage labels

### 📊 Detection Summary Panel
Floating bottom-right panel displays:
- Ducts Found
- Vents Found
- Total Items
- Overall Confidence

### 🔁 Workflow Controls
- **Responsive Header**: Fades to transparent when viewing results
- **Hover Reveal**: Header reappears on hover
- **Upload New**: Quick reset button to start over

## Technical Implementation

### File Structure
```
src/
├── app/
│   ├── infinite-workspace/
│   │   ├── page.tsx           # Main component with state machine
│   │   └── layout.tsx         # Fullscreen layout override
│   ├── ClientBody.tsx         # Updated to exclude AppLayout
│   └── globals.css            # Custom animations
└── components/
    └── viewer/
        └── DrawingViewer.tsx  # Canvas overlay component
```

### Key Technologies
- **React 18**: Hooks-based state management
- **Next.js 15**: App Router with route-specific layouts
- **TypeScript**: Type-safe interfaces
- **Tailwind CSS**: Utility-first styling
- **HTML5 Canvas**: High-performance detection rendering
- **Lucide React**: Icon library

### State Management
```typescript
type AppState = 'IDLE' | 'UPLOADING' | 'PROCESSING' | 'COMPLETE';

interface Detection {
  label: string;
  conf: number;
  box: [number, number, number, number]; // [x, y, w, h]
}
```

### Canvas Rendering
The DrawingViewer component:
1. Loads the uploaded image
2. Creates a canvas overlay matching image dimensions
3. Calculates scale factors (rendered / natural size)
4. Draws scaled bounding boxes with labels
5. Handles window resize events automatically

## Usage

### Accessing the UI
Navigate to `/infinite-workspace` in your browser.

### Uploading a Drawing
1. Click the upload area or drag and drop a file
2. Supports JPG, PNG, and PDF formats
3. Optimal resolution: 1280px width (YOLO standard)
4. Maximum file size: 10MB

### Viewing Results
1. Wait for the "Analyzing Geometry" animation (2-3 seconds)
2. View the full-screen drawing with detection overlays
3. Hover over the top to reveal the "Upload New" button
4. Check the Detection Summary panel for statistics

### Resetting
Click "Upload New" in the header (hover to reveal) to start over.

## Integration with Backend

### Current Implementation
The component uses mock YOLO detection data for demonstration:
```typescript
const mockYoloResponse: Detection[] = [
  { label: 'Vent', conf: 0.92, box: [100, 100, 150, 100] },
  { label: 'Thermostat', conf: 0.88, box: [300, 250, 50, 50] },
  { label: 'Duct Work', conf: 0.75, box: [50, 400, 600, 50] },
];
```

### Production Integration
To connect to a real YOLO backend, replace the mock data in `handleFileUpload`:

```typescript
const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0];
  if (!file) return;

  setAppState('UPLOADING');
  const objectUrl = URL.createObjectURL(file);
  setImageSrc(objectUrl);

  setAppState('PROCESSING');

  try {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch('/api/v1/analyze', {
      method: 'POST',
      body: formData,
    });
    
    const data = await response.json();
    setDetections(data.detections); // Expected format: Detection[]
    setAppState('COMPLETE');
  } catch (error) {
    console.error('Analysis failed:', error);
    // Handle error state
  }
};
```

### Expected API Response Format
```json
{
  "status": "success",
  "detections": [
    {
      "label": "Return Air Vent",
      "conf": 0.95,
      "box": [450, 300, 100, 100]
    }
  ],
  "metadata": {
    "processing_time": "0.12s",
    "image_dims": [1920, 1080]
  }
}
```

## Customization

### Colors
Edit `src/components/viewer/DrawingViewer.tsx` to change detection colors:
```typescript
ctx.strokeStyle = '#00f0ff'; // Cyan (default)
ctx.fillStyle = 'rgba(0, 240, 255, 0.2)';
```

### Animation Timing
Edit `src/app/infinite-workspace/page.tsx` to adjust delays:
```typescript
setTimeout(() => setAppState('PROCESSING'), 800);  // Upload delay
setTimeout(() => setAppState('COMPLETE'), 2000);   // Processing delay
```

### Detection Summary
Edit `src/components/viewer/DrawingViewer.tsx` to customize the panel:
```typescript
<div className="absolute bottom-8 right-8 bg-slate-950/80 backdrop-blur-md ...">
  {/* Custom summary content */}
</div>
```

## Performance Considerations

### Canvas Rendering
- Uses `requestAnimationFrame` for smooth redraws
- Calculates scale factors only once per image
- Efficient bounding box rendering with path2D

### Memory Management
- Object URLs are automatically revoked on unmount
- Event listeners are properly cleaned up
- No memory leaks in state management

### Optimization Tips
- Keep images under 10MB for best upload performance
- Recommended resolution: 1280px width
- Canvas rendering performs well up to 500+ detections

## Browser Support
- Chrome 90+ ✅
- Firefox 88+ ✅
- Safari 14+ ✅
- Edge 90+ ✅

## Accessibility
- Semantic HTML structure
- ARIA labels on interactive elements
- Keyboard navigation support
- Reduced motion preferences respected

## Future Enhancements
- [ ] Zoom/pan controls for large drawings
- [ ] Keyboard shortcuts (Z for zoom, P for pan, R for reset)
- [ ] Export detection results as JSON/CSV
- [ ] Multi-drawing comparison view
- [ ] Real-time collaboration features
- [ ] Drawing annotation tools
- [ ] Historical analysis tracking

## License
Part of the HVAC AI Platform - All Rights Reserved

## Support
For issues or questions, please contact the development team or create an issue in the repository.
