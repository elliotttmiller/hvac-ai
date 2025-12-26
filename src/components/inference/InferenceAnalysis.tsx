'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { Segment, CountResult } from '@/types/analysis';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Upload,
  Scan,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
  X,
  FileBarChart,
  Eye,
} from 'lucide-react';
import { toast } from 'sonner';

// --- Types ---
type AnalysisState = 'idle' | 'analyzing';
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';

// --- Color Palette for Classes ---
const CLASS_COLORS: Record<string, string> = {
  valve: '#ef4444', // Red
  instrument: '#3b82f6', // Blue
  sensor: '#10b981', // Emerald
  duct: '#f59e0b', // Amber
  default: '#8b5cf6', // Violet
};

function getColorForLabel(label: string) {
  const lower = label.toLowerCase();
  if (lower.includes('valve')) return CLASS_COLORS.valve;
  if (lower.includes('instrument') || lower.includes('computer') || lower.includes('plc')) return CLASS_COLORS.instrument;
  if (lower.includes('sensor')) return CLASS_COLORS.sensor;
  if (lower.includes('duct')) return CLASS_COLORS.duct;
  return CLASS_COLORS.default;
}

interface InferenceAnalysisProps {
  initialImage?: File | null;
  initialSegments?: Segment[];
  initialCount?: { total_objects_found: number; counts_by_category: Record<string, number> } | null;
}

export default function InferenceAnalysis({ initialImage, initialSegments, initialCount }: InferenceAnalysisProps) {
  // --- State ---
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>('idle');
  const [segments, setSegments] = useState<Segment[]>([]);
  const [countResult, setCountResult] = useState<CountResult | null>(null);
  
  // Visualization Options
  const [showLabels, setShowLabels] = useState(true);
  const [showFill, setShowFill] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Zoom & Pan state
  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);

  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenBgRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const scaleRef = useRef<number>(1);
  const imageRef = useRef<HTMLImageElement | null>(null);
  // Cache Path2D objects per-segment for fast hit-testing and redraws
  // Use a Map keyed by segment index (or id) to avoid reconstructing Path2D each frame
  const pathCacheRef = useRef<Map<number, Path2D>>(new Map());

  // --- 1. Image Initialization & Cleanup ---
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    imageRef.current = img;
    // Hydrate initial props if provided
    if (initialImage) setUploadedImage(initialImage);
    if (initialSegments) setSegments(initialSegments);
    if (initialCount) setCountResult(initialCount as CountResult);
    
    // Capture refs for cleanup (avoid stale closure issues)
    const offscreenBg = offscreenBgRef;
    const imageRefCopy = imageRef;
  const pathCache = pathCacheRef;
    
    // Cleanup on unmount: clear all caches and release memory
    return () => {
      // Clear offscreen canvas
      offscreenBg.current = null;
      // Clear any cached Path2D objects to free memory
      try {
        pathCache.current?.clear();
      } catch {
        /* defensive: ignore if not available */
      }
      // Release image reference
      if (imageRefCopy.current) {
        imageRefCopy.current.src = '';
        imageRefCopy.current = null;
      }
    };
  }, [initialImage, initialSegments, initialCount]);

  // Utility: resize a canvas to image size with DPR handling and optimization
  const resizeCanvasToImage = useCallback((canvas: HTMLCanvasElement, width: number, height: number) => {
    const dpr = window.devicePixelRatio || 1;
    // Subpixel precision: round to 2 decimal places for true 1:1 pixel mapping
    canvas.width = Math.round(width * dpr * 100) / 100;
    canvas.height = Math.round(height * dpr * 100) / 100;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true, // Off-main-thread rendering for better performance
      willReadFrequently: false, // Optimize for drawing operations
    });
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // Enable high-quality anti-aliasing
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
    }
  }, []);

  // Create offscreen canvas for background to cache high-res image
  const createOffscreenBackground = useCallback((displayW: number, displayH: number) => {
    const img = imageRef.current;
    if (!img || !img.src) return;
    const dpr = window.devicePixelRatio || 1;
    const offscreen = document.createElement('canvas');
    // Subpixel precision for offscreen canvas
    offscreen.width = Math.round(displayW * dpr * 100) / 100;
    offscreen.height = Math.round(displayH * dpr * 100) / 100;
    const ctx = offscreen.getContext('2d', {
      alpha: false, // Background doesn't need alpha
      desynchronized: true,
      willReadFrequently: false,
    });
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      // High-quality image rendering
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, 0, 0, displayW, displayH);
    }
    offscreenBgRef.current = offscreen;
  }, []);

  // Draw the background from offscreen cache with zoom/pan transform
  const drawBackground = useCallback(() => {
    const bg = bgCanvasRef.current;
    const offscreen = offscreenBgRef.current;
    if (!bg || !offscreen) return;
    const ctx = bg.getContext('2d');
    if (!ctx) return;
    ctx.save();
    ctx.clearRect(0, 0, bg.width, bg.height);
    // Apply zoom & pan transform
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    // Draw cached offscreen image
    const w = parseFloat(bg.style.width || '0');
    const h = parseFloat(bg.style.height || '0');
    ctx.drawImage(offscreen, 0, 0, w, h);
    ctx.restore();
  }, [zoom, panX, panY]);

  // Draw overlays (bounding boxes and labels) with zoom/pan transform
  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.src) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Apply zoom & pan transform
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    // Configure rendering quality
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.lineJoin = 'round'; // Smooth corners
    ctx.lineCap = 'round'; // Smooth line ends

    // Loop segments and draw oriented bounding boxes (OBB)
    for (let index = 0; index < segments.length; index++) {
      const segment = segments[index];
      const isHovered = index === hoveredIndex;
      const baseColor = getColorForLabel(segment.label);

  const obb = segment.obb;
      const s = scaleRef.current || 1;

      if (obb && typeof obb.x_center === 'number') {
        // Scale centers and sizes to display coordinates
        const cx = Math.round(obb.x_center * s * 100) / 100;
        const cy = Math.round(obb.y_center * s * 100) / 100;
        const w = Math.round(obb.width * s * 100) / 100;
        const h = Math.round(obb.height * s * 100) / 100;
        const rot = obb.rotation || 0;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(rot);

        // Draw stroke
        ctx.lineWidth = isHovered ? 3.5 : 2;
        ctx.strokeStyle = baseColor;
        ctx.strokeRect(-w / 2, -h / 2, w, h);

        // Draw fill if enabled
        if (showFill || isHovered) {
          ctx.globalAlpha = isHovered ? 0.15 : 0.08;
          ctx.fillStyle = baseColor;
          ctx.fillRect(-w / 2, -h / 2, w, h);
        }

        ctx.restore();

        // Labels: position at un-rotated top-left corner (approximate)
        if (showLabels || isHovered) {
          const labelText = `${segment.label} ${Math.round(segment.score * 100)}%`;
          ctx.font = isHovered
            ? 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif'
            : '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif';
          const textMetrics = ctx.measureText(labelText);
          const pad = 6;
          const textW = textMetrics.width + pad * 2;
          const textH = isHovered ? 26 : 20;

          // place label at top-left of the box without rotating the label itself
          const labelX = Math.round((obb.x_center - obb.width / 2) * s * 100) / 100;
          const labelY = Math.round((obb.y_center - obb.height / 2) * s * 100) / 100 - textH;
          const cornerRadius = 4;

          ctx.save();
          ctx.beginPath();
          ctx.moveTo(labelX + cornerRadius, labelY);
          ctx.lineTo(labelX + textW - cornerRadius, labelY);
          ctx.quadraticCurveTo(labelX + textW, labelY, labelX + textW, labelY + cornerRadius);
          ctx.lineTo(labelX + textW, labelY + textH - cornerRadius);
          ctx.quadraticCurveTo(labelX + textW, labelY + textH, labelX + textW - cornerRadius, labelY + textH);
          ctx.lineTo(labelX + cornerRadius, labelY + textH);
          ctx.quadraticCurveTo(labelX, labelY + textH, labelX, labelY + textH - cornerRadius);
          ctx.lineTo(labelX, labelY + cornerRadius);
          ctx.quadraticCurveTo(labelX, labelY, labelX + cornerRadius, labelY);
          ctx.closePath();

          if (isHovered) {
            const gradient = ctx.createLinearGradient(labelX, labelY, labelX, labelY + textH);
            gradient.addColorStop(0, baseColor);
            gradient.addColorStop(1, baseColor + 'cc');
            ctx.fillStyle = gradient;
          } else {
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
          }
          ctx.fill();
          ctx.restore();

          ctx.save();
          ctx.shadowColor = 'rgba(0,0,0,0.5)';
          ctx.shadowBlur = 2;
          ctx.shadowOffsetX = 0;
          ctx.shadowOffsetY = 1;
          ctx.fillStyle = '#fff';
          ctx.fillText(labelText, labelX + pad, labelY + (isHovered ? 17 : 14));
          ctx.restore();
        }

      } else {
        // Fallback: if no OBB, try drawing axis-aligned bbox for backward compatibility
        const [x, y, x2, y2] = segment.bbox || [0, 0, 0, 0];
        const sx = Math.round(x * s * 100) / 100;
        const sy = Math.round(y * s * 100) / 100;
        const sw = Math.round((x2 - x) * s * 100) / 100;
        const sh = Math.round((y2 - y) * s * 100) / 100;

        ctx.lineWidth = isHovered ? 3.5 : 2;
        ctx.strokeStyle = baseColor;
        ctx.strokeRect(sx, sy, sw, sh);
        if (showFill || isHovered) {
          ctx.save();
          ctx.globalAlpha = isHovered ? 0.15 : 0.08;
          ctx.fillStyle = baseColor;
          ctx.fillRect(sx, sy, sw, sh);
          ctx.restore();
        }
      }
    }
    ctx.restore();
  }, [segments, hoveredIndex, showLabels, showFill, zoom, panX, panY]);

  // Handle Image Load & resize canvases (with proper cleanup to prevent memory leaks)
  useEffect(() => {
    const img = imageRef.current;
    if (!img) return;

    const handleLoad = () => {
      const bg = bgCanvasRef.current;
      const overlay = overlayCanvasRef.current;
      if (!bg || !overlay) return;
      const parent = containerRef.current;
  const availableW = parent ? Math.max(200, parent.clientWidth) : Math.min(img.naturalWidth, window.innerWidth * 0.7);
  // maintain aspect ratio, compute display size to fit container width
  const aspect = img.naturalWidth ? img.naturalHeight / img.naturalWidth : 1;
  // Fit image to the available container width by default (allow upscaling to fill viewport)
  const displayW = Math.max(200, availableW - 32); // small padding
      const displayH = Math.round(displayW * aspect);
      // scale ref = display / natural
      scaleRef.current = displayW / img.naturalWidth;
      resizeCanvasToImage(bg, displayW, displayH);
      resizeCanvasToImage(overlay, displayW, displayH);
      // Create offscreen background cache
      createOffscreenBackground(displayW, displayH);
      // Clear path cache when image changes
      pathCacheRef.current.clear();
      drawBackground();
      drawOverlay();
    };
    
    const handleError = () => {
      toast.error('Failed to load image. Please try another file.', { duration: 4000 });
      console.error('[InferenceAnalysis] Image load error');
    };

    img.addEventListener('load', handleLoad);
    img.addEventListener('error', handleError);
    
    if (uploadedImage) {
      const url = URL.createObjectURL(uploadedImage);
      img.src = url;
      
      // Cleanup: revoke blob URL and remove listeners to prevent memory leaks
      return () => {
        URL.revokeObjectURL(url);
        img.removeEventListener('load', handleLoad);
        img.removeEventListener('error', handleError);
        // Clear src to release image memory
        img.src = '';
      };
    }
    
    // Cleanup when component unmounts
    return () => {
      img.removeEventListener('load', handleLoad);
      img.removeEventListener('error', handleError);
    };
  }, [uploadedImage, resizeCanvasToImage, createOffscreenBackground, drawBackground, drawOverlay]);

  // Redraw canvases when zoom/pan or segments change
  useEffect(() => {
    requestAnimationFrame(() => {
      drawBackground();
      drawOverlay();
    });
  }, [drawBackground, drawOverlay]);

  // Zoom & Pan handlers with smooth zoom range (0.5x to 3x industry standard)
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY * -0.0015; // Smoother zoom increment
    const newZoom = Math.min(Math.max(0.5, zoom + delta), 3);
    setZoom(newZoom);
  }, [zoom]);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0) { // left click
      setIsDragging(true);
      dragStartRef.current = { x: e.clientX - panX, y: e.clientY - panY };
    }
  }, [panX, panY]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
    dragStartRef.current = null;
  }, []);

  const handleMouseMoveCanvas = useCallback((e: React.MouseEvent) => {
    if (isDragging && dragStartRef.current) {
      setPanX(e.clientX - dragStartRef.current.x);
      setPanY(e.clientY - dragStartRef.current.y);
    }
  }, [isDragging]);

  const resetView = useCallback(() => {
    setZoom(1);
    setPanX(0);
    setPanY(0);
  }, []);

  // Mouse move handler: bbox hit-testing only
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const overlay = overlayCanvasRef.current;
    const img = imageRef.current;
    if (!overlay || !img) return;
    const rect = overlay.getBoundingClientRect();
    // Transform mouse coords by inverse of zoom/pan to get canvas-space coords
    const mouseX = (e.clientX - rect.left - panX) / zoom;
    const mouseY = (e.clientY - rect.top - panY) / zoom;

    // Iterate top-most first for OBB hit testing (fallback to bbox if absent)
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      const s = scaleRef.current || 1;

      if (seg.obb && typeof seg.obb.x_center === 'number') {
        const cx = seg.obb.x_center * s;
        const cy = seg.obb.y_center * s;
        const w = seg.obb.width * s;
        const h = seg.obb.height * s;
        const rot = seg.obb.rotation || 0;

        // Translate point into box-local coordinates by rotating by -rot
        const dx = mouseX - cx;
        const dy = mouseY - cy;
        const cosR = Math.cos(rot);
        const sinR = Math.sin(rot);
        const localX = dx * cosR + dy * sinR; // rotate by -rot
        const localY = -dx * sinR + dy * cosR;

        if (Math.abs(localX) <= w / 2 && Math.abs(localY) <= h / 2) {
          if (hoveredIndex !== i) setHoveredIndex(i);
          return;
        }
      } else if (seg.bbox) {
        const [x1, y1, x2, y2] = seg.bbox;
        if (mouseX >= x1 * s && mouseX <= x2 * s && mouseY >= y1 * s && mouseY <= y2 * s) {
          if (hoveredIndex !== i) setHoveredIndex(i);
          return;
        }
      }
    }
    if (hoveredIndex !== null) setHoveredIndex(null);
  }, [segments, hoveredIndex, zoom, panX, panY]);

  // --- 4. API Call ---
  const handleAnalyze = async () => {
    if (!uploadedImage) {
      toast.error('Please upload an image first');
      return;
    }
    
    setAnalysisState('analyzing');
    setSegments([]);
    setCountResult(null);

  const formData = new FormData();
  // Append both keys to be compatible with different backends/proxies
  formData.append('image', uploadedImage);
  formData.append('file', uploadedImage);
    
    // Industry-standard YOLO confidence threshold (0.50 = balanced precision/recall)
    // Lower values (0.25-0.40) = more detections but more false positives
    // Higher values (0.60-0.75) = fewer detections but higher precision
    formData.append('conf_threshold', '0.50');
    
    // NMS (Non-Maximum Suppression) threshold for overlapping detections
    // 0.45 is industry standard (removes duplicate bounding boxes effectively)
    formData.append('nms_threshold', '0.45');

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) {
        const errorText = await res.text();
        let errorMessage = 'Analysis failed';
        
        // Provide specific error feedback based on status code
        if (res.status === 413) {
          errorMessage = 'Image too large for server. Please resize to <10MB.';
        } else if (res.status === 415) {
          errorMessage = 'Unsupported image format. Use JPG, PNG, or TIFF.';
        } else if (res.status === 500) {
          errorMessage = 'Server error during analysis. Please try again.';
        } else if (res.status === 503) {
          errorMessage = 'Analysis service unavailable. Please try again later.';
        } else {
          try {
            const errorData = JSON.parse(errorText);
            errorMessage = errorData.message || errorMessage;
          } catch {
            // Use default error message if parsing fails
          }
        }
        
        toast.error(errorMessage, { duration: 5000 });
        throw new Error(errorMessage);
      }
      
      const data = await res.json();
      
      // Validate response data structure
      if (!data.segments || !Array.isArray(data.segments)) {
        toast.error('Invalid response format from analysis service');
        throw new Error('Invalid response structure');
      }
      
      setSegments(data.segments);
      setCountResult(data);
      
      const count = data.total_objects_found || data.segments.length;
      toast.success(`Analysis complete: Found ${count} component${count !== 1 ? 's' : ''}`, { duration: 3000 });
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to analyze image';
      if (!errorMsg.includes('Analysis')) {
        toast.error(errorMsg, { duration: 4000 });
      }
      console.error('[InferenceAnalysis] Analysis error:', e);
    } finally {
      setAnalysisState('idle');
    }
  };

  const handleDrop = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];
      
      // Industry-standard 10MB limit for optimal upload/processing performance
      const maxSize = 10 * 1024 * 1024; // 10MB
      if (file.size > maxSize) {
        toast.error(
          `File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds 10MB limit. For best results, resize images to ~1280px width (YOLO standard).`,
          { duration: 5000 }
        );
        return;
      }
      
      // Validate image format (including TIFF for technical drawings)
      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
      const fileType = file.type.toLowerCase();
      if (!validTypes.includes(fileType) && !file.name.match(/\.(jpg|jpeg|png|tiff|tif|bmp)$/i)) {
        toast.error('Please upload a valid image file (JPG, PNG, TIFF, or BMP)', { duration: 4000 });
        return;
      }
      
      // Success feedback
      setUploadedImage(file);
      toast.success(`Loaded ${file.name} (${(file.size / 1024).toFixed(0)}KB)`, { duration: 2000 });
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: { 
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/tiff': ['.tiff', '.tif'],
      'image/bmp': ['.bmp']
    },
    maxFiles: 1,
    multiple: false
  });

  // --- Render ---
  return (
    <div className="container mx-auto py-8 space-y-6">
      
      {/* Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle>HVAC Blueprint Analysis</CardTitle>
          <CardDescription>Upload a P&ID or Floor Plan to detect symbols</CardDescription>
        </CardHeader>
        <CardContent>
          {!uploadedImage ? (
            <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700">Drop blueprint here</p>
              <p className="text-sm text-gray-500">Supports PNG, JPG (High Res recommended)</p>
            </div>
          ) : (
            <div className="flex items-center justify-between bg-slate-50 p-4 rounded-lg border">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 bg-blue-100 rounded flex items-center justify-center text-blue-600">
                  <Scan className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium text-slate-900">{uploadedImage.name}</p>
                  <p className="text-xs text-slate-500">{(uploadedImage.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleAnalyze} disabled={analysisState === 'analyzing'}>
                  {analysisState === 'analyzing' ? <Loader2 className="animate-spin mr-2 h-4 w-4" /> : <Scan className="mr-2 h-4 w-4" />}
                  Analyze Diagram
                </Button>
                <Button variant="ghost" size="icon" onClick={() => { setUploadedImage(null); setSegments([]); setCountResult(null); }}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Visualization Canvas */}
      {uploadedImage && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Main Canvas Area */}
          <Card className="lg:col-span-3 overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between py-4">
              <CardTitle className="text-base">Visual Inspection</CardTitle>
              
              {/* Visualization Controls */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch id="show-labels" checked={showLabels} onCheckedChange={setShowLabels} />
                  <Label htmlFor="show-labels" className="text-sm cursor-pointer">Labels</Label>
                </div>
                <div className="flex items-center gap-2">
                  <Switch id="show-fill" checked={showFill} onCheckedChange={setShowFill} />
                  <Label htmlFor="show-fill" className="text-sm cursor-pointer">Fill</Label>
                </div>
                <div className="flex items-center gap-2 border-l pl-4" role="group" aria-label="Zoom controls">
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => setZoom(Math.min(3, zoom * 1.2))}
                    aria-label="Zoom in (or press + key)"
                    disabled={zoom >= 3}
                  >
                    Zoom In
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={() => setZoom(Math.max(0.5, zoom / 1.2))}
                    aria-label="Zoom out (or press - key)"
                    disabled={zoom <= 0.5}
                  >
                    Zoom Out
                  </Button>
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={resetView}
                    aria-label="Reset view to default (or press 0 key)"
                  >
                    Reset View
                  </Button>
                  <span className="text-xs text-gray-500" role="status" aria-live="polite" aria-atomic="true">
                    {Math.round(zoom * 100)}%
                  </span>
                </div>
              </div>
            </CardHeader>
            
            <CardContent
              ref={containerRef}
              className="p-0 bg-slate-900 relative min-h-[500px] flex items-center justify-center overflow-auto"
              role="img"
              aria-label={`HVAC blueprint visualization with ${segments.length} detected components. Use +/- to zoom, arrow keys to pan.`}
              tabIndex={0}
              onWheel={handleWheel}
              onKeyDown={(e) => {
                // Keyboard controls for accessibility
                if (e.key === '+' || e.key === '=') {
                  e.preventDefault();
                  setZoom(Math.min(3, zoom * 1.2));
                } else if (e.key === '-' || e.key === '_') {
                  e.preventDefault();
                  setZoom(Math.max(0.5, zoom / 1.2));
                } else if (e.key === '0') {
                  e.preventDefault();
                  resetView();
                } else if (e.key === 'ArrowUp') {
                  e.preventDefault();
                  setPanY(panY + 30);
                } else if (e.key === 'ArrowDown') {
                  e.preventDefault();
                  setPanY(panY - 30);
                } else if (e.key === 'ArrowLeft') {
                  e.preventDefault();
                  setPanX(panX + 30);
                } else if (e.key === 'ArrowRight') {
                  e.preventDefault();
                  setPanX(panX - 30);
                }
              }}
            >
              <div className="relative w-full flex items-center justify-center">
                <canvas
                  ref={bgCanvasRef}
                  className="max-w-full h-auto shadow-2xl block"
                  aria-hidden="true"
                />
                <canvas
                  ref={overlayCanvasRef}
                  onMouseMove={handleMouseMove}
                  onMouseDown={handleMouseDown}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={() => { setHoveredIndex(null); setIsDragging(false); }}
                  className="absolute top-0 left-0 max-w-full h-auto"
                  aria-label={`${segments.length} component overlays`}
                  role="presentation"
                  style={{ pointerEvents: 'auto', touchAction: 'none' }}
                />
              </div>
              {/* Hover Tooltip Overlay */}
              {hoveredIndex !== null && segments[hoveredIndex] && (
                <div 
                  className="absolute bottom-4 left-4 bg-black/80 backdrop-blur text-white p-3 rounded-lg shadow-xl border border-white/10 z-10"
                >
                  <p className="font-bold text-sm text-blue-400">{segments[hoveredIndex].label}</p>
                  <p className="text-xs text-gray-300">Confidence: {(segments[hoveredIndex].score * 100).toFixed(1)}%</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Sidebar Stats */}
          <Card className="lg:col-span-1 h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FileBarChart className="h-4 w-4" /> Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              {countResult ? (
                <div className="space-y-4">
                  <div className="text-center p-4 bg-slate-50 rounded-lg border">
                    <div className="text-3xl font-bold text-slate-900">{countResult.total_objects_found}</div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide font-semibold">Components Found</div>
                  </div>
                  
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                    {Object.entries(countResult.counts_by_category)
                      .sort(([,a], [,b]) => b - a)
                      .map(([label, count]) => (
                      <div key={label} className="flex items-center justify-between text-sm p-2 hover:bg-slate-50 rounded group">
                        <span className="truncate max-w-[140px] text-slate-700" title={label}>{label}</span>
                        <Badge variant="secondary" className="group-hover:bg-blue-100 group-hover:text-blue-700 transition-colors">
                          {count}
                        </Badge>
                      </div>
                    ))}
                  </div>
                  
                  <Button variant="outline" className="w-full text-xs" onClick={() => toast.info("Export feature coming soon")}>
                    <Download className="mr-2 h-3 w-3" /> Export CSV
                  </Button>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400 text-sm">
                  Run analysis to see component breakdown
                </div>
              )}
            </CardContent>
          </Card>

        </div>
      )}
    </div>
  );
}