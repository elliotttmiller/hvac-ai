'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { Segment, CountResult } from '@/types/analysis';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Scan,
  FileBarChart,
  Maximize,
  Minimize,
  RotateCcw,
  Upload
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';

const CLASS_COLORS: Record<string, string> = {
  valve: '#ef4444', 
  instrument: '#3b82f6', 
  sensor: '#10b981', 
  duct: '#f59e0b', 
  default: '#8b5cf6', 
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
  const isControlled = !!(initialImage || initialSegments);

  const [uploadedImage, setUploadedImage] = useState<File | null>(initialImage || null);
  const [segments, setSegments] = useState<Segment[]>(initialSegments || []);
  const [countResult, setCountResult] = useState<CountResult | null>(
    initialCount ? (initialCount as CountResult) : null
  );
  
  const [showLabels, setShowLabels] = useState(true);
  const [showFill, setShowFill] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

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
  const pathCacheRef = useRef<Map<string, Path2D>>(new Map());
  // Stable refs to hold latest draw functions so effects can call them
  const drawBackgroundRef = useRef<() => void>(() => {});
  const drawOverlayRef = useRef<() => void>(() => {});

  useEffect(() => {
    if (initialImage) setUploadedImage(initialImage);
    if (initialSegments) {
      setSegments(initialSegments);
      pathCacheRef.current.clear();
    }
    if (initialCount) setCountResult(initialCount as CountResult);
  }, [initialImage, initialSegments, initialCount]);

  useEffect(() => {
    if (typeof window === 'undefined' || !uploadedImage) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    const url = URL.createObjectURL(uploadedImage);
    img.src = url;
    imageRef.current = img;

    // copy pathCacheRef to a local variable for stable cleanup reference
    const pathCache = pathCacheRef.current;

    const handleLoad = () => {
      const bg = bgCanvasRef.current;
      const overlay = overlayCanvasRef.current;
      if (!bg || !overlay) return;

      const parent = containerRef.current;
      const availableW = parent ? Math.max(200, parent.clientWidth) : window.innerWidth * 0.7;
      
      const aspect = img.naturalHeight / img.naturalWidth;
      const displayW = Math.max(200, availableW - 32);
      const displayH = Math.round(displayW * aspect);

      scaleRef.current = displayW / img.naturalWidth;
      
      resizeCanvasToImage(bg, displayW, displayH);
      resizeCanvasToImage(overlay, displayW, displayH);
      createOffscreenBackground(displayW, displayH);
      
      requestAnimationFrame(() => {
        // call the latest functions via refs so this effect doesn't need to
        // include drawBackground/drawOverlay in its dependency array
        drawBackgroundRef.current();
        drawOverlayRef.current();
      });
    };

    img.addEventListener('load', handleLoad);
    return () => {
      URL.revokeObjectURL(url);
      img.removeEventListener('load', handleLoad);
      pathCache.clear();
    };
  }, [uploadedImage]);

  const resizeCanvasToImage = (canvas: HTMLCanvasElement, width: number, height: number) => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const createOffscreenBackground = (w: number, h: number) => {
    const img = imageRef.current;
    if (!img) return;
    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    const ctx = offscreen.getContext('2d', { alpha: false });
    if (ctx) {
      ctx.drawImage(img, 0, 0, w, h);
      offscreenBgRef.current = offscreen;
    }
  };

  const drawBackground = useCallback(() => {
    const bg = bgCanvasRef.current;
    const offscreen = offscreenBgRef.current;
    if (!bg || !offscreen) return;
    const ctx = bg.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, bg.width, bg.height);
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);
    ctx.drawImage(offscreen, 0, 0, parseFloat(bg.style.width), parseFloat(bg.style.height));
    ctx.restore();
  }, [zoom, panX, panY]);

  // Keep refs up-to-date so effects can call the latest implementations
  useEffect(() => {
    drawBackgroundRef.current = drawBackground;
  }, [drawBackground]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(panX, panY);
    ctx.scale(zoom, zoom);

    segments.forEach((seg, index) => {
      const isHovered = index === hoveredIndex;
      const color = getColorForLabel(seg.label);
      const s = scaleRef.current;
      // Use a stable string key for the path cache (seg.id may be undefined or numeric)
      const idKey = String(seg.id ?? index);
      let path = pathCacheRef.current.get(idKey);
      
      if (!path) {
        path = new Path2D();
        
        // 1. Use OBB Corners (Provided by Backend)
        if (seg.points && seg.points.length === 4) {
          path.moveTo(seg.points[0][0] * s, seg.points[0][1] * s);
          for (let i = 1; i < seg.points.length; i++) {
            path.lineTo(seg.points[i][0] * s, seg.points[i][1] * s);
          }
          path.closePath();
        } 
        // 2. Fallback: Calculate from OBB Parameters
        else if (seg.obb) {
          const { x_center, y_center, width, height, rotation } = seg.obb;
          const cx = x_center * s;
          const cy = y_center * s;
          const w = width * s;
          const h = height * s;
          
          const cos = Math.cos(rotation);
          const sin = Math.sin(rotation);
          const dx = w / 2;
          const dy = h / 2;
          
          path.moveTo(cx + (-dx)*cos - (-dy)*sin, cy + (-dx)*sin + (-dy)*cos);
          path.lineTo(cx + (dx)*cos - (-dy)*sin, cy + (dx)*sin + (-dy)*cos);
          path.lineTo(cx + (dx)*cos - (dy)*sin, cy + (dx)*sin + (dy)*cos);
          path.lineTo(cx + (-dx)*cos - (dy)*sin, cy + (-dx)*sin + (dy)*cos);
          path.closePath();
        } 
        // 3. Fallback: Standard BBox
        else if (seg.bbox) {
           const [x1, y1, x2, y2] = seg.bbox;
           path.rect(x1 * s, y1 * s, (x2 - x1) * s, (y2 - y1) * s);
        }
        
        pathCacheRef.current.set(idKey, path);
      }

      ctx.lineWidth = isHovered ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.stroke(path);

      if (showFill || isHovered) {
        ctx.fillStyle = color;
        ctx.globalAlpha = isHovered ? 0.3 : 0.15;
        ctx.fill(path);
        ctx.globalAlpha = 1.0;
      }

      if (showLabels || isHovered) {
        let lx = 0, ly = 0;
        if (seg.bbox) {
            lx = seg.bbox[0] * s;
            ly = seg.bbox[1] * s;
        } else if (seg.obb) {
            lx = (seg.obb.x_center - seg.obb.width/2) * s;
            ly = (seg.obb.y_center - seg.obb.height/2) * s;
        }

        ctx.font = isHovered ? 'bold 14px sans-serif' : '12px sans-serif';
        const text = `${seg.label} ${(seg.score * 100).toFixed(0)}%`;
        const metrics = ctx.measureText(text);
        
        ctx.fillStyle = 'rgba(0,0,0,0.8)';
        ctx.fillRect(lx, ly - 20, metrics.width + 8, 20);
        ctx.fillStyle = color;
        ctx.fillText(text, lx + 4, ly - 6);
      }
    });

    ctx.restore();
  }, [segments, hoveredIndex, showFill, showLabels, zoom, panX, panY]);

  useEffect(() => {
    drawOverlayRef.current = drawOverlay;
  }, [drawOverlay]);

  useEffect(() => {
    requestAnimationFrame(() => {
      drawBackground();
      drawOverlay();
    });
  }, [drawBackground, drawOverlay]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (!uploadedImage) return;
    const delta = e.deltaY * -0.001;
    setZoom(z => Math.min(Math.max(0.5, z + delta), 4));
  }, [uploadedImage]);

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    dragStartRef.current = { x: e.clientX - panX, y: e.clientY - panY };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && dragStartRef.current) {
      setPanX(e.clientX - dragStartRef.current.x);
      setPanY(e.clientY - dragStartRef.current.y);
      return;
    }

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    
    const mouseX = (e.clientX - rect.left - panX) / zoom;
    const mouseY = (e.clientY - rect.top - panY) / zoom;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let found = null;
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      const path = pathCacheRef.current.get(String(seg.id ?? i));
      if (path && ctx.isPointInPath(path, mouseX, mouseY)) {
        found = i;
        break;
      }
    }
    setHoveredIndex(found);
  };

  const onDrop = useCallback((files: File[]) => { if(files.length) setUploadedImage(files[0]); }, []);
  const { getRootProps, getInputProps } = useDropzone({ onDrop, disabled: isControlled, accept: { 'image/*': [] } });

  if (!uploadedImage && !isControlled) {
    return (
       <Card className="border-dashed border-2 p-10 text-center cursor-pointer hover:bg-slate-50" {...getRootProps()}>
         <input {...getInputProps()} />
         <Upload className="h-10 w-10 mx-auto text-slate-400 mb-4"/>
         <p>Drop image to analyze</p>
       </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-full">
      <Card className="lg:col-span-3 overflow-hidden flex flex-col h-[600px]">
        <CardHeader className="py-3 px-4 border-b flex flex-row items-center justify-between bg-slate-50">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Scan className="h-4 w-4" /> Visual Inspection
          </CardTitle>
          <div className="flex items-center gap-2">
             <div className="flex items-center gap-2 mr-4">
                <Switch checked={showLabels} onCheckedChange={setShowLabels} id="lbl" />
                <Label htmlFor="lbl" className="text-xs">Labels</Label>
             </div>
             <Button variant="outline" size="icon" onClick={() => setZoom(z => Math.min(4, z * 1.2))}><Maximize className="h-4 w-4" /></Button>
             <Button variant="outline" size="icon" onClick={() => setZoom(z => Math.max(0.5, z / 1.2))}><Minimize className="h-4 w-4" /></Button>
             <Button variant="outline" size="icon" onClick={() => { setZoom(1); setPanX(0); setPanY(0); }}><RotateCcw className="h-4 w-4" /></Button>
          </div>
        </CardHeader>
        
        <div 
          ref={containerRef}
          className="flex-1 bg-slate-900 relative overflow-hidden cursor-move"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={() => setIsDragging(false)}
          onMouseLeave={() => { setIsDragging(false); setHoveredIndex(null); }}
        >
          <canvas ref={bgCanvasRef} className="absolute top-0 left-0" />
          <canvas ref={overlayCanvasRef} className="absolute top-0 left-0" />
          
          {hoveredIndex !== null && segments[hoveredIndex] && (
            <div className="absolute bottom-4 left-4 bg-black/80 text-white p-3 rounded shadow-lg pointer-events-none z-10">
              <p className="font-bold text-sm text-blue-400">{segments[hoveredIndex].label}</p>
              <p className="text-xs">Confidence: {(segments[hoveredIndex].score * 100).toFixed(1)}%</p>
            </div>
          )}
        </div>
      </Card>

      <Card className="lg:col-span-1 flex flex-col h-[600px]">
        <CardHeader className="py-3 border-b">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <FileBarChart className="h-4 w-4" /> Components
          </CardTitle>
        </CardHeader>
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {countResult ? (
            Object.entries(countResult.counts_by_category)
              .sort(([,a], [,b]) => b - a)
              .map(([label, count]) => (
                <div key={label} className="flex justify-between items-center p-2 hover:bg-slate-100 rounded text-sm">
                  <span className="capitalize">{label}</span>
                  <Badge variant="secondary">{count}</Badge>
                </div>
              ))
          ) : (
            <div className="text-center text-slate-400 mt-10">No components detected</div>
          )}
        </div>
      </Card>
    </div>
  );
}