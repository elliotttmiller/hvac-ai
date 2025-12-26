'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { Segment, CountResult } from '@/types/analysis';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
// label display modes are controlled via dropdown; no UI Switch required
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

// Cache assigned colors per label so the same label always receives the same color
const labelColorCache = new Map<string, string>();

function labelToColor(label: string) {
  if (!label) return CLASS_COLORS.default;
  const key = String(label);
  const cached = labelColorCache.get(key);
  if (cached) return cached;

  // djb2 hash
  let hash = 5381;
  for (let i = 0; i < key.length; i++) {
    hash = ((hash << 5) + hash) + key.charCodeAt(i); /* hash * 33 + c */
    hash = hash | 0;
  }
  const hue = Math.abs(hash) % 360;

  // Use pleasant saturation/lightness for good contrast on dark backgrounds
  const saturation = 65;
  const lightness = 55;
  const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  labelColorCache.set(key, color);
  return color;
}

function getColorForLabel(label: string) {
  // Always generate a unique color per full label string. This ensures
  // subclasses (e.g. "Valve Gate", "Valve Globe") get distinct colors.
  try {
    return labelToColor(label || '');
  } catch (err) {
    return CLASS_COLORS.default;
  }
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
  
  const [showFill, setShowFill] = useState(true);
  const [labelDisplay, setLabelDisplay] = useState<'boxes'|'boxes-names'|'boxes-names-score'|'none'>('boxes-names-score');
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [highlightedCategory, setHighlightedCategory] = useState<string | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const [zoom, setZoom] = useState(1);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  // Separate the image offset (initial centering) from the user pan.
  const imageOffsetRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
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

  // Ensure path cache is cleared whenever segments change so we rebuild paths for
  // new/updated segment geometry.
  useEffect(() => {
    pathCacheRef.current.clear();
    // Request a repaint
    requestAnimationFrame(() => drawOverlayRef.current());
  }, [segments]);

  useEffect(() => {
    if (typeof window === 'undefined' || !uploadedImage) return;

    const img = new Image();
    // don't set crossOrigin for blob/object URLs (can cause failures)
    const url = URL.createObjectURL(uploadedImage);
    img.src = url;
    imageRef.current = img;

    // copy pathCacheRef to a local variable for stable cleanup reference
    const pathCache = pathCacheRef.current;

    let resizeObserver: ResizeObserver | null = null;

    const updateCanvasSize = () => {
      const bg = bgCanvasRef.current;
      const overlay = overlayCanvasRef.current;
      const parent = containerRef.current;
      if (!bg || !overlay || !parent || !img.naturalWidth) return;

      // Use container available width/height to compute display size so
      // the image fits responsively inside the viewport.
      const availableW = Math.max(200, parent.clientWidth - 0);
      const availableH = Math.max(120, parent.clientHeight - 0);

      // Fit by width first, then constrain by height while preserving aspect
      const aspect = img.naturalHeight / img.naturalWidth;
      let displayW = availableW;
      let displayH = Math.round(displayW * aspect);
      if (displayH > availableH) {
        // too tall for container — fit by height instead
        displayH = availableH;
        displayW = Math.round(displayH / aspect);
      }
      // Prevent upsampling beyond image natural dimensions
      displayW = Math.min(displayW, img.naturalWidth);
      displayH = Math.min(displayH, img.naturalHeight);

      scaleRef.current = displayW / img.naturalWidth;

      resizeCanvasToImage(bg, displayW, displayH);
      resizeCanvasToImage(overlay, displayW, displayH);
      createOffscreenBackground(displayW, displayH);
  // Clear path cache when display size (scaleRef) changes so paths are
  // regenerated with correct coordinates.
  pathCacheRef.current.clear();

  // center the image inside the container (store as offset, leave pan at 0)
  const centerX = Math.round((parent.clientWidth - displayW) / 2);
  const centerY = Math.round((parent.clientHeight - displayH) / 2);
  imageOffsetRef.current = { x: centerX, y: centerY };
  // reset current pan so image is centered
  setPanX(0);
  setPanY(0);

      requestAnimationFrame(() => {
        drawBackgroundRef.current();
        drawOverlayRef.current();
      });
    };

    const handleLoad = () => {
      updateCanvasSize();

      // watch for container resizes to keep the image responsive
        if (containerRef.current && 'ResizeObserver' in window) {
          const observed = containerRef.current;
          resizeObserver = new ResizeObserver(() => updateCanvasSize());
          resizeObserver.observe(observed);
        }
    };

    const handleError = (ev: Event | string) => {
      console.error('Image load error', ev);
    };

    img.addEventListener('load', handleLoad);
    img.addEventListener('error', handleError as EventListener);

    // Also update on window resize as a fallback
    const onWin = () => updateCanvasSize();
    window.addEventListener('resize', onWin);

    return () => {
      URL.revokeObjectURL(url);
      img.removeEventListener('load', handleLoad);
      img.removeEventListener('error', handleError as EventListener);
      window.removeEventListener('resize', onWin);
      // Use the observed element reference captured earlier when possible
      if (resizeObserver) {
        try { resizeObserver.disconnect(); } catch {}
      }
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
  // Translate by image offset plus any user pan, then apply zoom
  const offsetX = imageOffsetRef.current.x + panX;
  const offsetY = imageOffsetRef.current.y + panY;
  ctx.translate(offsetX, offsetY);
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
  const offsetX2 = imageOffsetRef.current.x + panX;
  const offsetY2 = imageOffsetRef.current.y + panY;
  ctx.translate(offsetX2, offsetY2);
  ctx.scale(zoom, zoom);

    segments.forEach((seg, index) => {
      const isHovered = index === hoveredIndex;
      const color = getColorForLabel(seg.label);
      const matchesFilter = !filterCategory || seg.label === filterCategory;
      const matchesHighlight = !highlightedCategory || seg.label === highlightedCategory;
      const s = scaleRef.current;
      // Use a stable string key for the path cache (seg.id may be undefined or numeric)
      const idKey = String(seg.id ?? index);
      // include display scale in the key so path is rebuilt when canvas scale changes
      const cacheKey = `${idKey}@${s}`;
      let path = pathCacheRef.current.get(cacheKey);
      
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
        
        pathCacheRef.current.set(cacheKey, path);
      }

      // Label display modes: 'none' | 'boxes' | 'boxes-names' | 'boxes-names-score'
      const shouldDrawBoxes = labelDisplay !== 'none';
      const shouldDrawNames = labelDisplay === 'boxes-names' || labelDisplay === 'boxes-names-score';
      const shouldDrawScore = labelDisplay === 'boxes-names-score';
      // If a filter is active, dim non-matching items by lowering alpha and
      // skipping labels; matching items are emphasized.
      const isMatch = matchesFilter && matchesHighlight;
      if (!matchesFilter) {
        // draw faint outline for non-matches (so user can still see context)
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(200,200,200,0.08)';
        ctx.stroke(path);
      }

      if (isMatch || !filterCategory) {
        // emphasis for hovered or highlighted
        ctx.lineWidth = isHovered || (highlightedCategory && seg.label === highlightedCategory) ? 3 : 2;
        ctx.strokeStyle = color;
        ctx.stroke(path);

        if (showFill || isHovered) {
          ctx.fillStyle = color;
          ctx.globalAlpha = isHovered ? 0.35 : 0.15;
          ctx.fill(path);
          ctx.globalAlpha = 1.0;
        }
      }

  if ((shouldDrawNames || isHovered) && (seg.label || seg.score !== undefined) && (isMatch || !filterCategory)) {
        let lx = 0, ly = 0;
        if (seg.bbox) {
            lx = seg.bbox[0] * s;
            ly = seg.bbox[1] * s;
        } else if (seg.obb) {
            lx = (seg.obb.x_center - seg.obb.width/2) * s;
            ly = (seg.obb.y_center - seg.obb.height/2) * s;
        }

        ctx.font = isHovered ? 'bold 14px sans-serif' : '12px sans-serif';
        const text = shouldDrawScore ? `${seg.label} ${(seg.score * 100).toFixed(0)}%` : `${seg.label}`;
        const metrics = ctx.measureText(text);

        ctx.fillStyle = 'rgba(0,0,0,0.8)';
        ctx.fillRect(lx, ly - 20, metrics.width + 8, 20);
        ctx.fillStyle = color;
        ctx.fillText(text, lx + 4, ly - 6);
      }
    });

    ctx.restore();
  }, [segments, hoveredIndex, showFill, labelDisplay, zoom, panX, panY, filterCategory, highlightedCategory]);
  
  // redraw overlay when filter or highlight changes
  useEffect(() => {
    drawOverlayRef.current = drawOverlayRef.current; // touch ref to ensure effect deps
  }, [filterCategory, highlightedCategory]);

  // Focus (center & zoom) to the union bbox of all segments in a category
  const focusOnCategory = useCallback((category: string) => {
    if (!segments || segments.length === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    segments.forEach(s => {
      if (s.label !== category) return;
      if (s.bbox) {
        minX = Math.min(minX, s.bbox[0]);
        minY = Math.min(minY, s.bbox[1]);
        maxX = Math.max(maxX, s.bbox[2]);
        maxY = Math.max(maxY, s.bbox[3]);
      } else if (s.obb) {
        const x1 = s.obb.x_center - s.obb.width/2;
        const y1 = s.obb.y_center - s.obb.height/2;
        minX = Math.min(minX, x1);
        minY = Math.min(minY, y1);
        maxX = Math.max(maxX, x1 + s.obb.width);
        maxY = Math.max(maxY, y1 + s.obb.height);
      } else if (s.points && s.points.length) {
        s.points.forEach(p => {
          minX = Math.min(minX, p[0]);
          minY = Math.min(minY, p[1]);
          maxX = Math.max(maxX, p[0]);
          maxY = Math.max(maxY, p[1]);
        });
      }
    });
    if (!isFinite(minX)) return;
    const bboxW = Math.max(1, maxX - minX);
    const bboxH = Math.max(1, maxY - minY);

    const parent = containerRef.current;
    const presentScale = scaleRef.current;
    if (!parent) return;

    const targetW = parent.clientWidth * 0.8;
    const targetH = parent.clientHeight * 0.8;
    const zX = targetW / (bboxW * presentScale);
    const zY = targetH / (bboxH * presentScale);
    const newZoom = Math.min(Math.max(0.5, Math.min(zX, zY)), 6);

    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const centerCanvasX = parent.clientWidth / 2;
    const centerCanvasY = parent.clientHeight / 2;

    const newPanX = centerCanvasX - imageOffsetRef.current.x - cx * presentScale * newZoom;
    const newPanY = centerCanvasY - imageOffsetRef.current.y - cy * presentScale * newZoom;

    setZoom(newZoom);
    setPanX(newPanX);
    setPanY(newPanY);
  }, [segments]);

  // include filter/highlight in overlay draw deps
  // (added filterCategory / highlightedCategory to dependency list)

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
    // account for image offset when starting a drag
    dragStartRef.current = { x: e.clientX - (imageOffsetRef.current.x + panX), y: e.clientY - (imageOffsetRef.current.y + panY) };
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging && dragStartRef.current) {
      setPanX(e.clientX - dragStartRef.current.x - imageOffsetRef.current.x);
      setPanY(e.clientY - dragStartRef.current.y - imageOffsetRef.current.y);
      return;
    }

    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    // Compute mouse position in css pixels relative to canvas
    const cssX = e.clientX - rect.left;
    const cssY = e.clientY - rect.top;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Apply the same transforms used during drawing (DPR + imageOffset + pan + zoom)
    ctx.save();
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const offsetX = imageOffsetRef.current.x + panX;
    const offsetY = imageOffsetRef.current.y + panY;
    ctx.translate(offsetX, offsetY);
    ctx.scale(zoom, zoom);

    let found = null;
    // iterate from topmost to bottommost
    for (let i = segments.length - 1; i >= 0; i--) {
      const seg = segments[i];
      const s = scaleRef.current;
      const cacheKey = `${String(seg.id ?? i)}@${s}`;
      const path = pathCacheRef.current.get(cacheKey);
      if (path && ctx.isPointInPath(path, cssX, cssY)) {
        found = i;
        break;
      }
    }

    ctx.restore();
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
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-full">
      <Card className="lg:col-span-4 overflow-hidden flex flex-col h-[760px]">
        <CardHeader className="py-3 px-4 border-b flex flex-row items-center justify-between bg-slate-50">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Scan className="h-4 w-4" /> Visual Inspection
          </CardTitle>
          <div className="flex items-center gap-2">
             <div className="flex items-center gap-2 mr-4">
                <Label htmlFor="labelDisplay" className="text-xs mr-2">Labels</Label>
                <select
                  id="labelDisplay"
                  value={labelDisplay}
                  onChange={e => setLabelDisplay(e.target.value as 'boxes'|'boxes-names'|'boxes-names-score'|'none')}
                  className="text-xs bg-white/90 rounded px-2 py-1 border"
                >
                  <option value="boxes">Boxes only</option>
                  <option value="boxes-names">Boxes + Names</option>
                  <option value="boxes-names-score">Boxes + Names + Confidence</option>
                  <option value="none">None</option>
                </select>
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
                  .map(([label, count]) => {
                    const color = getColorForLabel(label as string);
                    const displayLabel = (label as string).replace(/_/g, ' ');
                    const active = filterCategory === label;
                    return (
                      <div
                        key={label}
                        onClick={() => { if (active) { setFilterCategory(null); } else { setFilterCategory(label); focusOnCategory(label); } }}
                        onMouseEnter={() => setHighlightedCategory(label)}
                        onMouseLeave={() => setHighlightedCategory(null)}
                        role="button"
                        aria-pressed={active}
                        className={`flex justify-between items-center p-2 rounded text-sm cursor-pointer ${active ? 'bg-slate-100' : 'hover:bg-slate-50'}`}
                      >
                        <div className="flex items-center gap-3">
                          <div style={{ width: 10, height: 10, backgroundColor: color, borderRadius: 2 }} />
                          <span className="capitalize">{displayLabel}</span>
                        </div>
                        <Badge variant="secondary" style={{ borderColor: color, color }}>{count}</Badge>
                      </div>
                    );
                  })
          ) : (
            <div className="text-center text-slate-400 mt-10">No components detected</div>
          )}
        </div>
      </Card>
    </div>
  );
}