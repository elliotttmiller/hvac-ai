'use client';

import React, { useEffect, useRef, useCallback, useState } from 'react';
import OpenSeadragon from 'openseadragon';
import type { EditableAnnotation, ViewportBounds, RenderConfig } from '@/types/deep-zoom';
import { SpatialAnnotationIndex } from '@/lib/spatial-index';

interface DeepZoomViewerProps {
  imageUrl: string;
  annotations: EditableAnnotation[];
  selectedId: string | null;
  hoveredId: string | null;
  renderConfig: RenderConfig;
  onSelect?: (id: string | null) => void;
  onHover?: (id: string | null) => void;
  // Optional callback when an annotation is edited in the viewer
  onAnnotationUpdate?: (id: string, updates: Partial<EditableAnnotation>) => void;
  spatialIndex: SpatialAnnotationIndex;
}

const CLASS_COLORS: Record<string, string> = {
  valve: '#ef4444',
  instrument: '#3b82f6',
  sensor: '#10b981',
  duct: '#f59e0b',
  vent: '#8b5cf6',
  default: '#6b7280',
};

function getColorForLabel(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes('valve')) return CLASS_COLORS.valve;
  if (lower.includes('instrument') || lower.includes('computer')) return CLASS_COLORS.instrument;
  if (lower.includes('sensor')) return CLASS_COLORS.sensor;
  if (lower.includes('duct')) return CLASS_COLORS.duct;
  if (lower.includes('vent')) return CLASS_COLORS.vent;
  return CLASS_COLORS.default;
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  label: string,
  score: number,
  x: number,
  y: number,
  color: string,
  isHighlighted: boolean,
  textContent?: string,
  textConfidence?: number
) {
  // Prefer extracted text content over class label if available
  const displayText = textContent 
    ? `${textContent} (${Math.round((textConfidence || 0) * 100)}%)`
    : `${label} ${Math.round(score * 100)}%`;
  
  // Use monospace font for extracted text to signify "Read Data"
  const fontFamily = textContent ? 'monospace' : 'sans-serif';
  ctx.font = isHighlighted 
    ? `bold 14px ${fontFamily}` 
    : `12px ${fontFamily}`;
  
  const metrics = ctx.measureText(displayText);
  const padding = 4;
  const labelWidth = metrics.width + padding * 2;
  const labelHeight = 20;

  // High-contrast background for extracted text
  const bgColor = textContent ? 'rgba(0, 255, 0, 0.9)' : 'rgba(0, 0, 0, 0.8)';
  ctx.fillStyle = bgColor;
  ctx.fillRect(x, y - labelHeight - 2, labelWidth, labelHeight);
  
  // Text color - black for extracted text for better contrast on green background
  ctx.fillStyle = textContent ? '#000000' : color;
  ctx.fillText(displayText, x + padding, y - 6);
}

export default function DeepZoomViewer({
  imageUrl,
  annotations,
  selectedId,
  hoveredId,
  renderConfig,
  onSelect,
  onHover,
  spatialIndex,
}: DeepZoomViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const osdRef = useRef<OpenSeadragon.Viewer | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isReady, setIsReady] = useState(false);
  const renderAnnotationsRef = useRef<() => void>(() => {});

  const requestRender = useCallback(() => {
    if (animationFrameRef.current) return;
    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      renderAnnotationsRef.current();
    });
  }, []);

  useEffect(() => {
    if (!viewerRef.current) return;

    const viewer = OpenSeadragon({
      element: viewerRef.current,
      prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
      tileSources: { type: 'image', url: imageUrl },
      showNavigationControl: true,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      animationTime: 0.5,
      springStiffness: 10,
      minZoomLevel: 0.5,
      gestureSettingsMouse: {
        scrollToZoom: true,
        clickToZoom: false,
        dblClickToZoom: true,
        pinchToZoom: true,
      },
    });

    osdRef.current = viewer;

    viewer.addHandler('open', () => setIsReady(true));
    viewer.addHandler('animation', () => requestRender());
    viewer.addHandler('resize', () => {
      if (canvasRef.current && viewerRef.current) {
        const rect = viewerRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width;
        canvasRef.current.height = rect.height;
      }
      requestRender();
    });

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      viewer.destroy();
      osdRef.current = null;
    };
  }, [imageUrl, requestRender]);

  const getViewportBounds = useCallback((): ViewportBounds | null => {
    const viewer = osdRef.current;
    if (!viewer) return null;
    const bounds = viewer.viewport.getBounds();
    const imageBounds = viewer.world.getItemAt(0)?.viewportToImageRectangle(bounds);
    return imageBounds ? { x: imageBounds.x, y: imageBounds.y, width: imageBounds.width, height: imageBounds.height } : null;
  }, []);

  const imageToCanvas = useCallback((x: number, y: number): [number, number] | null => {
    const viewer = osdRef.current;
    if (!viewer) return null;
    const item = viewer.world.getItemAt(0);
    if (!item) return null;
    const p = viewer.viewport.viewportToViewerElementCoordinates(item.imageToViewportCoordinates(new OpenSeadragon.Point(x, y)));
    return [p.x, p.y];
  }, []);

  const renderAnnotations = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !isReady) return;
    const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const bounds = getViewportBounds();
    if (!bounds) return;

    const visibleAnnotations = spatialIndex.search(bounds);
    const shouldShowLabels = renderConfig.showLabels;
    const shouldShowFill = renderConfig.showFill;

    for (const indexed of visibleAnnotations) {
      const ann = indexed.annotation;
      const isSelected = ann.id === selectedId;
      const isHovered = ann.id === hoveredId;
      const color = getColorForLabel(ann.label);

      ctx.beginPath();
      let firstPoint: [number, number] | null = null;

      // 1. OBB Corner Rendering (Preferred for YOLOv11-OBB)
      // The 'points' array contains the 4 calculated corners of the rotated box.
      // Using these directly is faster and more accurate than re-calculating rotation math.
      if (ann.points && ann.points.length === 4) {
        firstPoint = imageToCanvas(ann.points[0][0], ann.points[0][1]);
        if (firstPoint) ctx.moveTo(firstPoint[0], firstPoint[1]);
        for (let i = 1; i < ann.points.length; i++) {
          const pt = imageToCanvas(ann.points[i][0], ann.points[i][1]);
          if (pt) ctx.lineTo(pt[0], pt[1]);
        }
        ctx.closePath();
      } 
      // 2. OBB Parameter Calculation (Fallback)
      // If we only have center/width/height/rotation, we calculate the corners manually.
      else if (ann.obb && typeof ann.obb.x_center === 'number') {
        const { x_center, y_center, width, height, rotation } = ann.obb;
        // Calculate 4 corners based on rotation
        const corners = [
          [-width/2, -height/2], [width/2, -height/2],
          [width/2, height/2], [-width/2, height/2]
        ].map(([dx, dy]) => {
          const x = x_center + dx * Math.cos(rotation) - dy * Math.sin(rotation);
          const y = y_center + dx * Math.sin(rotation) + dy * Math.cos(rotation);
          return [x, y];
        });
        
        firstPoint = imageToCanvas(corners[0][0], corners[0][1]);
        if (firstPoint) ctx.moveTo(firstPoint[0], firstPoint[1]);
        for (let i = 1; i < corners.length; i++) {
          const pt = imageToCanvas(corners[i][0], corners[i][1]);
          if (pt) ctx.lineTo(pt[0], pt[1]);
        }
        ctx.closePath();
      }
      // 3. Standard BBox (Fallback)
      else {
        const [x1, y1, x2, y2] = ann.bbox;
        const p1 = imageToCanvas(x1, y1);
        const p2 = imageToCanvas(x2, y1);
        const p3 = imageToCanvas(x2, y2);
        const p4 = imageToCanvas(x1, y2);
        if (p1 && p2 && p3 && p4) {
          firstPoint = p1;
          ctx.moveTo(p1[0], p1[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.lineTo(p3[0], p3[1]);
          ctx.lineTo(p4[0], p4[1]);
          ctx.closePath();
        }
      }

      if (shouldShowFill || isHovered || isSelected) {
        ctx.globalAlpha = isSelected ? 0.4 : isHovered ? 0.35 : renderConfig.opacity;
        ctx.fillStyle = color;
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }

  ctx.lineWidth = isSelected ? 1.6 : isHovered ? 1.2 : 0.9;
      ctx.strokeStyle = color;
      ctx.stroke();

      if (firstPoint && (shouldShowLabels || isSelected || isHovered)) {
        drawLabel(
          ctx, 
          ann.label, 
          ann.score, 
          firstPoint[0], 
          firstPoint[1], 
          color, 
          isSelected || isHovered,
          ann.textContent,
          ann.textConfidence
        );
      }
    }
  }, [getViewportBounds, imageToCanvas, isReady, renderConfig, spatialIndex, selectedId, hoveredId]);

  useEffect(() => {
    renderAnnotationsRef.current = renderAnnotations;
  }, [renderAnnotations]);

  const handleInteraction = useCallback((e: React.MouseEvent<HTMLCanvasElement>, type: 'click' | 'move') => {
    const viewer = osdRef.current;
    if (!viewer || (!onSelect && !onHover)) return;

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const viewportPoint = viewer.viewport.viewerElementToViewportCoordinates(
      new OpenSeadragon.Point(e.clientX - rect.left, e.clientY - rect.top)
    );
    const item = viewer.world.getItemAt(0);
    if (!item) return;
    const imagePoint = item.viewportToImageCoordinates(viewportPoint);

    const hit = spatialIndex.findAtPoint(imagePoint.x, imagePoint.y);

    if (type === 'click' && onSelect) onSelect(hit ? hit.id : null);
    if (type === 'move' && onHover) onHover(hit ? hit.id : null);
  }, [onSelect, onHover, spatialIndex]);

  return (
    <div className="relative w-full h-full bg-slate-900">
      <div ref={viewerRef} className="absolute inset-0" />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-auto"
        onClick={(e) => handleInteraction(e, 'click')}
        onMouseMove={(e) => handleInteraction(e, 'move')}
        style={{ cursor: hoveredId ? 'pointer' : 'default' }}
      />
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900 text-white">
          <p>Loading DeepZoom Viewport...</p>
        </div>
      )}
    </div>
  );
}