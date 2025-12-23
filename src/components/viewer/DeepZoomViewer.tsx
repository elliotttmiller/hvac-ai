'use client';

/**
 * DeepZoomViewer Component
 * Google Maps-style viewport with OpenSeadragon for tile-based rendering
 * and Canvas overlay for high-performance annotation rendering
 */

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
  onAnnotationUpdate?: (id: string, updates: Partial<EditableAnnotation>) => void;
  spatialIndex: SpatialAnnotationIndex;
}

// Class color palette
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

export default function DeepZoomViewer({
  imageUrl,
  annotations,
  selectedId,
  hoveredId,
  renderConfig,
  onSelect,
  onHover,
  onAnnotationUpdate,
  spatialIndex,
}: DeepZoomViewerProps) {
  const viewerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const osdRef = useRef<OpenSeadragon.Viewer | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [isReady, setIsReady] = useState(false);

  // Initialize OpenSeadragon viewer
  useEffect(() => {
    if (!viewerRef.current) return;

    const viewer = OpenSeadragon({
      element: viewerRef.current,
      prefixUrl: 'https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/',
      tileSources: {
        type: 'image',
        url: imageUrl,
      },
      showNavigationControl: true,
      showNavigator: true,
      navigatorPosition: 'BOTTOM_RIGHT',
      animationTime: 0.5,
      springStiffness: 10,
      maxZoomPixelRatio: 3,
      minZoomLevel: 0.5,
      visibilityRatio: 0.5,
      constrainDuringPan: false,
      gestureSettingsMouse: {
        scrollToZoom: true,
        clickToZoom: false,
        dblClickToZoom: true,
        pinchToZoom: true,
        flickEnabled: true,
        flickMinSpeed: 100,
        flickMomentum: 0.4,
      },
    });

    osdRef.current = viewer;

    viewer.addHandler('open', () => {
      setIsReady(true);
      console.log('[DeepZoomViewer] OpenSeadragon ready');
    });

    viewer.addHandler('animation', () => {
      requestRender();
    });

    viewer.addHandler('resize', () => {
      if (canvasRef.current && viewerRef.current) {
        const rect = viewerRef.current.getBoundingClientRect();
        canvasRef.current.width = rect.width;
        canvasRef.current.height = rect.height;
      }
      requestRender();
    });

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      viewer.destroy();
      osdRef.current = null;
    };
  }, [imageUrl]);

  // Request animation frame for rendering
  const requestRender = useCallback(() => {
    if (animationFrameRef.current) return;

    animationFrameRef.current = requestAnimationFrame(() => {
      animationFrameRef.current = null;
      renderAnnotations();
    });
  }, []);

  // Get current viewport bounds in image coordinates
  const getViewportBounds = useCallback((): ViewportBounds | null => {
    const viewer = osdRef.current;
    if (!viewer) return null;

    const bounds = viewer.viewport.getBounds();
    const imageBounds = viewer.world.getItemAt(0)?.viewportToImageRectangle(bounds);

    if (!imageBounds) return null;

    return {
      x: imageBounds.x,
      y: imageBounds.y,
      width: imageBounds.width,
      height: imageBounds.height,
    };
  }, []);

  // Transform image coordinates to canvas coordinates
  const imageToCanvas = useCallback((x: number, y: number): [number, number] | null => {
    const viewer = osdRef.current;
    if (!viewer) return null;

    const item = viewer.world.getItemAt(0);
    if (!item) return null;

    const imagePoint = new OpenSeadragon.Point(x, y);
    const viewportPoint = item.imageToViewportCoordinates(imagePoint);
    const canvasPoint = viewer.viewport.viewportToViewerElementCoordinates(viewportPoint);

    return [canvasPoint.x, canvasPoint.y];
  }, []);

  // Render annotations on canvas with viewport culling
  const renderAnnotations = useCallback(() => {
    const canvas = canvasRef.current;
    const viewer = osdRef.current;
    if (!canvas || !viewer || !isReady) return;

    const ctx = canvas.getContext('2d', {
      alpha: true,
      desynchronized: true,
    });
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Get visible annotations using viewport culling
    const viewportBounds = getViewportBounds();
    if (!viewportBounds) return;

    const visibleAnnotations = spatialIndex.search(viewportBounds);

    // Get current zoom level for LOD
    const zoom = viewer.viewport.getZoom();
    const lodLevel = zoom < 1 ? 'overview' : zoom < 2 ? 'medium' : 'detail';

    // Configure rendering based on LOD
    const shouldShowLabels = lodLevel !== 'overview' && renderConfig.showLabels;
    const shouldShowFill = renderConfig.showFill;

    // Render each visible annotation
    for (const indexed of visibleAnnotations) {
      const ann = indexed.annotation;
      const isSelected = ann.id === selectedId;
      const isHovered = ann.id === hoveredId;
      const color = getColorForLabel(ann.label);

      const [x1, y1, x2, y2] = ann.bbox;

      // Transform to canvas coordinates
      const topLeft = imageToCanvas(x1, y1);
      const bottomRight = imageToCanvas(x2, y2);

      if (!topLeft || !bottomRight) continue;

      const [cx1, cy1] = topLeft;
      const [cx2, cy2] = bottomRight;
      const width = cx2 - cx1;
      const height = cy2 - cy1;

      // Skip if too small to render
      if (width < 2 || height < 2) continue;

      // Render polygon if available, otherwise bbox
      if (ann.polygon && lodLevel !== 'overview') {
        ctx.beginPath();
        const firstPoint = imageToCanvas(ann.polygon[0][0], ann.polygon[0][1]);
        if (firstPoint) {
          ctx.moveTo(firstPoint[0], firstPoint[1]);
          for (let i = 1; i < ann.polygon.length; i++) {
            const point = imageToCanvas(ann.polygon[i][0], ann.polygon[i][1]);
            if (point) {
              ctx.lineTo(point[0], point[1]);
            }
          }
          ctx.closePath();

          // Fill
          if (shouldShowFill || isHovered || isSelected) {
            ctx.globalAlpha = isSelected ? 0.4 : isHovered ? 0.35 : renderConfig.opacity;
            ctx.fillStyle = color;
            ctx.fill();
            ctx.globalAlpha = 1.0;
          }

          // Stroke
          ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 2;
          ctx.strokeStyle = color;
          ctx.stroke();
        }
      } else {
        // Render bounding box
        if (shouldShowFill || isHovered || isSelected) {
          ctx.globalAlpha = isSelected ? 0.4 : isHovered ? 0.35 : renderConfig.opacity;
          ctx.fillStyle = color;
          ctx.fillRect(cx1, cy1, width, height);
          ctx.globalAlpha = 1.0;
        }

        ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 2;
        ctx.strokeStyle = color;
        ctx.strokeRect(cx1, cy1, width, height);
      }

      // Render label
      if (shouldShowLabels || isSelected || isHovered) {
        const labelText = `${ann.label} ${Math.round(ann.score * 100)}%`;
        ctx.font = isSelected || isHovered ? 'bold 14px sans-serif' : '12px sans-serif';

        const metrics = ctx.measureText(labelText);
        const padding = 4;
        const labelWidth = metrics.width + padding * 2;
        const labelHeight = 20;

        // Label background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(cx1, cy1 - labelHeight - 2, labelWidth, labelHeight);

        // Label text
        ctx.fillStyle = color;
        ctx.fillText(labelText, cx1 + padding, cy1 - 6);
      }

      // Render resize handles for selected annotation
      if (isSelected && lodLevel === 'detail') {
        const handleSize = 8;
        ctx.fillStyle = color;
        // Corner handles
        ctx.fillRect(cx1 - handleSize / 2, cy1 - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(cx2 - handleSize / 2, cy1 - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(cx1 - handleSize / 2, cy2 - handleSize / 2, handleSize, handleSize);
        ctx.fillRect(cx2 - handleSize / 2, cy2 - handleSize / 2, handleSize, handleSize);
      }
    }

    // Render SAHI grid if enabled
    if (renderConfig.showGrid) {
      renderSAHIGrid(ctx, viewportBounds);
    }
  }, [
    annotations,
    selectedId,
    hoveredId,
    renderConfig,
    spatialIndex,
    getViewportBounds,
    imageToCanvas,
    isReady,
  ]);

  // Render SAHI grid overlay
  const renderSAHIGrid = useCallback(
    (ctx: CanvasRenderingContext2D, bounds: ViewportBounds) => {
      const tileSize = 640; // Standard YOLO/SAHI tile size
      const overlap = 64; // Standard overlap

      const startX = Math.floor(bounds.x / tileSize) * tileSize;
      const startY = Math.floor(bounds.y / tileSize) * tileSize;
      const endX = Math.ceil((bounds.x + bounds.width) / tileSize) * tileSize;
      const endY = Math.ceil((bounds.y + bounds.height) / tileSize) * tileSize;

      ctx.strokeStyle = 'rgba(255, 255, 0, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 5]);

      // Draw vertical lines
      for (let x = startX; x <= endX; x += tileSize) {
        const canvasPoints = [imageToCanvas(x, startY), imageToCanvas(x, endY)];
        if (canvasPoints[0] && canvasPoints[1]) {
          ctx.beginPath();
          ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
          ctx.lineTo(canvasPoints[1][0], canvasPoints[1][1]);
          ctx.stroke();
        }
      }

      // Draw horizontal lines
      for (let y = startY; y <= endY; y += tileSize) {
        const canvasPoints = [imageToCanvas(startX, y), imageToCanvas(endX, y)];
        if (canvasPoints[0] && canvasPoints[1]) {
          ctx.beginPath();
          ctx.moveTo(canvasPoints[0][0], canvasPoints[0][1]);
          ctx.lineTo(canvasPoints[1][0], canvasPoints[1][1]);
          ctx.stroke();
        }
      }

      ctx.setLineDash([]);
    },
    [imageToCanvas]
  );

  // Handle mouse click for selection
  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const viewer = osdRef.current;
      if (!viewer || !onSelect) return;

      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      // Convert canvas coordinates to viewport coordinates
      const viewportPoint = viewer.viewport.viewerElementToViewportCoordinates(
        new OpenSeadragon.Point(canvasX, canvasY)
      );

      // Convert viewport to image coordinates
      const item = viewer.world.getItemAt(0);
      if (!item) return;

      const imagePoint = item.viewportToImageCoordinates(viewportPoint);

      // Find annotation at this point
      const hit = spatialIndex.findAtPoint(imagePoint.x, imagePoint.y);

      if (hit) {
        onSelect(hit.id);
      } else {
        onSelect(null);
      }
    },
    [onSelect, spatialIndex]
  );

  // Handle mouse move for hover
  const handleCanvasMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const viewer = osdRef.current;
      if (!viewer || !onHover) return;

      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;

      const canvasX = e.clientX - rect.left;
      const canvasY = e.clientY - rect.top;

      const viewportPoint = viewer.viewport.viewerElementToViewportCoordinates(
        new OpenSeadragon.Point(canvasX, canvasY)
      );

      const item = viewer.world.getItemAt(0);
      if (!item) return;

      const imagePoint = item.viewportToImageCoordinates(viewportPoint);

      const hit = spatialIndex.findAtPoint(imagePoint.x, imagePoint.y);

      if (hit) {
        onHover(hit.id);
      } else {
        onHover(null);
      }
    },
    [onHover, spatialIndex]
  );

  // Trigger re-render when dependencies change
  useEffect(() => {
    if (isReady) {
      requestRender();
    }
  }, [annotations, selectedId, hoveredId, renderConfig, isReady, requestRender]);

  return (
    <div className="relative w-full h-full bg-slate-900">
      <div ref={viewerRef} className="absolute inset-0" />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-auto"
        onClick={handleCanvasClick}
        onMouseMove={handleCanvasMove}
        style={{ cursor: hoveredId ? 'pointer' : 'default' }}
      />
      {!isReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-900 text-white">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mx-auto mb-4" />
            <p>Loading viewport...</p>
          </div>
        </div>
      )}
    </div>
  );
}
