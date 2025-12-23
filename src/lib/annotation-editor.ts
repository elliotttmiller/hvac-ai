/**
 * Annotation Editor Utilities
 * Helper functions for HITL (Human-in-the-Loop) annotation editing
 */

import type { EditableAnnotation } from '@/types/deep-zoom';

export type ResizeHandle =
  | 'top-left'
  | 'top-right'
  | 'bottom-left'
  | 'bottom-right'
  | 'top'
  | 'bottom'
  | 'left'
  | 'right'
  | null;

export interface ResizeState {
  handle: ResizeHandle;
  startX: number;
  startY: number;
  startBbox: [number, number, number, number];
}

/**
 * Get resize handle at a point
 */
export function getResizeHandleAtPoint(
  annotation: EditableAnnotation,
  pointX: number,
  pointY: number,
  scale: number = 1
): ResizeHandle {
  const [x1, y1, x2, y2] = annotation.bbox;
  const handleSize = 8 / scale; // Adjust for zoom level

  // Corner handles
  if (isPointNear(pointX, pointY, x1, y1, handleSize)) return 'top-left';
  if (isPointNear(pointX, pointY, x2, y1, handleSize)) return 'top-right';
  if (isPointNear(pointX, pointY, x1, y2, handleSize)) return 'bottom-left';
  if (isPointNear(pointX, pointY, x2, y2, handleSize)) return 'bottom-right';

  // Edge handles
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;

  if (isPointNear(pointX, pointY, centerX, y1, handleSize)) return 'top';
  if (isPointNear(pointX, pointY, centerX, y2, handleSize)) return 'bottom';
  if (isPointNear(pointX, pointY, x1, centerY, handleSize)) return 'left';
  if (isPointNear(pointX, pointY, x2, centerY, handleSize)) return 'right';

  return null;
}

/**
 * Check if a point is near another point
 */
function isPointNear(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  threshold: number
): boolean {
  const dx = x1 - x2;
  const dy = y1 - y2;
  return Math.sqrt(dx * dx + dy * dy) <= threshold;
}

/**
 * Compute new bounding box during resize
 */
export function computeResizedBbox(
  handle: ResizeHandle,
  startBbox: [number, number, number, number],
  deltaX: number,
  deltaY: number,
  minSize: number = 10
): [number, number, number, number] {
  let [x1, y1, x2, y2] = startBbox;

  switch (handle) {
    case 'top-left':
      x1 += deltaX;
      y1 += deltaY;
      break;
    case 'top-right':
      x2 += deltaX;
      y1 += deltaY;
      break;
    case 'bottom-left':
      x1 += deltaX;
      y2 += deltaY;
      break;
    case 'bottom-right':
      x2 += deltaX;
      y2 += deltaY;
      break;
    case 'top':
      y1 += deltaY;
      break;
    case 'bottom':
      y2 += deltaY;
      break;
    case 'left':
      x1 += deltaX;
      break;
    case 'right':
      x2 += deltaX;
      break;
  }

  // Ensure minimum size and correct order
  if (x2 - x1 < minSize) {
    if (handle?.includes('left')) x1 = x2 - minSize;
    else x2 = x1 + minSize;
  }

  if (y2 - y1 < minSize) {
    if (handle?.includes('top')) y1 = y2 - minSize;
    else y2 = y1 + minSize;
  }

  // Ensure x1 < x2 and y1 < y2
  const minX = Math.min(x1, x2);
  const maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2);
  const maxY = Math.max(y1, y2);

  return [minX, minY, maxX, maxY];
}

/**
 * Get cursor style for resize handle
 */
export function getCursorForHandle(handle: ResizeHandle): string {
  switch (handle) {
    case 'top-left':
    case 'bottom-right':
      return 'nwse-resize';
    case 'top-right':
    case 'bottom-left':
      return 'nesw-resize';
    case 'top':
    case 'bottom':
      return 'ns-resize';
    case 'left':
    case 'right':
      return 'ew-resize';
    default:
      return 'default';
  }
}

/**
 * Validate annotation bbox
 */
export function validateBbox(
  bbox: [number, number, number, number],
  imageWidth: number,
  imageHeight: number
): [number, number, number, number] {
  let [x1, y1, x2, y2] = bbox;

  // Clamp to image bounds
  x1 = Math.max(0, Math.min(x1, imageWidth));
  y1 = Math.max(0, Math.min(y1, imageHeight));
  x2 = Math.max(0, Math.min(x2, imageWidth));
  y2 = Math.max(0, Math.min(y2, imageHeight));

  // Ensure proper order
  const minX = Math.min(x1, x2);
  const maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2);
  const maxY = Math.max(y1, y2);

  return [minX, minY, maxX, maxY];
}

/**
 * Create a new annotation from a drawn rectangle
 */
export function createAnnotationFromRect(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  defaultLabel: string = 'Unknown',
  defaultScore: number = 1.0
): Omit<EditableAnnotation, 'id' | 'isDirty' | 'isNew'> {
  // Ensure proper order
  const minX = Math.min(x1, x2);
  const maxX = Math.max(x1, x2);
  const minY = Math.min(y1, y2);
  const maxY = Math.max(y1, y2);

  return {
    label: defaultLabel,
    score: defaultScore,
    bbox: [minX, minY, maxX, maxY],
  };
}

/**
 * Check if two bboxes overlap
 */
export function bboxesOverlap(
  bbox1: [number, number, number, number],
  bbox2: [number, number, number, number]
): boolean {
  const [x1a, y1a, x2a, y2a] = bbox1;
  const [x1b, y1b, x2b, y2b] = bbox2;

  return !(x2a < x1b || x2b < x1a || y2a < y1b || y2b < y1a);
}

/**
 * Compute IoU (Intersection over Union) between two bboxes
 */
export function computeIoU(
  bbox1: [number, number, number, number],
  bbox2: [number, number, number, number]
): number {
  const [x1a, y1a, x2a, y2a] = bbox1;
  const [x1b, y1b, x2b, y2b] = bbox2;

  // Compute intersection
  const xLeft = Math.max(x1a, x1b);
  const yTop = Math.max(y1a, y1b);
  const xRight = Math.min(x2a, x2b);
  const yBottom = Math.min(y2a, y2b);

  if (xRight < xLeft || yBottom < yTop) return 0;

  const intersectionArea = (xRight - xLeft) * (yBottom - yTop);

  // Compute union
  const area1 = (x2a - x1a) * (y2a - y1a);
  const area2 = (x2b - x1b) * (y2b - y1b);
  const unionArea = area1 + area2 - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}
