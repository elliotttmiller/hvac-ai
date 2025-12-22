/**
 * Spatial Index Utility
 * R-Tree based spatial indexing for efficient viewport culling and hit-testing
 */

import RBush from 'rbush';
import type { IndexedAnnotation, ViewportBounds, EditableAnnotation } from '@/types/deep-zoom';

export class SpatialAnnotationIndex {
  private tree: RBush<IndexedAnnotation>;

  constructor() {
    this.tree = new RBush<IndexedAnnotation>();
  }

  /**
   * Insert an annotation into the spatial index
   */
  insert(annotation: EditableAnnotation): void {
    const indexed = this.annotationToIndexed(annotation);
    this.tree.insert(indexed);
  }

  /**
   * Remove an annotation from the spatial index
   */
  remove(annotation: EditableAnnotation): void {
    const indexed = this.annotationToIndexed(annotation);
    this.tree.remove(indexed, (a: IndexedAnnotation, b: IndexedAnnotation) => a.id === b.id);
  }

  /**
   * Update an annotation in the spatial index
   */
  update(oldAnnotation: EditableAnnotation, newAnnotation: EditableAnnotation): void {
    this.remove(oldAnnotation);
    this.insert(newAnnotation);
  }

  /**
   * Search for annotations within viewport bounds
   * Returns only annotations that intersect with the viewport
   */
  search(bounds: ViewportBounds): IndexedAnnotation[] {
    const searchBounds = {
      minX: bounds.x,
      minY: bounds.y,
      maxX: bounds.x + bounds.width,
      maxY: bounds.y + bounds.height,
    };
    return this.tree.search(searchBounds);
  }

  /**
   * Find annotation at a specific point (for mouse hit-testing)
   */
  findAtPoint(x: number, y: number): IndexedAnnotation | null {
    const results = this.tree.search({
      minX: x,
      minY: y,
      maxX: x,
      maxY: y,
    });

    // Return the topmost (last in array) annotation at this point
    return results.length > 0 ? results[results.length - 1] : null;
  }

  /**
   * Clear all annotations from the index
   */
  clear(): void {
    this.tree.clear();
  }

  /**
   * Get total count of indexed annotations
   */
  size(): number {
    return this.tree.all().length;
  }

  /**
   * Bulk load annotations for better performance
   */
  bulkLoad(annotations: EditableAnnotation[]): void {
    const indexed = annotations.map((ann) => this.annotationToIndexed(ann));
    this.tree.load(indexed);
  }

  /**
   * Convert annotation to indexed format for R-Tree
   */
  private annotationToIndexed(annotation: EditableAnnotation): IndexedAnnotation {
    const [x1, y1, x2, y2] = annotation.bbox;
    return {
      minX: Math.min(x1, x2),
      minY: Math.min(y1, y2),
      maxX: Math.max(x1, x2),
      maxY: Math.max(y1, y2),
      id: annotation.id,
      annotation,
    };
  }
}

/**
 * Test if a point is inside a polygon using ray casting algorithm
 */
export function isPointInPolygon(point: [number, number], polygon: number[][]): boolean {
  const [x, y] = point;
  let inside = false;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];

    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;

    if (intersect) inside = !inside;
  }

  return inside;
}

/**
 * Calculate bounding box from polygon
 */
export function getBoundsFromPolygon(polygon: number[][]): [number, number, number, number] {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const [x, y] of polygon) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  return [minX, minY, maxX, maxY];
}
