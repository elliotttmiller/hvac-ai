/**
 * Deep Zoom Image Types
 * Type definitions for tile-based deep zoom rendering system
 */

export interface TileInfo {
  url: string;
  level: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DZIConfig {
  imageUrl: string;
  tileSize: number;
  overlap: number;
  format: 'jpg' | 'png';
  maxLevel: number;
  width: number;
  height: number;
}

export interface ViewportBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface AnnotationEdit {
  id: string;
  type: 'add' | 'modify' | 'delete' | 'reclassify';
  data: Partial<EditableAnnotation>;
  timestamp: number;
}

export interface EditableAnnotation {
  id: string;
  label: string;
  score: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  polygon?: number[][];
  isDirty: boolean;
  isNew: boolean;
}

export interface AnnotationState {
  annotations: Map<string, EditableAnnotation>;
  selectedId: string | null;
  hoveredId: string | null;
  dirtyIds: Set<string>;
  deletedIds: Set<string>;
  confidenceThreshold: number;
}

export interface SpatialIndex {
  insert: (item: IndexedAnnotation) => void;
  remove: (item: IndexedAnnotation) => void;
  search: (bounds: ViewportBounds) => IndexedAnnotation[];
  clear: () => void;
}

export interface IndexedAnnotation {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  id: string;
  annotation: EditableAnnotation;
}

export interface DeltaSavePayload {
  added: EditableAnnotation[];
  modified: EditableAnnotation[];
  deleted: string[];
  verification_status?: 'pending' | 'verified' | 'rejected';
}

export interface InferenceProgress {
  status: 'processing' | 'complete' | 'error';
  progress: number; // 0-100
  currentTile?: { x: number; y: number };
  detections?: EditableAnnotation[];
  message?: string;
}

export type LODLevel = 'overview' | 'medium' | 'detail';

export interface RenderConfig {
  showLabels: boolean;
  showFill: boolean;
  showGrid: boolean; // SAHI grid overlay
  opacity: number;
  lodLevel: LODLevel;
}
