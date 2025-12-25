/**
 * Object detection results using bounding boxes.
 * The platform uses YOLOv11 object detection model with bounding box outputs.
 * 
 * Note: Polygon and mask fields are deprecated and kept for backward compatibility.
 * New implementations should only use bbox for object detection.
 */

export interface Segment {
  label: string;
  score: number;
  bbox: number[]; // [x1, y1, x2, y2] bounding box coordinates
  // Deprecated fields (kept for backward compatibility)
  mask?: never;
  polygon?: never;
  mask_png?: never;
}

export interface CountResult {
  total_objects_found: number;
  counts_by_category: Record<string, number>;
}

export interface AnalysisResult {
  analysis_id?: string;
  status?: string;
  file_name?: string;
  detected_components?: unknown[];
  total_components?: number;
  processing_time_ms?: number;
  processing_time_seconds?: number;
  segments?: Segment[];
  counts_by_category?: Record<string, number>;
  total_objects_found?: number;
}
