/**
 * @deprecated RLE-based masks (COCO RLE) are deprecated in this project.
 * The platform now prefers vector polygon coordinates (see `Segment.polygon`).
 *
 * Migration: backend services return `segments[].polygon` (vector coordinates)
 * and `segments[].mask` (RLE) will be removed in a future release. If
 * you still need RLE, encode polygons to RLE on-demand server-side using
 * an optional dependency such as `pycocotools`.
 */
export interface RLEMask {
  /** Size as [height, width] */
  size: [number, number];
  /** COCO RLE counts (may be bytes or a string depending on encoder) */
  counts: string;
}

export interface Segment {
  label: string;
  score: number;
  // Legacy RLE mask (optional). New pipeline prefers vector polygons.
  mask?: RLEMask;
  bbox: number[];
  // Optional polygon coordinates. Can be a single polygon (array of [x,y])
  // or multiple polygons ([[[x,y], ...], [[x,y], ...]]).
  polygon?: number[][] | number[][][];
  // Optional PNG (base64) representation of the mask produced server-side
  mask_png?: string;
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
