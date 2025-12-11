export interface RLEMask {
  size: [number, number];
  counts: string;
}

export interface Segment {
  label: string;
  score: number;
  mask: RLEMask;
  bbox: number[];
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
