/**
 * Annotation Store
 * Centralized state management for annotations with dirty tracking and delta computation
 */

import { useState, useCallback, useRef, useMemo } from 'react';
import type {
  EditableAnnotation,
  AnnotationState,
  AnnotationEdit,
  DeltaSavePayload,
} from '@/types/deep-zoom';
import type { Segment } from '@/types/analysis';
import { SpatialAnnotationIndex } from '@/lib/spatial-index';

export function useAnnotationStore() {
  const [state, setState] = useState<AnnotationState>({
    annotations: new Map(),
    selectedId: null,
    hoveredId: null,
    dirtyIds: new Set(),
    deletedIds: new Set(),
    confidenceThreshold: 0.5,
  });

  const spatialIndexRef = useRef(new SpatialAnnotationIndex());
  const editHistoryRef = useRef<AnnotationEdit[]>([]);

  /**
   * Initialize annotations from segments (YOLO object detection results)
   */
  const initializeFromSegments = useCallback((segments: Segment[]) => {
    const annotations = new Map<string, EditableAnnotation>();
    const newAnnotations: EditableAnnotation[] = [];

    segments.forEach((seg, idx) => {
      const id = `ann_${Date.now()}_${idx}`;

      // Prefer oriented bounding box (OBB) when available. Compute an
      // axis-aligned bounding box (AABB) from the OBB for spatial indexing
      // and compatibility with legacy UI components.
      let bbox: [number, number, number, number] = [0, 0, 0, 0];
      let obb = undefined;

      if (seg.obb && typeof seg.obb.x_center === 'number') {
        obb = {
          x_center: seg.obb.x_center,
          y_center: seg.obb.y_center,
          width: seg.obb.width,
          height: seg.obb.height,
          rotation: seg.obb.rotation ?? 0,
        };
        const cx = obb.x_center;
        const cy = obb.y_center;
        const w = obb.width;
        const h = obb.height;
        bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
      } else if (Array.isArray(seg.bbox) && seg.bbox.length === 4) {
        const [x1, y1, x2, y2] = seg.bbox;
        bbox = [x1, y1, x2, y2];
      }

      const annotation: EditableAnnotation = {
        id,
        label: seg.label,
        score: seg.score,
        bbox,
        obb,
        isDirty: false,
        isNew: false,
      };
      annotations.set(id, annotation);
      newAnnotations.push(annotation);
    });

    // Bulk load into spatial index for better performance
    spatialIndexRef.current.clear();
    spatialIndexRef.current.bulkLoad(newAnnotations);

    setState((prev) => ({
      ...prev,
      annotations,
      dirtyIds: new Set(),
      deletedIds: new Set(),
      selectedId: null,
      hoveredId: null,
    }));

    editHistoryRef.current = [];
  }, []);

  /**
   * Add a new annotation (phantom creation)
   */
  const addAnnotation = useCallback((annotation: Omit<EditableAnnotation, 'id' | 'isDirty' | 'isNew'>) => {
    const id = `ann_new_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const newAnnotation: EditableAnnotation = {
      ...annotation,
      id,
      isDirty: true,
      isNew: true,
    };

    spatialIndexRef.current.insert(newAnnotation);

    setState((prev) => {
      const annotations = new Map(prev.annotations);
      annotations.set(id, newAnnotation);
      const dirtyIds = new Set(prev.dirtyIds);
      dirtyIds.add(id);

      return {
        ...prev,
        annotations,
        dirtyIds,
        selectedId: id,
      };
    });

    editHistoryRef.current.push({
      id,
      type: 'add',
      data: newAnnotation,
      timestamp: Date.now(),
    });

    return id;
  }, []);

  /**
   * Update an existing annotation
   */
  const updateAnnotation = useCallback((id: string, updates: Partial<EditableAnnotation>) => {
    setState((prev) => {
      const annotation = prev.annotations.get(id);
      if (!annotation) return prev;

      const oldAnnotation = annotation;
      const updatedAnnotation: EditableAnnotation = {
        ...annotation,
        ...updates,
        isDirty: true,
      };

      // Update spatial index
      spatialIndexRef.current.update(oldAnnotation, updatedAnnotation);

      const annotations = new Map(prev.annotations);
      annotations.set(id, updatedAnnotation);
      const dirtyIds = new Set(prev.dirtyIds);
      dirtyIds.add(id);

      return {
        ...prev,
        annotations,
        dirtyIds,
      };
    });

    editHistoryRef.current.push({
      id,
      type: 'modify',
      data: updates,
      timestamp: Date.now(),
    });
  }, []);

  /**
   * Delete an annotation
   */
  const deleteAnnotation = useCallback((id: string) => {
    setState((prev) => {
      const annotation = prev.annotations.get(id);
      if (!annotation) return prev;

      // Remove from spatial index
      spatialIndexRef.current.remove(annotation);

      const annotations = new Map(prev.annotations);
      annotations.delete(id);
      const deletedIds = new Set(prev.deletedIds);
      if (!annotation.isNew) {
        deletedIds.add(id);
      }
      const dirtyIds = new Set(prev.dirtyIds);
      dirtyIds.delete(id);

      return {
        ...prev,
        annotations,
        deletedIds,
        dirtyIds,
        selectedId: prev.selectedId === id ? null : prev.selectedId,
      };
    });

    editHistoryRef.current.push({
      id,
      type: 'delete',
      data: {},
      timestamp: Date.now(),
    });
  }, []);

  /**
   * Reclassify an annotation (change label)
   */
  const reclassifyAnnotation = useCallback((id: string, newLabel: string) => {
    updateAnnotation(id, { label: newLabel });

    editHistoryRef.current.push({
      id,
      type: 'reclassify',
      data: { label: newLabel },
      timestamp: Date.now(),
    });
  }, [updateAnnotation]);

  /**
   * Set confidence threshold for filtering
   */
  const setConfidenceThreshold = useCallback((threshold: number) => {
    setState((prev) => ({
      ...prev,
      confidenceThreshold: threshold,
    }));
  }, []);

  /**
   * Set selected annotation
   */
  const setSelectedId = useCallback((id: string | null) => {
    setState((prev) => ({
      ...prev,
      selectedId: id,
    }));
  }, []);

  /**
   * Set hovered annotation
   */
  const setHoveredId = useCallback((id: string | null) => {
    setState((prev) => ({
      ...prev,
      hoveredId: id,
    }));
  }, []);

  /**
   * Get filtered annotations based on confidence threshold
   */
  const filteredAnnotations = useMemo(() => {
    const filtered: EditableAnnotation[] = [];
    state.annotations.forEach((ann) => {
      if (ann.score >= state.confidenceThreshold) {
        filtered.push(ann);
      }
    });
    return filtered;
  }, [state.annotations, state.confidenceThreshold]);

  /**
   * Compute delta payload for saving
   */
  const computeDelta = useCallback((): DeltaSavePayload => {
    const added: EditableAnnotation[] = [];
    const modified: EditableAnnotation[] = [];
    const deleted = Array.from(state.deletedIds);

    state.dirtyIds.forEach((id) => {
      const annotation = state.annotations.get(id);
      if (annotation) {
        if (annotation.isNew) {
          added.push(annotation);
        } else {
          modified.push(annotation);
        }
      }
    });

    return {
      added,
      modified,
      deleted,
    };
  }, [state.annotations, state.dirtyIds, state.deletedIds]);

  /**
   * Clear dirty flags after successful save
   */
  const clearDirtyFlags = useCallback(() => {
    setState((prev) => {
      const annotations = new Map(prev.annotations);
      prev.dirtyIds.forEach((id) => {
        const ann = annotations.get(id);
        if (ann) {
          annotations.set(id, { ...ann, isDirty: false, isNew: false });
        }
      });

      return {
        ...prev,
        annotations,
        dirtyIds: new Set(),
        deletedIds: new Set(),
      };
    });

    editHistoryRef.current = [];
  }, []);

  /**
   * Check if there are unsaved changes
   */
  const hasUnsavedChanges = useMemo(() => {
    return state.dirtyIds.size > 0 || state.deletedIds.size > 0;
  }, [state.dirtyIds.size, state.deletedIds.size]);

  return {
    // State
    annotations: state.annotations,
    selectedId: state.selectedId,
    hoveredId: state.hoveredId,
    confidenceThreshold: state.confidenceThreshold,
    filteredAnnotations,
    hasUnsavedChanges,

    // Actions
    initializeFromSegments,
    addAnnotation,
    updateAnnotation,
    deleteAnnotation,
    reclassifyAnnotation,
    setConfidenceThreshold,
    setSelectedId,
    setHoveredId,
    computeDelta,
    clearDirtyFlags,

    // Spatial index
    spatialIndex: spatialIndexRef.current,

    // Edit history
    editHistory: editHistoryRef.current,
  };
}
