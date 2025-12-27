'use client';

/**
 * DeepZoomInferenceAnalysis Component
 * Complete integration of deep-zoom viewport, annotation management, and HITL editing
 */

import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { toast } from 'sonner';
import DeepZoomViewer from '@/components/viewer/DeepZoomViewer';
import AnnotationSidebar from '@/components/inference/AnnotationSidebar';
import { useAnnotationStore } from '@/lib/annotation-store';
import { useQuoteStore } from '@/lib/pricing-store';
import type { RenderConfig } from '@/types/deep-zoom';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  Upload,
  Scan,
  Loader2,
  X,
  Eye,
  EyeOff,
  Grid3x3,
  Layers,
} from 'lucide-react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';

export default function DeepZoomInferenceAnalysis() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Render configuration
  const [renderConfig, setRenderConfig] = useState<RenderConfig>({
    showLabels: true,
    showFill: true,
    showGrid: false,
    opacity: 0.2,
    lodLevel: 'detail',
  });

  // Annotation store
  const {
    annotations,
    selectedId,
    hoveredId,
    confidenceThreshold,
    filteredAnnotations,
    hasUnsavedChanges,
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
    spatialIndex,
  } = useAnnotationStore();

  // Quote store - for pricing/quote data
  const { setQuote } = useQuoteStore();

  // Handle file upload
  const handleDrop = useCallback((files: File[]) => {
    if (files.length > 0) {
      const file = files[0];

      const maxSize = 50 * 1024 * 1024; // 50MB for large blueprints
      if (file.size > maxSize) {
        toast.error(
          `File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds 50MB limit.`,
          { duration: 5000 }
        );
        return;
      }

      const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/bmp'];
      const fileType = file.type.toLowerCase();
      if (!validTypes.includes(fileType) && !file.name.match(/\.(jpg|jpeg|png|tiff|tif|bmp)$/i)) {
        toast.error('Please upload a valid image file (JPG, PNG, TIFF, or BMP)', {
          duration: 4000,
        });
        return;
      }

      setUploadedImage(file);
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      toast.success(`Loaded ${file.name} (${(file.size / 1024).toFixed(0)}KB)`, { duration: 2000 });
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/tiff': ['.tiff', '.tif'],
      'image/bmp': ['.bmp'],
    },
    maxFiles: 1,
    multiple: false,
  });

  // Handle analysis
  const handleAnalyze = async () => {
    if (!uploadedImage) {
      toast.error('Please upload an image first');
      return;
    }

    setIsAnalyzing(true);

    const formData = new FormData();
    // Provide both names for the uploaded file to maximize compatibility
    formData.append('image', uploadedImage);
    formData.append('file', uploadedImage);
    formData.append('conf_threshold', '0.50');
    formData.append('nms_threshold', '0.45');

    try {
      const res = await fetch(`${API_BASE_URL}/api/hvac/analyze?stream=1`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorText = await res.text();
        let errorMessage = 'Analysis failed';

        if (res.status === 413) {
          errorMessage = 'Image too large for server. Please resize to <10MB.';
        } else if (res.status === 415) {
          errorMessage = 'Unsupported image format. Use JPG, PNG, or TIFF.';
        } else if (res.status === 500) {
          errorMessage = 'Server error during analysis. Please try again.';
        } else if (res.status === 503) {
          errorMessage = 'Analysis service unavailable. Please try again later.';
        }

        toast.error(errorMessage, { duration: 5000 });
        throw new Error(errorMessage);
      }

      const data = await res.json();

      // Handle new response format: { detections, quote, image_shape }
      // Also handle legacy format: { segments, total_objects_found }
      const detections = data.detections || data.segments || [];
      
      if (!detections || !Array.isArray(detections)) {
        toast.error('Invalid response format from analysis service');
        throw new Error('Invalid response structure');
      }

      // Initialize annotations from detections
      initializeFromSegments(detections);

      // Extract and store quote if present in response
      if (data.quote) {
        setQuote(data.quote);
        toast.success(
          `Analysis complete: ${detections.length} component(s) found, quote generated`,
          { duration: 3000 }
        );
      } else {
        toast.success(`Analysis complete: Found ${detections.length} component${detections.length !== 1 ? 's' : ''}`, {
          duration: 3000,
        });
      }
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to analyze image';
      if (!errorMsg.includes('Analysis')) {
        toast.error(errorMsg, { duration: 4000 });
      }
      console.error('[DeepZoomInferenceAnalysis] Analysis error:', e);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Handle save
  const handleSave = async () => {
    if (!hasUnsavedChanges) {
      toast.info('No changes to save');
      return;
    }

    setIsSaving(true);

    try {
      const delta = computeDelta();

      console.log('[DeepZoomInferenceAnalysis] Delta to save:', delta);

      // NOTE: Annotation persistence is currently a client-side feature.
      // The backend doesn't have a /api/hvac/annotations/save endpoint yet.
      // For now, we store changes locally and in the annotation store.
      // TODO: Implement server-side annotation persistence in the future.

      clearDirtyFlags();
      toast.success(
        `Saved locally: ${delta.added.length} new, ${delta.modified.length} modified, ${delta.deleted.length} deleted annotations. ` +
        `(Server persistence coming soon)`,
        { duration: 3000 }
      );
    } catch (e) {
      const errorMsg = e instanceof Error ? e.message : 'Failed to save changes';
      toast.error(errorMsg, { duration: 4000 });
      console.error('[DeepZoomInferenceAnalysis] Save error:', e);
    } finally {
      setIsSaving(false);
    }
  };

  // Cleanup URL when component unmounts
  useEffect(() => {
    return () => {
      if (imageUrl) {
        URL.revokeObjectURL(imageUrl);
      }
    };
  }, [imageUrl]);

  // Auto-scroll sidebar to selected annotation
  useEffect(() => {
    if (selectedId) {
      // Sidebar will handle scrolling via virtualized list
    }
  }, [selectedId]);

  return (
    <div className="container mx-auto py-8 space-y-6 max-w-[1920px]">
      {/* Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle>HVAC Blueprint Deep-Zoom Analysis</CardTitle>
          <CardDescription>
            Upload a P&ID or Floor Plan for high-resolution analysis with Google Maps-style navigation
          </CardDescription>
        </CardHeader>
        <CardContent>
          {!uploadedImage ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${
                isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700">Drop large blueprint here</p>
              <p className="text-sm text-gray-500">
                Supports PNG, JPG, TIFF (up to 50MB) - Optimized for 10,000px+ images
              </p>
            </div>
          ) : (
            <div className="flex items-center justify-between bg-slate-50 p-4 rounded-lg border">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 bg-blue-100 rounded flex items-center justify-center text-blue-600">
                  <Scan className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium text-slate-900">{uploadedImage.name}</p>
                  <p className="text-xs text-slate-500">
                    {(uploadedImage.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleAnalyze} disabled={isAnalyzing}>
                  {isAnalyzing ? (
                    <Loader2 className="animate-spin mr-2 h-4 w-4" />
                  ) : (
                    <Scan className="mr-2 h-4 w-4" />
                  )}
                  Analyze Blueprint
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    setUploadedImage(null);
                    setImageUrl(null);
                    initializeFromSegments([]);
                  }}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Main Viewport and Sidebar */}
      {imageUrl && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Main Viewport */}
          <Card className="lg:col-span-3">
            <CardHeader className="pb-4">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">Deep-Zoom Viewport</CardTitle>

                {/* Viewport Controls */}
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Switch
                      id="show-labels"
                      checked={renderConfig.showLabels}
                      onCheckedChange={(checked) =>
                        setRenderConfig((prev) => ({ ...prev, showLabels: checked }))
                      }
                    />
                    <Label htmlFor="show-labels" className="text-sm cursor-pointer">
                      Labels
                    </Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      id="show-fill"
                      checked={renderConfig.showFill}
                      onCheckedChange={(checked) =>
                        setRenderConfig((prev) => ({ ...prev, showFill: checked }))
                      }
                    />
                    <Label htmlFor="show-fill" className="text-sm cursor-pointer">
                      Fill
                    </Label>
                  </div>
                  <div className="flex items-center gap-2">
                    <Switch
                      id="show-grid"
                      checked={renderConfig.showGrid}
                      onCheckedChange={(checked) =>
                        setRenderConfig((prev) => ({ ...prev, showGrid: checked }))
                      }
                    />
                    <Label htmlFor="show-grid" className="text-sm cursor-pointer">
                      <Grid3x3 className="h-4 w-4 inline mr-1" />
                      SAHI Grid
                    </Label>
                  </div>
                  <div className="flex items-center gap-2 border-l pl-4">
                    <Label className="text-sm">Opacity</Label>
                    <Slider
                      value={[renderConfig.opacity * 100]}
                      onValueChange={([value]) =>
                        setRenderConfig((prev) => ({ ...prev, opacity: value / 100 }))
                      }
                      min={0}
                      max={100}
                      step={5}
                      className="w-20"
                    />
                  </div>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-0">
              <div className="relative h-[700px] bg-slate-900">
                <DeepZoomViewer
                  imageUrl={imageUrl}
                  annotations={filteredAnnotations}
                  selectedId={selectedId}
                  hoveredId={hoveredId}
                  renderConfig={renderConfig}
                  onSelect={setSelectedId}
                  onHover={setHoveredId}
                  onAnnotationUpdate={updateAnnotation}
                  spatialIndex={spatialIndex}
                />
              </div>
            </CardContent>
          </Card>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <AnnotationSidebar
              annotations={annotations}
              filteredAnnotations={filteredAnnotations}
              selectedId={selectedId}
              confidenceThreshold={confidenceThreshold}
              onConfidenceChange={setConfidenceThreshold}
              onSelect={setSelectedId}
              onDelete={deleteAnnotation}
              onReclassify={reclassifyAnnotation}
              hasUnsavedChanges={hasUnsavedChanges}
              onSave={handleSave}
            />
          </div>
        </div>
      )}

      {/* Help Text */}
      {imageUrl && (
        <Card>
          <CardContent className="pt-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-slate-600">
              <div>
                <h4 className="font-semibold mb-2">Navigation</h4>
                <ul className="space-y-1">
                  <li>• Drag to pan viewport</li>
                  <li>• Scroll to zoom in/out</li>
                  <li>• Double-click to zoom in</li>
                  <li>• Click annotation to select</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Editing</h4>
                <ul className="space-y-1">
                  <li>• Click Edit icon to reclassify</li>
                  <li>• Click Delete icon to remove</li>
                  <li>• Adjust confidence slider to filter</li>
                  <li>• Changes auto-tracked</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-2">Performance</h4>
                <ul className="space-y-1">
                  <li>• Viewport culling enabled</li>
                  <li>• Only visible items rendered</li>
                  <li>• Spatial indexing for speed</li>
                  <li>• Optimized for 10K+ objects</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
