'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { RLEMask, Segment, CountResult } from '@/types/analysis';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  Upload,
  Scan,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
  X,
  MousePointerClick,
  FileBarChart,
} from 'lucide-react';
import { toast } from 'sonner';
import { decodeRLEMask, drawMaskOnCanvas } from '@/lib/rle-decoder';

// --- Type Definitions ---
type AnalysisState = 'idle' | 'segmenting' | 'counting';

// --- API Configuration ---
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';

export default function SAMAnalysis({
  initialImage,
  initialSegments,
  initialCount
}: {
  initialImage?: File | null;
  initialSegments?: Segment[];
  initialCount?: CountResult | null;
}) {
  // --- State Management ---
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [countResult, setCountResult] = useState<CountResult | null>(null);
  const [clickMode, setClickMode] = useState(false);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Initialize as null so this module does not access browser globals during SSR
  const imageRef = useRef<HTMLImageElement | null>(null);

  // Ensure we have an Image object on the client (avoid using DOM APIs during SSR)
  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!imageRef.current) {
      const img = new Image();
      // allow cross-origin if images served remotely
      try { img.crossOrigin = 'anonymous'; } catch (e) { /* ignore */ }
      imageRef.current = img;
    }
    return () => {
      // cleanup: dereference to help GC
      imageRef.current = null;
    };
  }, []);

  // --- Core Drawing Logic ---
  const drawCanvasContent = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.src) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 1. Draw the base image onto the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img as CanvasImageSource, 0, 0, canvas.width, canvas.height);

    // 2. Draw all segment masks and labels on top
    segments.forEach((segment, index) => {
      try {
        const maskArray = decodeRLEMask(segment.mask);
        if (maskArray.length === 0) return;

        const colors: [number, number, number][] = [
          [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]
        ];
        const color = colors[index % colors.length];

        // Draw the mask
        drawMaskOnCanvas(ctx, maskArray, color, 0.5);

        // Draw bounding box and label
        const [x, y, w, h] = segment.bbox;
        ctx.strokeStyle = `rgb(${color.join(',')})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        
        const labelText = `${segment.label} (${(segment.score * 100).toFixed(1)}%)`;
        ctx.font = '14px Arial';
        const textMetrics = ctx.measureText(labelText);
        const labelWidth = textMetrics.width + 10;
        const labelHeight = 22;
        const labelX = Math.max(0, x); // Prevent drawing off-canvas
        const labelY = Math.max(labelHeight, y); // Prevent drawing off-canvas

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(labelX, labelY - labelHeight, labelWidth, labelHeight);
        
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, labelX + 5, labelY - 7);
      } catch (e) {
        console.error('Failed to draw segment:', segment.label, e);
      }
    });
  }, [segments]);

  // --- Image Loading and Canvas Sizing ---
  useEffect(() => {
    const img = imageRef.current;
    const canvas = canvasRef.current;

    if (!img) return;

    const handleImageLoad = () => {
      if (canvas) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        requestAnimationFrame(drawCanvasContent);
      }
    };

    img.addEventListener('load', handleImageLoad);

    if (uploadedImage) {
      const url = URL.createObjectURL(uploadedImage);
      img.src = url;
      // Cleanup function
      return () => {
        URL.revokeObjectURL(url);
        img.removeEventListener('load', handleImageLoad);
      };
    } else {
      // If no uploaded image, clear the src if element exists
      img.src = '';
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  }, [uploadedImage, drawCanvasContent]);

  // If parent provided initial props, initialize internal state on mount
  useEffect(() => {
    if (initialImage) {
      setUploadedImage(initialImage);
    }
    if (initialSegments && initialSegments.length > 0) {
      setSegments(initialSegments);
    }
    if (initialCount) {
      setCountResult(initialCount);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Re-draw canvas whenever segments change
  useEffect(() => {
    requestAnimationFrame(drawCanvasContent);
  }, [segments, drawCanvasContent]);

  // --- Event Handlers ---
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setUploadedImage(acceptedFiles[0]);
      setSegments([]);
      setCountResult(null);
      setError(null);
      setClickMode(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    maxFiles: 1,
    multiple: false,
  });

  const handleCanvasClick = useCallback(async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!clickMode || !uploadedImage || !canvasRef.current || analysisState !== 'idle') return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const point = {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    };

    setAnalysisState('segmenting');
    setError(null);

    try {
      if (!API_BASE_URL) throw new Error('API URL not configured.');

      const formData = new FormData();
      formData.append('image', uploadedImage);
      formData.append('coords', `${point.x},${point.y}`);

      const response = await fetch(`${API_BASE_URL}/api/analyze`, { method: 'POST', body: formData });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Segmentation request failed');
      }

      const data = await response.json();
      if (data.segments && data.segments.length > 0) {
        // coerce to Segment[]
        setSegments(prev => [...prev, ...data.segments as Segment[]]);
        toast.success(`Segmented: ${data.segments[0].label}`);
      } else {
        toast.info('No object found at this location.');
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      toast.error(message);
    } finally {
      setAnalysisState('idle');
    }
  }, [clickMode, uploadedImage, analysisState]);

  const handleCountAll = useCallback(async () => {
    if (!uploadedImage || analysisState !== 'idle') return;

    setAnalysisState('counting');
    setError(null);
    setCountResult(null);

    let toastId: string | number | undefined;

    try {
      if (!API_BASE_URL) throw new Error('API URL not configured.');
      
      const formData = new FormData();
      formData.append('image', uploadedImage);

      toastId = toast.loading('Analyzing and counting all components...', { duration: Infinity });

      const response = await fetch(`${API_BASE_URL}/api/count`, { method: 'POST', body: formData });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Counting request failed');
      }

      const data = await response.json();
  setCountResult(data as CountResult);
  toast.success(`Analysis complete. Found ${data.total_objects_found} objects.`, { id: toastId });
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      toast.error(message, { id: toastId, duration: 5000 });
    } finally {
      setAnalysisState('idle');
    }
  }, [uploadedImage, analysisState]);

  const handleClear = () => {
    setSegments([]);
    setCountResult(null);
    setError(null);
  };
  
  const exportToCSV = useCallback(() => {
    if (!countResult) return;

    const csvContent = [
      ['Component Type', 'Count'],
      ...Object.entries(countResult.counts_by_category).map(([label, count]) => [label, count.toString()])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'component-counts.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [countResult]);

  const isBusy = analysisState !== 'idle';

  return (
    <div className="container mx-auto py-8 space-y-6">
      {/* ... Header remains the same ... */}

      <Card>
        <CardHeader>
            <CardTitle className="flex items-center gap-2"><Upload className="h-5 w-5" />Upload Diagram</CardTitle>
            <CardDescription>Upload a P&ID or HVAC diagram for analysis</CardDescription>
        </CardHeader>
        <CardContent>
          {!uploadedImage ? (
             <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'}`}>
                <input {...getInputProps()} />
                <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p className="text-lg font-medium mb-2">Drag & drop a diagram, or click to select</p>
                <p className="text-sm text-gray-500">Supports PNG, JPG, JPEG</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between border rounded-lg p-4 bg-gray-50">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="font-medium">{uploadedImage.name}</span>
                </div>
                <Button variant="ghost" size="sm" onClick={() => setUploadedImage(null)} disabled={isBusy}><X className="h-4 w-4" /></Button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button onClick={() => setClickMode(!clickMode)} variant={clickMode ? 'default' : 'outline'} disabled={isBusy}>
                  <MousePointerClick className="mr-2 h-4 w-4" />
                  {clickMode ? 'Click to Segment (Active)' : 'Enable Click-to-Segment'}
                </Button>
                <Button onClick={handleCountAll} disabled={isBusy}>
                  {analysisState === 'counting' ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Analyzing...</>) : (<><Scan className="mr-2 h-4 w-4" />Analyze & Count All</>)}
                </Button>
              </div>
              
              {(segments.length > 0 || countResult) && (
                <Button onClick={handleClear} variant="outline" size="sm" disabled={isBusy} className="w-full">
                  <X className="mr-2 h-4 w-4" />Clear Results
                </Button>
              )}
            </div>
          )}

          {error && (
            <Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>
          )}
        </CardContent>
      </Card>

      {uploadedImage && (
        <Card>
          <CardHeader>
            <CardTitle>Interactive Canvas</CardTitle>
            <CardDescription>
              {clickMode ? 'Click on any component to segment it' : 'Enable click-to-segment mode to interact'}
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center">
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={clickMode && !isBusy ? 'cursor-crosshair' : 'cursor-default'}
            />
            {analysisState === 'segmenting' && (
              <div className="mt-4 w-full text-center">
                <p className="text-sm text-gray-600 mt-2 flex items-center justify-center">
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Segmenting component...
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* --- Results Cards (Segments & Count Report) --- */}
      {/* These sections are well-structured and can remain as they are. */}
      {/* I've included them here for completeness. */}
      {segments.length > 0 && (
        <Card>
          <CardHeader><CardTitle>Segmentation Results</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            {segments.map((segment, idx) => (
              <div key={idx} className="flex items-center justify-between p-3 border rounded">
                <div>
                  <span className="font-medium">{segment.label}</span>
                  <Badge className="ml-2" variant="secondary">
                    {(segment.score * 100).toFixed(1)}%
                  </Badge>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {countResult && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2"><FileBarChart className="h-5 w-5" />Component Count Report</CardTitle>
                <CardDescription>Total objects found: {countResult.total_objects_found}</CardDescription>
              </div>
              <Button onClick={exportToCSV} variant="outline"><Download className="mr-2 h-4 w-4" />Export CSV</Button>
            </div>
          </CardHeader>
          <CardContent>
             <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b"><tr>
                      <th className="text-left py-2 px-4">Component Type</th>
                      <th className="text-right py-2 px-4">Count</th>
                      <th className="text-right py-2 px-4">Percentage</th>
                  </tr></thead>
                  <tbody>
                    {Object.entries(countResult.counts_by_category).sort((a, b) => b[1] - a[1]).map(([label, count]) => (
                        <tr key={label} className="border-b last:border-0">
                          <td className="py-2 px-4">{label}</td>
                          <td className="text-right py-2 px-4 font-medium">{count}</td>
                          <td className="text-right py-2 px-4 text-sm text-gray-600">
                            {((count / countResult.total_objects_found) * 100).toFixed(1)}%
                          </td>
                        </tr>
                    ))}
                  </tbody>
                </table>
              </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}