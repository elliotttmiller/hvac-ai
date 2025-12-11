'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
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
  FileBarChart
} from 'lucide-react';
import { toast } from 'sonner';
import { decodeRLEMask, drawMaskOnCanvas } from '@/lib/rle-decoder';

interface SegmentResult {
  label: string;
  score: number;
  mask: {
    size: [number, number];
    counts: string;
  };
  bbox: number[];
}

interface CountResult {
  total_objects_found: number;
  counts_by_category: Record<string, number>;
}

interface Point {
  x: number;
  y: number;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

export default function SAMAnalysis() {
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [countLoading, setCountLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState<SegmentResult[]>([]);
  const [countResult, setCountResult] = useState<CountResult | null>(null);
  const [clickMode, setClickMode] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setUploadedImage(file);
      setError(null);
      setSegments([]);
      setCountResult(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setImagePreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg']
    },
    maxFiles: 1,
    multiple: false
  });

  useEffect(() => {
    if (imagePreview && canvasRef.current && imageRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = imageRef.current;
      
      img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        
        // Redraw all segments when image loads
        if (segments.length > 0) {
          drawAllMasks();
        }
      };
    }
  }, [imagePreview]);
  
  // Redraw masks when segments change
  useEffect(() => {
    if (segments.length > 0 && canvasRef.current && imageRef.current) {
      drawAllMasks();
    }
  }, [segments]);

  const handleCanvasClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!clickMode || !uploadedImage || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Get click coordinates relative to canvas
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    await handleSegment({ x, y });
  };

  const handleSegment = async (point: Point) => {
    if (!uploadedImage) return;

    if (!API_BASE_URL) {
      const message = 'API URL not configured. Please set NEXT_PUBLIC_API_URL environment variable.';
      setError(message);
      toast.error(message);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', uploadedImage);
      
      const prompt = JSON.stringify({
        type: 'point',
        data: {
          coords: [point.x, point.y],
          label: 1
        }
      });
      formData.append('prompt', prompt);

      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Segmentation failed');
      }

      const data = await response.json();
      
      // Add new segments to the list (keep previous ones)
      if (data.segments && data.segments.length > 0) {
        setSegments(prev => [...prev, ...data.segments]);
        toast.success(`Segmented: ${data.segments[0].label}`);
      }
    } catch (err: any) {
      const message = err.message || 'Failed to segment component';
      setError(message);
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const handleCountAll = async () => {
    if (!uploadedImage) return;

    if (!API_BASE_URL) {
      const message = 'API URL not configured. Please set NEXT_PUBLIC_API_URL environment variable.';
      setError(message);
      toast.error(message);
      return;
    }

    setCountLoading(true);
    setError(null);
    setCountResult(null);

    let toastId: string | number | undefined;

    try {
      const formData = new FormData();
      formData.append('image', uploadedImage);

      toastId = toast.loading('Analyzing and counting components...', { duration: 0 });

      const response = await fetch(`${API_BASE_URL}/api/count`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Counting failed');
      }

      const data = await response.json();
      setCountResult(data);
      toast.success('Counting completed', { id: toastId });
    } catch (err: any) {
      const message = err.message || 'Failed to count components';
      setError(message);
      toast.error(message, { id: toastId, duration: 5000 });
    } finally {
      setCountLoading(false);
    }
  };

  const drawAllMasks = () => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all segment masks
    segments.forEach((segment, index) => {
      try {
        // Decode RLE mask
        const maskArray = decodeRLEMask(segment.mask);
        
        if (maskArray.length === 0) {
          console.warn('Empty mask for segment:', segment.label);
          return;
        }

        // Use different colors for different segments
        const colors: [number, number, number][] = [
          [0, 255, 0],    // Green
          [255, 0, 0],    // Red
          [0, 0, 255],    // Blue
          [255, 255, 0],  // Yellow
          [255, 0, 255],  // Magenta
          [0, 255, 255],  // Cyan
        ];
        const color = colors[index % colors.length];

        // Draw the mask with semi-transparency
        drawMaskOnCanvas(ctx, maskArray, color, 0.4);

        // Draw bounding box and label
        const [x, y, w, h] = segment.bbox;
        ctx.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Draw label background
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const labelText = `${segment.label} (${(segment.score * 100).toFixed(1)}%)`;
        const textMetrics = ctx.measureText(labelText);
        const labelWidth = textMetrics.width + 10;
        ctx.fillRect(x, y - 25, labelWidth, 25);

        // Draw label text
        ctx.fillStyle = 'white';
        ctx.font = '14px Arial';
        ctx.fillText(labelText, x + 5, y - 7);
      } catch (error) {
        console.error('Failed to draw mask for segment:', segment.label, error);
      }
    });
  };

  const exportToCSV = () => {
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
  };

  const handleRemoveFile = () => {
    setUploadedImage(null);
    setImagePreview(null);
    setSegments([]);
    setCountResult(null);
    setError(null);
    setClickMode(false);
  };

  return (
    <div className="container mx-auto py-8 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">SAM Analysis</h1>
          <p className="text-muted-foreground">
            AI-powered P&ID and HVAC diagram component segmentation and counting
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Diagram
          </CardTitle>
          <CardDescription>
            Upload a P&ID or HVAC diagram for interactive segmentation and automated counting
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!uploadedImage ? (
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'
              }`}
            >
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              {isDragActive ? (
                <p className="text-lg font-medium">Drop the diagram here...</p>
              ) : (
                <>
                  <p className="text-lg font-medium mb-2">
                    Drag & drop a diagram, or click to select
                  </p>
                  <p className="text-sm text-gray-500">
                    Supports PNG, JPG, JPEG
                  </p>
                </>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between border rounded-lg p-4 bg-gray-50">
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-5 w-5 text-green-500" />
                  <span className="font-medium">{uploadedImage.name}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRemoveFile}
                  disabled={loading || countLoading}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button
                  onClick={() => setClickMode(!clickMode)}
                  variant={clickMode ? 'default' : 'outline'}
                  disabled={loading || countLoading}
                  className="w-full"
                >
                  <MousePointerClick className="mr-2 h-4 w-4" />
                  {clickMode ? 'Click to Segment (Active)' : 'Enable Click-to-Segment'}
                </Button>
                <Button
                  onClick={handleCountAll}
                  disabled={loading || countLoading}
                  className="w-full"
                >
                  {countLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Scan className="mr-2 h-4 w-4" />
                      Analyze and Count All Components
                    </>
                  )}
                </Button>
              </div>
              
              {segments.length > 0 && (
                <Button
                  onClick={() => setSegments([])}
                  variant="outline"
                  size="sm"
                  disabled={loading || countLoading}
                  className="w-full"
                >
                  <X className="mr-2 h-4 w-4" />
                  Clear All Segments ({segments.length})
                </Button>
              )}
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {imagePreview && (
        <Card>
          <CardHeader>
            <CardTitle>Interactive Canvas</CardTitle>
            <CardDescription>
              {clickMode
                ? 'Click on any component to segment it'
                : 'Enable click-to-segment mode to interact with the diagram'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="relative inline-block max-w-full">
              <img
                ref={imageRef}
                src={imagePreview}
                alt="Uploaded diagram"
                className="max-w-full h-auto"
                style={{ display: 'block' }}
              />
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className={`absolute top-0 left-0 max-w-full h-auto ${
                  clickMode ? 'cursor-crosshair' : 'cursor-default'
                }`}
                style={{ pointerEvents: clickMode ? 'auto' : 'none' }}
              />
            </div>
            {loading && (
              <div className="mt-4">
                <Progress value={50} className="w-full" />
                <p className="text-sm text-center text-gray-600 mt-2">
                  Segmenting component...
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {segments.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Segmentation Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {segments.map((segment, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 border rounded">
                  <div>
                    <span className="font-medium">{segment.label}</span>
                    <Badge className="ml-2" variant="secondary">
                      {(segment.score * 100).toFixed(1)}% confidence
                    </Badge>
                  </div>
                  <div className="text-sm text-gray-500">
                    BBox: [{segment.bbox.join(', ')}]
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {countResult && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <FileBarChart className="h-5 w-5" />
                  Component Count Report
                </CardTitle>
                <CardDescription>
                  Total objects found: {countResult.total_objects_found}
                </CardDescription>
              </div>
              <Button onClick={exportToCSV} variant="outline">
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {Object.entries(countResult.counts_by_category)
                  .sort((a, b) => b[1] - a[1])
                  .map(([label, count]) => (
                    <div key={label} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate">{label}</span>
                        <Badge variant="default">{count}</Badge>
                      </div>
                    </div>
                  ))}
              </div>
              
              <Separator />
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b">
                    <tr>
                      <th className="text-left py-2 px-4">Component Type</th>
                      <th className="text-right py-2 px-4">Count</th>
                      <th className="text-right py-2 px-4">Percentage</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(countResult.counts_by_category)
                      .sort((a, b) => b[1] - a[1])
                      .map(([label, count]) => (
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
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
