'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import type { FileRejection } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import type { AnalysisResult, Segment } from '@/types/analysis';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import dynamic from 'next/dynamic';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Upload,
  FileText,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Wind,
  DollarSign,
  MapPin,
  X,
  Image as ImageIcon
} from 'lucide-react';

// --- Types ---

interface UploadedFile {
  file: File;
  previewUrl: string;
}

interface BackendDetection {
  id?: string;
  label: string;
  class?: string;
  score: number;
  confidence?: number;
  bbox: [number, number, number, number];
  points?: number[][];
  rotation?: number;
  obb?: {
    x_center: number;
    y_center: number;
    width: number;
    height: number;
    rotation: number;
  };
}

interface HVACBlueprintUploaderProps {
  onAnalysisComplete?: (result: AnalysisResult) => void;
}

const InferenceAnalysis = dynamic(() => import('@/components/inference/InferenceAnalysis'), { 
  ssr: false,
  loading: () => <div className="h-64 w-full bg-slate-50 animate-pulse rounded-lg flex items-center justify-center text-slate-400">Loading Visualization Engine...</div>
});

export default function HVACBlueprintUploader({ onAnalysisComplete }: HVACBlueprintUploaderProps) {
  const router = useRouter();
  
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  
  const [projectId, setProjectId] = useState('');
  const [location, setLocation] = useState('');

  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (uploadedFile?.previewUrl) {
        URL.revokeObjectURL(uploadedFile.previewUrl);
      }
    };
  }, [uploadedFile]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      setError(rejection.errors[0]?.message || 'File rejected. Please check format and size.');
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      if (file.size > 500 * 1024 * 1024) {
        setError('File is too large (Max 500MB)');
        return;
      }

      const previewUrl = URL.createObjectURL(file);
      setUploadedFile({ file, previewUrl });
      setError(null);
      setResult(null);
      setProgress(0);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    multiple: false,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/tiff': ['.tiff', '.tif'],
      'application/pdf': ['.pdf'],
      'application/octet-stream': ['.dwg', '.dxf'],
    },
  });

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setAnalyzing(false);
    setProgress(0);
    setError('Analysis cancelled by user.');
  };

  const handleRemoveFile = () => {
    if (analyzing) handleCancel();
    if (uploadedFile?.previewUrl) URL.revokeObjectURL(uploadedFile.previewUrl);
    setUploadedFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) return;

    setAnalyzing(true);
    setError(null);
    setProgress(5);
    setResult(null);

    abortControllerRef.current = new AbortController();

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile.file);
      formData.append('image', uploadedFile.file); // Duplicate for backend compatibility
      
      if (projectId) formData.append('projectId', projectId);
      if (location) formData.append('location', location);

      setProgress(15);

      const response = await fetch('/api/hvac/analyze?stream=1', {
        method: 'POST',
        body: formData,
        signal: abortControllerRef.current.signal,
        headers: {
          'ngrok-skip-browser-warning': 'true',
          'Accept': 'text/event-stream',
        },
      });

      if (!response.ok) {
        let errMsg = `Server Error (${response.status})`;
        try {
          const text = await response.text();
          try {
             const jsonErr = JSON.parse(text);
             errMsg = jsonErr.detail || jsonErr.error || text;
          } catch {
             errMsg = text || response.statusText;
          }
        } catch {}
        
        if (response.status === 422) {
          errMsg = "Validation Error: Backend did not receive 'file' or 'image' field.";
        }
        throw new Error(errMsg);
      }

      if (!response.body) throw new Error('No response body received.');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = '';

      setProgress(30);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffered += chunk;

        const parts = buffered.split('\n\n');
        buffered = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;

          // Handle SSE lines or raw JSON chunks
          const dataMatch = part.match(/^data:\s?(.+)/);
          const jsonStr = dataMatch ? dataMatch[1] : part;

          try {
            const data = JSON.parse(jsonStr);

            if (data.detections || data.segments) {
              const normalized = normalizeAnalysisResult(data, uploadedFile.file.name);
              setResult(normalized);
              setProgress(100);
              if (onAnalysisComplete) onAnalysisComplete(normalized);
            } else if (data.progress) {
              setProgress(data.progress);
            }
          } catch (e) {
            // Ignore incomplete JSON chunks
          }
        }
      }

    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      console.error('Analysis failed:', err);
      setError(err instanceof Error ? err.message : 'Unknown analysis error');
      setProgress(0);
    } finally {
      setAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  // Normalize Backend OBB/Poly/BBox to Frontend Segment
  const normalizeAnalysisResult = (data: BackendResponse, fileName: string): AnalysisResult => {
    
    const segments: Segment[] = (data.detections || []).map((d: BackendDetection, idx: number) => {
      
      // Prefer explicit bbox, fallback to OBB bounds
      let bbox = d.bbox; 
      if (!bbox && d.obb) {
        const { x_center, y_center, width, height } = d.obb;
        bbox = [
          x_center - width / 2,
          y_center - height / 2,
          x_center + width / 2,
          y_center + height / 2
        ];
      }

      return {
        id: d.id || `seg-${idx}`,
        label: d.label || d.class || 'unknown',
        score: d.confidence ?? d.score ?? 0,
        bbox: bbox || [0,0,0,0],
        points: d.points, 
        obb: d.obb,
        rotation: d.rotation ?? d.obb?.rotation ?? 0,
        displayFormat: d.points ? 'polygon' : 'bbox',
        displayMask: null
      };
    });

    const counts: Record<string, number> = {};
    segments.forEach(s => {
      counts[s.label] = (counts[s.label] || 0) + 1;
    });

    return {
      analysis_id: (data.id as string) || `local-${Date.now()}`,
      file_name: fileName,
      status: 'completed',
      segments: segments,
      total_components: segments.length,
      counts_by_category: counts,
      processing_time_seconds: (data.processing_time_seconds as number) || 0
    };
  };

  const formatTime = (res: AnalysisResult) => {
    return typeof res.processing_time_seconds === 'number' 
      ? res.processing_time_seconds.toFixed(2) 
      : '0.00';
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <Card className="border-2 shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-primary" />
            Upload HVAC Blueprint
          </CardTitle>
          <CardDescription>
            Supports PDF, DWG, DXF, PNG, JPG (max 500MB)
          </CardDescription>
        </CardHeader>
        
        <CardContent className="space-y-6">
          {!uploadedFile ? (
            <div 
              {...getRootProps()} 
              className={`
                relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all duration-200
                ${isDragActive 
                  ? 'border-primary bg-primary/5 scale-[1.01]' 
                  : 'border-slate-200 hover:border-primary/50 hover:bg-slate-50'
                }
              `}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center gap-3">
                <div className="p-4 bg-slate-100 rounded-full">
                  <Upload className="h-8 w-8 text-slate-400" />
                </div>
                <div>
                  <p className="text-lg font-semibold text-slate-700">
                    {isDragActive ? 'Drop blueprint now' : 'Click to upload or drag & drop'}
                  </p>
                  <p className="text-sm text-slate-500 mt-1">
                    High-resolution images work best for OBB detection
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="border rounded-xl overflow-hidden bg-slate-50">
              <div className="p-4 border-b flex items-center justify-between bg-white">
                <div className="flex items-center gap-3">
                  <div className="h-10 w-10 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600">
                    <FileText size={20} />
                  </div>
                  <div>
                    <p className="font-medium text-slate-900">{uploadedFile.file.name}</p>
                    <p className="text-xs text-slate-500">{(uploadedFile.file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                </div>
                <Button variant="ghost" size="icon" onClick={handleRemoveFile} className="text-slate-400 hover:text-red-500">
                  <X size={18} />
                </Button>
              </div>
              
              {uploadedFile.file.type.startsWith('image/') && (
                <div className="relative h-48 w-full bg-slate-100 flex items-center justify-center overflow-hidden">
                  <img 
                    src={uploadedFile.previewUrl} 
                    alt="Preview" 
                    className="h-full w-full object-contain opacity-90" 
                  />
                  <div className="absolute bottom-2 right-2">
                    <Badge variant="secondary" className="bg-white/90 backdrop-blur">
                      <ImageIcon className="w-3 h-3 mr-1" /> Preview
                    </Badge>
                  </div>
                </div>
              )}
            </div>
          )}

          {uploadedFile && !result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2">
              <div className="space-y-2">
                <Label htmlFor="projectId">Project ID</Label>
                <Input 
                  id="projectId" 
                  placeholder="e.g. PRJ-2024-001" 
                  value={projectId} 
                  onChange={e => setProjectId(e.target.value)}
                  disabled={analyzing} 
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <div className="relative">
                  <MapPin className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                  <Input 
                    id="location" 
                    className="pl-9" 
                    placeholder="e.g. Chicago, IL" 
                    value={location} 
                    onChange={e => setLocation(e.target.value)}
                    disabled={analyzing} 
                  />
                </div>
              </div>
            </div>
          )}

          {uploadedFile && !result && (
            <div className="space-y-4 pt-2">
              {!analyzing ? (
                <Button onClick={handleAnalyze} size="lg" className="w-full font-semibold shadow-lg shadow-primary/20">
                  <Wind className="mr-2 h-5 w-5" />
                  Start AI Analysis
                </Button>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-sm text-slate-600">
                    <span className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin text-primary" />
                      {progress < 30 ? 'Uploading & Preprocessing...' : 'Running YOLOv11-OBB Inference...'}
                    </span>
                    <span className="font-mono">{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                  <Button variant="outline" size="sm" onClick={handleCancel} className="w-full text-red-500 hover:text-red-600 hover:bg-red-50">
                    Cancel Analysis
                  </Button>
                </div>
              )}
            </div>
          )}

          {error && (
            <Alert variant="destructive" className="animate-in shake">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Analysis Failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {result && (
            <Alert className="bg-green-50 border-green-200 text-green-800 animate-in zoom-in-95">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription className="flex justify-between items-center">
                <span>
                  <strong>Success!</strong> Found {result.total_components} components.
                </span>
                <span className="text-xs opacity-70">
                  {formatTime(result)}s processing
                </span>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {result && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <Card className="overflow-hidden border-2 shadow-xl">
            <CardHeader className="bg-slate-50 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>ID: {result.analysis_id}</CardDescription>
                </div>
                <div className="flex gap-2">
                   <Button variant="outline" onClick={() => router.push(`/analysis/${result.analysis_id}`)}>
                     <DollarSign className="mr-2 h-4 w-4" />
                     Generate Quote
                   </Button>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="p-0">
              <div className="bg-slate-900 min-h-[500px] relative">
                <InferenceAnalysis
                  initialImage={uploadedFile?.file ?? null}
                  initialSegments={result.segments ?? undefined}
                  initialCount={{
                     total_objects_found: result.total_components || 0,
                     counts_by_category: result.counts_by_category || {}
                  }}
                />
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-200 border-t">
                <div className="bg-white p-4">
                  <p className="text-sm text-slate-500">Total Components</p>
                  <p className="text-2xl font-bold text-slate-900">{result.total_components}</p>
                </div>
                <div className="bg-white p-4">
                  <p className="text-sm text-slate-500">Unique Categories</p>
                  <p className="text-2xl font-bold text-slate-900">
                    {Object.keys(result.counts_by_category || {}).length}
                  </p>
                </div>
                <div className="bg-white p-4">
                  <p className="text-sm text-slate-500">Confidence Score</p>
                  <p className="text-2xl font-bold text-green-600">High</p>
                </div>
                <div className="bg-white p-4">
                  <p className="text-sm text-slate-500">Processing Time</p>
                  <p className="text-2xl font-bold text-slate-900">{formatTime(result)}s</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

// Backend response wrapper (avoid `any` in normalize function)
interface BackendResponse {
  id?: string;
  detections?: BackendDetection[];
  segments?: Segment[];
  processing_time_seconds?: number;
  counts_by_category?: Record<string, number>;
  [key: string]: unknown;
}