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

const QuoteDashboard = dynamic(() => import('@/components/hvac/QuoteDashboard'), {
  ssr: false,
  loading: () => <div className="h-64 w-full bg-slate-50 animate-pulse rounded-lg flex items-center justify-center text-slate-400">Loading Quote Dashboard...</div>
});

export default function HVACBlueprintUploader({ onAnalysisComplete }: HVACBlueprintUploaderProps) {
  const router = useRouter();
  
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [uploaderExpanded, setUploaderExpanded] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [showQuoteDashboard, setShowQuoteDashboard] = useState(false);
  
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
    <div className="space-y-6 max-w-7xl mx-auto">
      <Card className="border-2 shadow-sm">
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-2">
              <Upload className="h-5 w-5 text-primary" />
              <div>
                <CardTitle className="text-sm">Upload HVAC Blueprint</CardTitle>
                <CardDescription className="text-xs">Supports PDF, DWG, DXF, PNG, JPG (max 500MB)</CardDescription>
              </div>
            </div>
            <div>
              <Button variant="ghost" size="sm" onClick={() => setUploaderExpanded(e => !e)}>
                {uploaderExpanded ? 'Collapse' : 'Details'}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {!uploadedFile ? (
            <div 
              {...getRootProps()} 
              className={`
                relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all duration-150
                ${isDragActive 
                  ? 'border-primary bg-primary/5 scale-[1.01]' 
                  : 'border-slate-200 hover:border-primary/40 hover:bg-slate-50'
                }
              `}
            >
              <input {...getInputProps()} />
              <div className="flex items-center gap-4 justify-center">
                <div className="p-3 bg-slate-100 rounded-md">
                  <Upload className="h-6 w-6 text-slate-400" />
                </div>
                <div className="text-left">
                  <p className="text-base font-medium text-slate-700">
                    {isDragActive ? 'Drop blueprint now' : 'Click to upload or drag & drop'}
                  </p>
                  <p className="text-xs text-slate-500 mt-0.5">
                    High-resolution images work best for OBB detection
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-4 p-3 border rounded-lg bg-white">
              <div className="flex-shrink-0 h-14 w-14 bg-slate-100 rounded overflow-hidden flex items-center justify-center">
                {uploadedFile.file.type.startsWith('image/') ? (
                  <img src={uploadedFile.previewUrl} alt="thumb" className="h-full w-full object-cover" />
                ) : (
                  <div className="text-slate-400"><FileText size={18} /></div>
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <div className="truncate">
                    <p className="font-medium text-slate-900 truncate">{uploadedFile.file.name}</p>
                    <p className="text-xs text-slate-500">{(uploadedFile.file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button size="sm" onClick={() => setUploaderExpanded(e => !e)}>{uploaderExpanded ? 'Hide' : 'Preview'}</Button>
                    <Button onClick={handleAnalyze} size="sm" className="font-semibold">Start AI</Button>
                    <Button variant="ghost" size="icon" onClick={handleRemoveFile} className="text-slate-400 hover:text-red-500">
                      <X size={16} />
                    </Button>
                  </div>
                </div>
                {uploaderExpanded && (
                  <div className="mt-3 border rounded-md overflow-hidden">
                    {uploadedFile.file.type.startsWith('image/') && (
                      <div className="relative h-48 w-full bg-slate-100 flex items-center justify-center overflow-hidden">
                        <img 
                          src={uploadedFile.previewUrl} 
                          alt="Preview" 
                          className="h-full w-full object-contain opacity-95" 
                        />
                      </div>
                    )}
                    <div className="p-3 grid grid-cols-1 md:grid-cols-2 gap-3">
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
                    <div className="p-3">
                      {!analyzing ? (
                        <Button onClick={handleAnalyze} size="lg" className="w-full font-semibold">
                          <Wind className="mr-2 h-4 w-4" />
                          Start AI Analysis
                        </Button>
                      ) : (
                        <div className="space-y-2">
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
                  </div>
                )}
              </div>
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

      {result && !showQuoteDashboard && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <Card className="overflow-hidden border-2 shadow-xl">
            <CardHeader className="bg-slate-50 border-b">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Analysis Results</CardTitle>
                  <CardDescription>ID: {result.analysis_id}</CardDescription>
                </div>
                <div className="flex gap-2">
                   <Button onClick={() => setShowQuoteDashboard(true)} className="gap-2">
                     <DollarSign className="h-4 w-4" />
                     Generate Quote
                   </Button>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="p-0">
              <div className="bg-slate-900 min-h-[640px] relative">
                <InferenceAnalysis
                  initialImage={uploadedFile?.file ?? null}
                  initialSegments={result.segments ?? undefined}
                  initialCount={{
                     total_objects_found: result.total_components || 0,
                     counts_by_category: result.counts_by_category || {}
                  }}
                />
              </div>

              {/* Sleek metric strip: Total, Unique, Avg Confidence, Top Category */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2 bg-transparent border-t">
                <div className="bg-white p-3 flex flex-col">
                  <span className="text-xs text-slate-500">Total Components</span>
                  <span className="text-lg font-semibold text-slate-900">{result.total_components}</span>
                </div>
                <div className="bg-white p-3 flex flex-col">
                  <span className="text-xs text-slate-500">Unique Categories</span>
                  <span className="text-lg font-semibold text-slate-900">{Object.keys(result.counts_by_category || {}).length}</span>
                </div>
                <div className="bg-white p-3 flex flex-col">
                  <span className="text-xs text-slate-500">Avg Confidence</span>
                  <span className="text-lg font-semibold text-slate-900">{(() => {
                    const segs = result.segments || [];
                    if (!segs.length) return '—';
                    const avg = segs.reduce((s, x) => s + (x.score || 0), 0) / segs.length;
                    return `${Math.round(avg * 100)}%`;
                  })()}</span>
                </div>
                <div className="bg-white p-3 flex flex-col">
                  <span className="text-xs text-slate-500">Top Category</span>
                  <span className="text-sm font-semibold text-slate-900">{(() => {
                    const counts = result.counts_by_category || {};
                    const entries = Object.entries(counts);
                    if (entries.length === 0) return '—';
                    entries.sort((a, b) => b[1] - a[1]);
                    const [label, cnt] = entries[0];
                    return `${label.replace(/_/g,' ')} • ${cnt}`;
                  })()}</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
      
      {/* Quote Dashboard - Full Screen */}
      {result && showQuoteDashboard && (
        <div className="fixed inset-0 z-50 bg-white">
          <QuoteDashboard
            projectId={projectId || result.analysis_id || 'HVAC-PROJ-001'}
            location={location || 'Atlanta, GA'}
            analysisResult={result}
            imageFile={uploadedFile?.file ?? null}
          />
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