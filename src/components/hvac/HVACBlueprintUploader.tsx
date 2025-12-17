'use client';

import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import type { AnalysisResult, Segment } from '@/types/analysis';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import dynamic from 'next/dynamic';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
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
  X
} from 'lucide-react';

interface UploadedFile {
  file: File;
}

// Reuse a small, explicit shape for segment results to avoid `any` usage

// AnalysisResult and Segment types are imported from src/types/analysis.ts

interface HVACBlueprintUploaderProps {
  onAnalysisComplete?: (result: AnalysisResult) => void;
}

export default function HVACBlueprintUploader({ onAnalysisComplete }: HVACBlueprintUploaderProps) {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [projectId, setProjectId] = useState('');
  const [location, setLocation] = useState('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      if (file.size > 500 * 1024 * 1024) {
        setError('File size must be less than 500MB');
        return;
      }
      setUploadedFile({ file });
      setError(null);
      setResult(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    multiple: false
  });

  const handleAnalyze = async () => {
    if (!uploadedFile) return;
    setAnalyzing(true);
    setError(null);
    setProgress(10);
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile.file);
      if (projectId) formData.append('projectId', projectId);
      if (location) formData.append('location', location);

      // Use streaming endpoint through the Next.js proxy. We request
      // SSE by setting Accept and adding ?stream=1 so the proxy forwards
      // to the Python streaming endpoint.
      setProgress(20);
      const response = await fetch('/api/hvac/analyze?stream=1', {
        method: 'POST',
        body: formData,
        headers: {
          'ngrok-skip-browser-warning': '69420',
          Accept: 'text/event-stream',
        }
      });

      if (!response.ok || !response.body) {
        // Fallback to non-streaming JSON path
        const fallback = await response.json().catch(() => null);
        throw new Error(fallback?.error || 'Analysis failed');
      }

      // Stream parser
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffered = '';

      setProgress(30);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffered += decoder.decode(value, { stream: true });

        // SSE events are separated by double-newline
        const parts = buffered.split('\n\n');
        buffered = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;
          // Find the data: line(s)
          const lines = part.split('\n').map(l => l.trim()).filter(Boolean);
          const dataLine = lines.find(l => l.startsWith('data:'));
          if (!dataLine) continue;
          const jsonStr = dataLine.replace(/^data:\s?/, '');
          try {
            const obj = JSON.parse(jsonStr);
            if (obj.type === 'progress') {
              if (typeof obj.percent === 'number') setProgress(obj.percent);
            } else if (obj.type === 'status') {
              // Optionally surface status messages in the UI/logs
              // setProgress(Math.max(progress, 5));
            } else if (obj.type === 'result') {
              const finalRaw = obj.result as AnalysisResult;
              const final = normalizeAnalysisResult(finalRaw);
              setResult(final);
              setProgress(100);
              if (onAnalysisComplete) onAnalysisComplete(final);
            } else if (obj.type === 'error') {
              setError(obj.message || 'Analysis error');
            }
          } catch (e) {
            console.error('Failed to parse SSE payload', e, jsonStr);
          }
        }
      }

      // Ensure reading complete
      setProgress(100);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message || 'Failed to analyze blueprint');
      } else {
        setError('Failed to analyze blueprint');
      }
    } finally {
      setAnalyzing(false);
    }
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setResult(null);
    setError(null);
    setProgress(0);
  };

  const InferenceAnalysis = dynamic(() => import('@/components/inference/InferenceAnalysis'), { ssr: false });

  // Helper to normalize processing time to seconds and format safely
  const formatProcessingTime = (r: AnalysisResult | null, decimals = 2) => {
    if (!r) return (0).toFixed(decimals);
    const secs = typeof r.processing_time_seconds === 'number'
      ? r.processing_time_seconds
      : (typeof r.processing_time_ms === 'number' ? r.processing_time_ms / 1000 : undefined);
    return (typeof secs === 'number') ? secs.toFixed(decimals) : (0).toFixed(decimals);
  };

  // Normalize analysis result to prefer polygon masks for display when available.
  /* eslint-disable @typescript-eslint/no-explicit-any */
  const normalizeAnalysisResult = (r: any): AnalysisResult => {
    if (!r) return r;
    const normalized = { ...r } as any;
    if (Array.isArray(r.segments)) {
      normalized.segments = r.segments.map((s: any) => {
        const seg = { ...s } as any;
        // Prefer polygon if available, otherwise rle or mask
        if (seg.polygon) {
          seg.displayFormat = 'polygon';
          seg.displayMask = seg.polygon;
        } else if (seg.rle || seg.mask) {
          seg.displayFormat = 'rle';
          seg.displayMask = seg.rle || seg.mask;
        } else {
          seg.displayFormat = 'bbox';
          seg.displayMask = null;
        }
        return seg as Segment;
      });
    }
    return normalized as AnalysisResult;
  };
  /* eslint-enable @typescript-eslint/no-explicit-any */

  // Small sub-component to handle safe navigation to the analysis details page
  function ViewResultsButton({ result }: { result: AnalysisResult }) {
    const router = useRouter();
    const analysisId = result?.analysis_id;

    const handleClick = () => {
      if (!analysisId) return; // guard
      router.push(`/analysis/${analysisId}`);
    };

    return (
      <Button className="w-full" onClick={handleClick} disabled={!analysisId}>
        <DollarSign className="mr-2 h-4 w-4" />View Detailed Results & Generate Estimate
      </Button>
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload HVAC Blueprint
          </CardTitle>
          <CardDescription>
            Upload PDF, DWG, DXF, or image files for AI-powered analysis
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {!uploadedFile ? (
            <div {...getRootProps()} className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${isDragActive ? 'border-primary bg-primary/5' : 'border-gray-300 hover:border-primary'}`}>
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              {isDragActive ? (
                <p className="text-lg font-medium">Drop the blueprint here...</p>
              ) : (
                <>
                  <p className="text-lg font-medium mb-2">Drag & drop a blueprint, or click to select</p>
                  <p className="text-sm text-gray-500">Supports PDF, DWG, DXF, PNG, JPG (max 500MB)</p>
                </>
              )}
            </div>
          ) : (
            <div className="border rounded-lg p-4 bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <FileText className="h-10 w-10 text-blue-500" />
                  <div>
                    <p className="font-medium">{uploadedFile.file.name}</p>
                    <p className="text-sm text-gray-500">{(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB</p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={handleRemoveFile} disabled={analyzing}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}

          {uploadedFile && !result && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4">
              <div className="space-y-2">
                <Label htmlFor="projectId">Project ID (Optional)</Label>
                <Input id="projectId" placeholder="e.g., HVAC-2024-001" value={projectId} onChange={(e) => setProjectId(e.target.value)} disabled={analyzing} />
              </div>
              <div className="space-y-2">
                <Label htmlFor="location"><MapPin className="h-4 w-4 inline mr-1" />Location</Label>
                <Input id="location" placeholder="e.g., Chicago, IL" value={location} onChange={(e) => setLocation(e.target.value)} disabled={analyzing} />
              </div>
            </div>
          )}

          {uploadedFile && !result && (
            <Button onClick={handleAnalyze} disabled={analyzing} className="w-full" size="lg">
              {analyzing ? (<><Loader2 className="mr-2 h-4 w-4 animate-spin" />Analyzing Blueprint...</>) : (<><Wind className="mr-2 h-4 w-4" />Analyze HVAC System</>)}
            </Button>
          )}

          {analyzing && (
            <div className="space-y-2">
              <Progress value={progress} className="w-full" />
              <p className="text-sm text-center text-gray-600">{progress < 40 ? 'Uploading blueprint...' : progress < 80 ? 'Analyzing HVAC components...' : 'Finalizing results...'}</p>
            </div>
          )}

          {error && (<Alert variant="destructive"><AlertCircle className="h-4 w-4" /><AlertDescription>{error}</AlertDescription></Alert>)}

          {result && (
            <Alert className="bg-green-50 border-green-200">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-800">
                <strong>Analysis Complete!</strong> Detected {result.total_components || 0} HVAC components in {formatProcessingTime(result, 2)} seconds.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>Blueprint: {result.file_name} | Analysis ID: {result.analysis_id || 'N/A'}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Render SAMAnalysis inline so users can interact without navigation */}
            <div className="mb-4">
              <InferenceAnalysis
                initialImage={uploadedFile?.file ?? null}
                initialSegments={result.segments ?? undefined}
                initialCount={result.counts_by_category ? { total_objects_found: (result.total_components ?? result.total_objects_found ?? 0), counts_by_category: result.counts_by_category } : undefined}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-2"><Wind className="h-5 w-5 text-blue-500" /><span className="text-sm font-medium">Components</span></div>
                <p className="text-2xl font-bold">{result.total_components}</p>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-2"><CheckCircle2 className="h-5 w-5 text-green-500" /><span className="text-sm font-medium">Status</span></div>
                <Badge variant="default" className="bg-green-500">{result.status}</Badge>
              </div>
              <div className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-2"><FileText className="h-5 w-5 text-purple-500" /><span className="text-sm font-medium">Processing Time</span></div>
                <p className="text-2xl font-bold">{formatProcessingTime(result, 1)}s</p>
              </div>
            </div>
            <ViewResultsButton result={result} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}
