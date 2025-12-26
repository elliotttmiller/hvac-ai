"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import type { AnalysisResult } from '@/types/analysis';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Upload, FileText, Loader2, Wind, X, MapPin } from 'lucide-react';

export type AcceptMap = Record<string, string[]>;

export interface UploaderProps {
  acceptedMimeTypes?: AcceptMap;
  maxFiles?: number;
  maxSizeBytes?: number;
  showPreview?: boolean;
  showMetadataFields?: boolean | { projectId?: boolean; location?: boolean };
  action?: 'analyze' | 'upload' | 'none';
  analyzeEndpoint?: string;
  ariaLabel?: string;
  onFileSelected?: (file: File) => void;
  onUploadStart?: () => void;
  onUploadProgress?: (pct: number) => void;
  onAnalyzeStart?: () => void;
  onAnalyzeComplete?: (result: AnalysisResult) => void;
  onError?: (err: Error | string) => void;
}

const DEFAULT_ACCEPT: AcceptMap = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/tiff': ['.tiff', '.tif'],
  'application/pdf': ['.pdf'],
  'application/octet-stream': ['.dwg', '.dxf'],
};

export default function Uploader({
  acceptedMimeTypes = DEFAULT_ACCEPT,
  maxFiles = 1,
  maxSizeBytes = 500 * 1024 * 1024,
  showPreview = true,
  showMetadataFields = true,
  action = 'analyze',
  analyzeEndpoint = '/api/hvac/analyze?stream=1',
  ariaLabel = 'File uploader',
  onFileSelected,
  onUploadStart,
  onUploadProgress,
  onAnalyzeStart,
  onAnalyzeComplete,
  onError,
}: UploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [projectId, setProjectId] = useState('');
  const [location, setLocation] = useState('');
  const [progress, setProgress] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    if (rejectedFiles && rejectedFiles.length > 0) {
      const r = rejectedFiles[0];
      const msg = r?.errors?.[0]?.message || 'File rejected';
      setError(msg);
      onError?.(msg);
      return;
    }

    if (acceptedFiles.length === 0) return;
    const f = acceptedFiles[0];
    if (f.size > maxSizeBytes) {
      const msg = `File is too large (max ${(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB)`;
      setError(msg);
      onError?.(msg);
      return;
    }

    setFile(f);
    setError(null);
    onFileSelected?.(f);
    if (showPreview && f.type.startsWith('image/')) {
      const url = URL.createObjectURL(f);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }
  }, [maxSizeBytes, onFileSelected, onError, showPreview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: maxFiles > 1,
    maxFiles,
    accept: acceptedMimeTypes,
  });

  const handleCancel = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
    setAnalyzing(false);
    setProgress(0);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setError(null);
    setAnalyzing(true);
    setProgress(5);
    onAnalyzeStart?.();

    abortRef.current = new AbortController();

    try {
      const form = new FormData();
      form.append('file', file);
      form.append('image', file);
      if (projectId) form.append('projectId', projectId);
      if (location) form.append('location', location);

      const res = await fetch(analyzeEndpoint, {
        method: 'POST',
        body: form,
        signal: abortRef.current.signal,
        headers: { 'ngrok-skip-browser-warning': 'true', Accept: 'text/event-stream' },
      });

      if (!res.ok) {
        let msg = `Server Error (${res.status})`;
        try {
          const txt = await res.text();
          try { const j = JSON.parse(txt); msg = j.detail || j.error || txt; } catch { msg = txt || res.statusText; }
        } catch {}
        throw new Error(msg);
      }

      if (!res.body) throw new Error('No response body');

      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = '';
      setProgress(30);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split('\n\n');
        buf = parts.pop() || '';

        for (const part of parts) {
          if (!part.trim()) continue;
          const m = part.match(/^data:\s?(.+)/);
          const jsonStr = m ? m[1] : part;
          try {
            const data = JSON.parse(jsonStr);
            if (data.progress) {
              setProgress(data.progress);
              onUploadProgress?.(data.progress);
            }
            if (data.detections || data.segments) {
              // normalize like other components expect
              const normalized = normalizeResult(data, file.name);
              setProgress(100);
              onAnalyzeComplete?.(normalized);
            }
          } catch (e) {
            // ignore partial chunks
          }
        }
      }

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      onError?.(err instanceof Error ? err : String(err));
    } finally {
      setAnalyzing(false);
      abortRef.current = null;
    }
  };

  const normalizeResult = (data: any, fileName: string): AnalysisResult => {
    const detections = (data.detections || data.segments || []) as any[];
    const segments = detections.map((d, idx) => {
      let bbox = d.bbox;
      if (!bbox && d.obb) {
        const { x_center, y_center, width, height } = d.obb;
        bbox = [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2];
      }
      return {
        id: d.id || `seg-${idx}`,
        label: d.label || d.class || 'unknown',
        score: d.confidence ?? d.score ?? 0,
        bbox: bbox || [0, 0, 0, 0],
        points: d.points,
        obb: d.obb,
        rotation: d.rotation ?? d.obb?.rotation ?? 0,
        displayFormat: d.points ? 'polygon' : 'bbox',
        displayMask: null,
      } as any;
    });

    const counts: Record<string, number> = {};
    segments.forEach(s => { counts[s.label] = (counts[s.label] || 0) + 1; });

    return {
      analysis_id: (data.id as string) || `local-${Date.now()}`,
      file_name: fileName,
      status: 'completed',
      segments,
      total_components: segments.length,
      counts_by_category: counts,
      processing_time_seconds: (data.processing_time_seconds as number) || 0,
    } as AnalysisResult;
  };

  const showProjectField = typeof showMetadataFields === 'boolean' ? showMetadataFields : (showMetadataFields?.projectId ?? true);
  const showLocationField = typeof showMetadataFields === 'boolean' ? showMetadataFields : (showMetadataFields?.location ?? true);

  return (
    <div className="space-y-6 max-w-7xl mx-auto" aria-label={ariaLabel}>
      <Card className="border-2 shadow-sm">
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-2">
              <Upload className="h-5 w-5 text-primary" />
              <div>
                <CardTitle className="text-sm">Upload</CardTitle>
              </div>
            </div>
            <div>
              {/* controls could be extended */}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          {!file ? (
            <div {...getRootProps()} className={`relative border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all duration-150 ${isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-slate-200 hover:border-primary/40 hover:bg-slate-50'}`}>
              <input {...getInputProps()} aria-label="file input" />
              <div className="flex items-center gap-4 justify-center">
                <div className="p-3 bg-slate-100 rounded-md"><FileText className="h-6 w-6 text-slate-400" /></div>
                <div className="text-left">
                  <p className="text-base font-medium text-slate-700">{isDragActive ? 'Drop file to upload' : 'Click to upload or drag & drop'}</p>
                  <p className="text-xs text-slate-500 mt-0.5">Supported: images, pdf, dwg/dxf</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-4 p-3 border rounded-lg bg-white">
              <div className="flex-shrink-0 h-14 w-14 bg-slate-100 rounded overflow-hidden flex items-center justify-center">
                {previewUrl ? <img src={previewUrl} alt="thumb" className="h-full w-full object-cover" /> : <FileText className="text-slate-400" />}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <div className="truncate">
                    <p className="font-medium text-slate-900 truncate">{file.name}</p>
                    <p className="text-xs text-slate-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {action !== 'none' && <Button size="sm" onClick={() => { if (action === 'analyze') handleAnalyze(); }}>{analyzing ? <Loader2 className="animate-spin" /> : <Wind className="h-4 w-4" />}{' '}{action === 'analyze' ? 'Start AI' : 'Upload'}</Button>}
                    <Button variant="ghost" size="icon" onClick={() => { setFile(null); setPreviewUrl(null); setError(null); }} className="text-slate-400 hover:text-red-500"><X size={16} /></Button>
                  </div>
                </div>

                {(showProjectField || showLocationField) && (
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                    {showProjectField && (
                      <div className="space-y-2">
                        <Label htmlFor="projectId">Project ID</Label>
                        <Input id="projectId" placeholder="e.g. PRJ-2024-001" value={projectId} onChange={e => setProjectId(e.target.value)} />
                      </div>
                    )}
                    {showLocationField && (
                      <div className="space-y-2">
                        <Label htmlFor="location">Location</Label>
                        <div className="relative">
                          <MapPin className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
                          <Input id="location" className="pl-9" placeholder="e.g. Chicago, IL" value={location} onChange={e => setLocation(e.target.value)} />
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {analyzing && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm text-slate-600">
                <span className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin text-primary" />Processing...</span>
                <span className="font-mono">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              <Button variant="outline" size="sm" onClick={handleCancel} className="w-full text-red-500 hover:text-red-600 hover:bg-red-50">Cancel</Button>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
