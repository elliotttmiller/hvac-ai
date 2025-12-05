'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
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

interface AnalysisResult {
  analysis_id: string;
  status: string;
  file_name: string;
  detected_components: any[];
  total_components: number;
  processing_time_seconds: number;
}

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

      setProgress(30);
      const response = await fetch('/api/hvac/analyze', {
        method: 'POST',
        body: formData,
      });
      setProgress(70);

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Analysis failed');
      }

      const data = await response.json();
      setProgress(100);
      setResult(data);
      
      if (onAnalysisComplete) {
        onAnalysisComplete(data);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to analyze blueprint');
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
                <strong>Analysis Complete!</strong> Detected {result.total_components} HVAC components in {result.processing_time_seconds.toFixed(2)} seconds.
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>Blueprint: {result.file_name} | Analysis ID: {result.analysis_id}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
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
                <p className="text-2xl font-bold">{result.processing_time_seconds.toFixed(1)}s</p>
              </div>
            </div>
            <Button className="w-full" onClick={() => { window.location.href = `/analysis/${result.analysis_id}`; }}>
              <DollarSign className="mr-2 h-4 w-4" />View Detailed Results & Generate Estimate
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
