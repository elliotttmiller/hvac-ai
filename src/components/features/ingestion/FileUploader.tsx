'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Upload, FileText, Loader2, X } from 'lucide-react';

export type AcceptMap = Record<string, string[]>;

export interface FileUploaderProps {
  /** File type restrictions */
  accept?: AcceptMap;
  /** Maximum file size in bytes */
  maxSizeBytes?: number;
  /** Whether to show file preview */
  showPreview?: boolean;
  /** Custom title for the uploader */
  title?: string;
  /** Custom subtitle/description */
  subtitle?: string;
  /** Whether the uploader is currently processing */
  isProcessing?: boolean;
  /** Callback when a file is selected/uploaded */
  onUpload?: (file: File) => void;
  /** Callback when processing starts */
  onProcessingStart?: () => void;
  /** Callback when processing completes with result */
  onProcessingComplete?: (result: unknown) => void;
  /** Callback when an error occurs */
  onError?: (error: string) => void;
  /** Custom processing button text */
  processButtonText?: string;
  /** Whether to show processing controls */
  showProcessControls?: boolean;
}

const DEFAULT_ACCEPT: AcceptMap = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/tiff': ['.tiff', '.tif'],
  'application/pdf': ['.pdf'],
  'application/octet-stream': ['.dwg', '.dxf'],
};

export function FileUploader({
  accept = DEFAULT_ACCEPT,
  maxSizeBytes = 50 * 1024 * 1024, // 50MB default
  showPreview = true,
  title = "Upload File",
  subtitle = "Drag and drop or click to select",
  isProcessing = false,
  onUpload,
  onProcessingStart,
  onProcessingComplete,
  onError,
  processButtonText = "Process",
  showProcessControls = false,
}: FileUploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
    // Handle rejected files
    if (rejectedFiles && rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      const errorMessage = rejection?.errors?.[0]?.message || 'File rejected';
      setError(errorMessage);
      onError?.(errorMessage);
      return;
    }

    // Handle accepted files
    if (acceptedFiles.length === 0) return;

    const selectedFile = acceptedFiles[0];

    // Check file size
    if (selectedFile.size > maxSizeBytes) {
      const sizeError = `File is too large (max ${(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB)`;
      setError(sizeError);
      onError?.(sizeError);
      return;
    }

    // Set file and create preview
    setFile(selectedFile);
    setError(null);

    if (showPreview && selectedFile.type.startsWith('image/')) {
      const url = URL.createObjectURL(selectedFile);
      setPreviewUrl(url);
    } else {
      setPreviewUrl(null);
    }

    // Notify parent
    onUpload?.(selectedFile);
  }, [maxSizeBytes, showPreview, onUpload, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    maxFiles: 1,
    accept,
    disabled: isProcessing,
  });

  const handleCancel = () => {
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }
  };

  const clearFile = () => {
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  };

  return (
    <Card className="border-2 shadow-sm">
      <Card className="border-2 shadow-sm">
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-200
            ${isDragActive ? 'border-primary bg-primary/5 scale-[1.01]' : 'border-slate-200 hover:bg-slate-50'}
            ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
            ${file ? 'hidden' : ''}
          `}
        >
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4">
            <div className="p-4 bg-slate-100 rounded-full">
              <Upload className="h-8 w-8 text-slate-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
              <p className="text-sm text-slate-500 mt-1">{subtitle}</p>
              <p className="text-xs text-slate-400 mt-2">
                Max size: {(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB
              </p>
            </div>
          </div>
        </div>

        {file && (
          <div className="p-6">
            <div className="flex items-start gap-4 p-4 border rounded-lg bg-white">
              <div className="flex-shrink-0 h-16 w-16 bg-slate-100 rounded-lg overflow-hidden flex items-center justify-center">
                {previewUrl ? (
                  <img src={previewUrl} alt="Preview" className="h-full w-full object-cover" />
                ) : (
                  <FileText className="h-8 w-8 text-slate-400" />
                )}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-slate-900 truncate">{file.name}</h4>
                    <p className="text-sm text-slate-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>

                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={clearFile}
                    className="text-slate-400 hover:text-red-500 flex-shrink-0 ml-2"
                    disabled={isProcessing}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                {showProcessControls && (
                  <div className="mt-4 flex gap-2">
                    <Button
                      onClick={onProcessingStart}
                      disabled={isProcessing}
                      className="flex items-center gap-2"
                    >
                      {isProcessing ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Upload className="h-4 w-4" />
                      )}
                      {isProcessing ? 'Processing...' : processButtonText}
                    </Button>

                    {isProcessing && (
                      <Button
                        variant="outline"
                        onClick={handleCancel}
                        className="text-red-500 hover:text-red-600 hover:bg-red-50"
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {isProcessing && (
          <div className="p-6 border-t">
            <div className="space-y-3">
              <div className="flex items-center justify-between text-sm text-slate-600">
                <span className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  Processing file...
                </span>
              </div>
              <Progress value={100} className="h-2 animate-pulse" />
            </div>
          </div>
        )}

        {error && (
          <div className="p-6 border-t">
            <Alert variant="destructive">
              <AlertTitle>Upload Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          </div>
        )}
      </Card>
    </Card>
  );
}