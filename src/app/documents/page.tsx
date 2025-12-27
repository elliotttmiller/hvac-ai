"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Upload,
  FileText,
  Image,
  File,
  Loader2,
  CheckCircle,
  XCircle,
  Clock,
  ArrowLeft,
  Eye,
  Download
} from 'lucide-react';
import Link from 'next/link';
import { FileUploader } from '@/components/features/ingestion/FileUploader';

interface Document {
  id: string;
  name: string;
  type: string;
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  size: number;
  url: string;
  category: string;
  created_at: string;
  extracted_text?: string;
  confidence?: number;
}

export default function DocumentsPage() {
  const searchParams = useSearchParams();
  const projectId = searchParams.get('projectId');

  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [project, setProject] = useState<{ id: string; name: string; location?: string; status?: string; components?: number; estimatedCost?: number; date?: string; climateZone?: string; description?: string; } | null>(null);

  // Load project details
  useEffect(() => {
    if (!projectId) return;

    fetch('/api/projects', { headers: { 'ngrok-skip-browser-warning': '69420' } })
      .then(res => res.json())
      .then(data => {
        const foundProject = data.projects?.find((p: { id: string; name: string; location?: string; status?: string; components?: number; estimatedCost?: number; date?: string; climateZone?: string; description?: string; }) => p.id === projectId);
        setProject(foundProject);
      })
      .catch(err => console.error('Failed to load project', err));
  }, [projectId]);

  const loadDocuments = useCallback(async () => {
    if (!projectId) return;

    try {
      setLoading(true);
      const response = await fetch(`/api/upload?projectId=${projectId}`, {
        headers: { 'ngrok-skip-browser-warning': '69420' }
      });

      if (!response.ok) {
        throw new Error('Failed to load documents');
      }

      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to load documents:', err);
      setError('Failed to load documents');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  // Load documents for this project
  useEffect(() => {
    loadDocuments();
  }, [loadDocuments]);

  const handleFileUpload = async (file: File) => {
    if (!projectId) return;

    try {
      setUploading(true);
      setError(null);

      const formData = new FormData();
      formData.append('file', file);
      formData.append('projectId', projectId);
      formData.append('category', 'blueprint');

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
        headers: { 'ngrok-skip-browser-warning': '69420' }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();

      // Add the new document to the list
      setDocuments((prev: Document[]) => [result.document, ...prev]);

    } catch (err) {
      console.error('Upload failed:', err);
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const getFileIcon = (type: string) => {
    if (type.startsWith('image/')) return <Image className="h-4 w-4" />;
    if (type === 'application/pdf') return <FileText className="h-4 w-4" />;
    return <File className="h-4 w-4" />;
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (!projectId) {
    return (
      <div className="container mx-auto py-8">
        <Alert>
          <AlertDescription>Project ID is required</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/projects">
            <Button variant="outline" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Projects
            </Button>
          </Link>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              {project ? project.name : 'Project'} Documents
            </h1>
            <p className="text-muted-foreground">
              Upload and manage blueprints for analysis
            </p>
          </div>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Blueprint
          </CardTitle>
          <CardDescription>
            Upload HVAC blueprints, CAD files, or other project documents for AI analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <FileUploader
            onUpload={handleFileUpload}
            isProcessing={uploading}
            title="Drop your blueprint here"
            subtitle="Supports PDF, images, DWG, and DXF files up to 500MB"
            maxSizeBytes={500 * 1024 * 1024}
          />
        </CardContent>
      </Card>

      {/* Documents List */}
      <Card>
        <CardHeader>
          <CardTitle>Project Documents</CardTitle>
          <CardDescription>
            {documents.length} document{documents.length !== 1 ? 's' : ''} uploaded
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
              <span className="ml-2">Loading documents...</span>
            </div>
          ) : documents.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No documents uploaded yet</p>
              <p className="text-sm">Upload your first blueprint to get started</p>
            </div>
          ) : (
            <div className="space-y-4">
              {documents.map((doc: Document) => (
                <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    {getFileIcon(doc.type)}
                    <div>
                      <div className="font-medium">{doc.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {formatFileSize(doc.size)} • Uploaded {new Date(doc.created_at).toLocaleDateString()}
                      </div>
                      {doc.extracted_text && (
                        <div className="text-xs text-muted-foreground mt-1">
                          OCR: {doc.confidence ? `${doc.confidence}% confidence` : 'Completed'}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="flex items-center gap-1">
                      {getStatusIcon(doc.status)}
                      {doc.status}
                    </Badge>

                    <div className="flex gap-1">
                      <Button variant="outline" size="sm">
                        <Eye className="h-3 w-3" />
                      </Button>
                      <Button variant="outline" size="sm">
                        <Download className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}