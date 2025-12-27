"use client";

import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
  Download,
  X
} from 'lucide-react';
import Link from 'next/link';
import { FileUploader } from '@/components/features/ingestion/FileUploader';

interface Document {
  id: string;
  projectId: string;
  name: string;
  type: string;
  status: 'processing' | 'completed' | 'error';
  size: number;
  url: string;
  uploadedAt: string;
  extractedText?: string;
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
  const [viewerOpen, setViewerOpen] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);

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
      const response = await fetch(`/api/analysis?projectId=${projectId}`, {
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

      const response = await fetch('/api/analysis', {
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
      if (result.document) {
        setDocuments((prev: Document[]) => [result.document, ...prev]);
      }

      // Reload documents to ensure sync
      await loadDocuments();

    } catch (err) {
      console.error('Upload failed:', err);
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handlePreview = (doc: Document) => {
    setSelectedDocument(doc);
    setViewerOpen(true);
  };

  const handleDownload = (doc: Document) => {
    // Create a link and trigger download
    const link = document.createElement('a');
    link.href = doc.url;
    link.download = doc.name;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getFileIcon = (type: string | undefined) => {
    if (!type) return <File className="h-4 w-4" />;
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
                        {formatFileSize(doc.size)} • Uploaded {new Date(doc.uploadedAt).toLocaleDateString()}
                      </div>
                      {doc.extractedText && (
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
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handlePreview(doc)}
                        title="Preview document"
                      >
                        <Eye className="h-3 w-3" />
                      </Button>
                      <Button 
                        variant="outline" 
                        size="sm"
                        onClick={() => handleDownload(doc)}
                        title="Download document"
                      >
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

      {/* Document Viewer Modal */}
      <Dialog open={viewerOpen} onOpenChange={setViewerOpen}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between">
              <span>{selectedDocument?.name}</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setViewerOpen(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </DialogTitle>
          </DialogHeader>

          {selectedDocument && (
            <div className="space-y-4">
              <div className="bg-gray-100 rounded-lg p-4 flex items-center justify-center min-h-[400px] max-h-[500px] overflow-auto">
                {selectedDocument.type?.startsWith('image/') ? (
                  <img 
                    src={selectedDocument.url} 
                    alt={selectedDocument.name}
                    className="max-w-full max-h-full object-contain"
                  />
                ) : selectedDocument.type === 'application/pdf' ? (
                  <div className="text-center">
                    <FileText className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-600">PDF Viewer</p>
                    <p className="text-sm text-gray-500 mt-2">{selectedDocument.name}</p>
                    <Button 
                      variant="outline" 
                      className="mt-4"
                      onClick={() => handleDownload(selectedDocument)}
                    >
                      Download to View
                    </Button>
                  </div>
                ) : (
                  <div className="text-center">
                    <File className="h-16 w-16 mx-auto text-gray-400 mb-4" />
                    <p className="text-gray-600">Document Preview</p>
                    <p className="text-sm text-gray-500 mt-2">{selectedDocument.name}</p>
                    <Button 
                      variant="outline" 
                      className="mt-4"
                      onClick={() => handleDownload(selectedDocument)}
                    >
                      Download File
                    </Button>
                  </div>
                )}
              </div>

              {selectedDocument.extractedText && (
                <div className="space-y-2">
                  <h3 className="font-semibold text-sm">Extracted Text</h3>
                  <div className="bg-gray-50 rounded p-3 max-h-[200px] overflow-auto text-sm">
                    <p className="whitespace-pre-wrap break-words">{selectedDocument.extractedText}</p>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-600">File Size</p>
                  <p className="font-medium">{formatFileSize(selectedDocument.size)}</p>
                </div>
                <div>
                  <p className="text-gray-600">Uploaded</p>
                  <p className="font-medium">{new Date(selectedDocument.uploadedAt).toLocaleString()}</p>
                </div>
                {selectedDocument.confidence && (
                  <div>
                    <p className="text-gray-600">OCR Confidence</p>
                    <p className="font-medium">{selectedDocument.confidence}%</p>
                  </div>
                )}
                <div>
                  <p className="text-gray-600">File Type</p>
                  <p className="font-medium">{selectedDocument.type || 'Unknown'}</p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}