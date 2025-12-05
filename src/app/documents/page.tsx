"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Upload,
  FileText,
  Image,
  Eye,
  Download,
  Trash2,
  Search,
  Filter,
  CheckCircle2,
  Clock,
  AlertTriangle,
  File,
  FileImage,
  FileSpreadsheet,
  RotateCcw,
  Zap
} from "lucide-react";
import FileUpload from "@/components/documents/FileUpload";

interface Document {
  id: string;
  name: string;
  type: 'dwg' | 'dxf' | 'pdf' | 'csv' | 'xlsx' | 'image';
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  size: string;
  uploadDate: Date;
  processedDate?: Date;
  category?: string;
  extractedText?: number;
  confidence?: number;
}

const mockDocuments: Document[] = [
  {
    id: '1',
    name: 'architectural-floor-plans.dwg',
    type: 'dwg',
    status: 'completed',
    size: '2.4 MB',
    uploadDate: new Date(Date.now() - 3600000),
    processedDate: new Date(Date.now() - 3400000),
    category: 'Architectural Drawings',
    extractedText: 45,
    confidence: 94
  },
  {
    id: '2',
    name: 'structural-details.pdf',
    type: 'pdf',
    status: 'processing',
    size: '1.8 MB',
    uploadDate: new Date(Date.now() - 1800000),
    category: 'Structural Drawings',
    extractedText: 0,
    confidence: 0
  },
  {
    id: '3',
    name: 'project-specs.xlsx',
    type: 'xlsx',
    status: 'completed',
    size: '456 KB',
    uploadDate: new Date(Date.now() - 7200000),
    processedDate: new Date(Date.now() - 7000000),
    category: 'Project Specifications',
    extractedText: 128,
    confidence: 98
  },
  {
    id: '4',
    name: 'site-photos.zip',
    type: 'image',
    status: 'completed',
    size: '12.3 MB',
    uploadDate: new Date(Date.now() - 10800000),
    processedDate: new Date(Date.now() - 10200000),
    category: 'Site Documentation',
    extractedText: 23,
    confidence: 87
  },
  {
    id: '5',
    name: 'electrical-schematics.dxf',
    type: 'dxf',
    status: 'error',
    size: '3.1 MB',
    uploadDate: new Date(Date.now() - 14400000),
    category: 'Electrical Drawings',
    extractedText: 0,
    confidence: 0
  }
];

const getFileIcon = (type: string) => {
  switch (type) {
    case 'dwg':
    case 'dxf':
      return <File className="h-4 w-4" />;
    case 'pdf':
      return <FileText className="h-4 w-4" />;
    case 'csv':
    case 'xlsx':
      return <FileSpreadsheet className="h-4 w-4" />;
    case 'image':
      return <FileImage className="h-4 w-4" />;
    default:
      return <File className="h-4 w-4" />;
  }
};

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'uploaded':
      return <Badge variant="secondary">Uploaded</Badge>;
    case 'processing':
      return <Badge className="bg-blue-500">Processing</Badge>;
    case 'completed':
      return <Badge className="bg-green-500">Completed</Badge>;
    case 'error':
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="secondary">Unknown</Badge>;
  }
};

export default function DocumentsPage() {
  const [documents] = useState<Document[]>(mockDocuments);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    // Handle file drop logic here
  };

  const uploadedCount = documents.filter(d => d.status === 'uploaded').length;
  const processingCount = documents.filter(d => d.status === 'processing').length;
  const completedCount = documents.filter(d => d.status === 'completed').length;
  const errorCount = documents.filter(d => d.status === 'error').length;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Document Processing Center</h1>
          <p className="text-muted-foreground">
            Upload, process, and manage construction documents with AI
          </p>
        </div>
        <Button>
          <Upload className="mr-2 h-4 w-4" />
          Upload Documents
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{documents.length}</div>
            <p className="text-xs text-muted-foreground">
              Across all projects
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{processingCount}</div>
            <p className="text-xs text-muted-foreground">
              Currently being processed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Completed</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{completedCount}</div>
            <p className="text-xs text-muted-foreground">
              Successfully processed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">OCR Accuracy</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">94.2%</div>
            <p className="text-xs text-muted-foreground">
              Average extraction accuracy
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Upload Area */}
      <FileUpload
        projectId="default-project"
        onUploadComplete={(documents) => {
          console.log('Upload completed:', documents);
          // In a real app, this would refresh the document list
        }}
        onUploadError={(error) => {
          console.error('Upload error:', error);
        }}
      />

      {/* Document Management */}
      <Tabs defaultValue="all" className="space-y-4">
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="all">All Documents</TabsTrigger>
            <TabsTrigger value="processing">Processing</TabsTrigger>
            <TabsTrigger value="completed">Completed</TabsTrigger>
            <TabsTrigger value="errors">Errors</TabsTrigger>
          </TabsList>
          <div className="flex space-x-2">
            <Button variant="outline" size="sm">
              <Search className="mr-2 h-4 w-4" />
              Search
            </Button>
            <Button variant="outline" size="sm">
              <Filter className="mr-2 h-4 w-4" />
              Filter
            </Button>
          </div>
        </div>

        <TabsContent value="all" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Document Library</CardTitle>
              <CardDescription>
                All uploaded and processed documents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {documents.map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg hover:bg-muted/50">
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-primary/10 rounded-lg">
                        {getFileIcon(doc.type)}
                      </div>
                      <div>
                        <h3 className="font-medium">{doc.name}</h3>
                        <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                          <span>{doc.size}</span>
                          <span>•</span>
                          <span>Uploaded {doc.uploadDate.toLocaleString()}</span>
                          {doc.category && (
                            <>
                              <span>•</span>
                              <span>{doc.category}</span>
                            </>
                          )}
                        </div>
                        {doc.status === 'processing' && (
                          <Progress value={65} className="w-32 h-2 mt-2" />
                        )}
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      {doc.status === 'completed' && doc.confidence && (
                        <div className="text-right">
                          <p className="text-sm font-medium">{doc.confidence}%</p>
                          <p className="text-xs text-muted-foreground">Confidence</p>
                        </div>
                      )}
                      {doc.extractedText && (
                        <div className="text-right">
                          <p className="text-sm font-medium">{doc.extractedText}</p>
                          <p className="text-xs text-muted-foreground">Text blocks</p>
                        </div>
                      )}
                      <div className="flex flex-col space-y-1">
                        {getStatusBadge(doc.status)}
                      </div>
                      <div className="flex space-x-1">
                        <Button size="sm" variant="outline">
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button size="sm" variant="outline">
                          <Download className="h-3 w-3" />
                        </Button>
                        {doc.status === 'error' && (
                          <Button size="sm" variant="outline">
                            <RotateCcw className="h-3 w-3" />
                          </Button>
                        )}
                        <Button size="sm" variant="outline" className="text-destructive">
                          <Trash2 className="h-3 w-3" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="processing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Documents Being Processed</CardTitle>
              <CardDescription>
                Files currently being analyzed by AI agents
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {documents.filter(doc => doc.status === 'processing').map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                        {getFileIcon(doc.type)}
                      </div>
                      <div>
                        <h3 className="font-medium">{doc.name}</h3>
                        <p className="text-sm text-muted-foreground">OCR extraction in progress...</p>
                        <Progress value={65} className="w-48 h-2 mt-2" />
                      </div>
                    </div>
                    <Badge className="bg-blue-500">Processing</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="completed" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Completed Documents</CardTitle>
              <CardDescription>
                Successfully processed documents ready for use
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {documents.filter(doc => doc.status === 'completed').map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                        {getFileIcon(doc.type)}
                      </div>
                      <div>
                        <h3 className="font-medium">{doc.name}</h3>
                        <p className="text-sm text-muted-foreground">
                          Processed {doc.processedDate?.toLocaleString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className="text-right">
                        <p className="text-sm font-medium">{doc.confidence}%</p>
                        <p className="text-xs text-muted-foreground">Accuracy</p>
                      </div>
                      <Badge className="bg-green-500">Completed</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="errors" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Processing Errors</CardTitle>
              <CardDescription>
                Documents that encountered issues during processing
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {documents.filter(doc => doc.status === 'error').map((doc) => (
                  <div key={doc.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-red-100 dark:bg-red-900 rounded-lg">
                        <AlertTriangle className="h-4 w-4 text-red-500" />
                      </div>
                      <div>
                        <h3 className="font-medium">{doc.name}</h3>
                        <p className="text-sm text-red-600">
                          Processing failed - file format not supported
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button size="sm" variant="outline">
                        <RotateCcw className="mr-2 h-3 w-3" />
                        Retry
                      </Button>
                      <Badge variant="destructive">Error</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
