import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// In-memory document store for development/demo
type Document = {
  id: string;
  name: string;
  type: string;
  status: 'uploaded' | 'processing' | 'completed' | 'error';
  size: number;
  url: string;
  project_id: string;
  uploaded_by: string | null;
  category: string;
  extracted_text?: string;
  confidence?: number;
  created_at: string;
  updated_at: string;
};

const documents: Document[] = [];

export async function POST(request: NextRequest) {
  try {
    // For demo purposes, we'll simulate user authentication
    const userId = null;

    const formData = await request.formData();
    const file = formData.get('file') as File;
    const projectId = formData.get('projectId') as string;
    const category = formData.get('category') as string;

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Validate file size (500MB limit)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      return NextResponse.json(
        { error: 'File too large. Maximum size is 500MB.' },
        { status: 400 }
      );
    }

    // Validate file type
    const allowedTypes = [
      'application/pdf',
      'image/jpeg',
      'image/png',
      'image/tiff',
      'application/dwg',
      'application/dxf',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/csv'
    ];

    const fileType = file.type || getFileTypeFromExtension(file.name);
    if (!allowedTypes.includes(fileType) && !isCADFile(file.name)) {
      return NextResponse.json(
        { error: 'Unsupported file type' },
        { status: 400 }
      );
    }

    // Generate unique filename
    const fileId = uuidv4();
    const fileExtension = path.extname(file.name);
    const fileName = `${fileId}${fileExtension}`;
    const uploadDir = path.join(process.cwd(), 'uploads');
    const filePath = path.join(uploadDir, fileName);

    // Ensure upload directory exists
    try {
      await mkdir(uploadDir, { recursive: true });
    } catch (error) {
      // Directory might already exist
    }

    // Save file to disk
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    await writeFile(filePath, buffer);

    // Create document record in memory
    const document: Document = {
      id: fileId,
      name: file.name,
      type: fileType,
      status: 'uploaded',
      size: file.size,
      url: `/uploads/${fileName}`,
      project_id: projectId,
      uploaded_by: userId,
      category: category || 'Uncategorized',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    documents.push(document);

    // Start OCR processing for supported files (simplified for demo)
    if (isImageFile(file.name) || file.type === 'application/pdf') {
      // Simulate OCR processing
      setTimeout(() => {
        const docIndex = documents.findIndex(d => d.id === fileId);
        if (docIndex !== -1) {
          documents[docIndex].status = 'completed';
          documents[docIndex].extracted_text = `Sample extracted text from ${file.name}`;
          documents[docIndex].confidence = 85;
          documents[docIndex].updated_at = new Date().toISOString();
        }
      }, 2000); // Simulate 2 second processing time
    } else {
      // Mark as completed for non-OCR files
      const docIndex = documents.findIndex(d => d.id === fileId);
      if (docIndex !== -1) {
        documents[docIndex].status = 'completed';
        documents[docIndex].updated_at = new Date().toISOString();
      }
    }

    return NextResponse.json({
      success: true,
      document: {
        id: document.id,
        name: document.name,
        type: document.type,
        status: document.status,
        size: document.size,
        url: document.url,
        category: document.category,
        uploadedAt: document.created_at
      }
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// GET endpoint to retrieve documents
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');
    const status = searchParams.get('status');

    let filteredDocuments = documents;

    if (projectId) {
      filteredDocuments = filteredDocuments.filter(doc => doc.project_id === projectId);
    }

    if (status) {
      filteredDocuments = filteredDocuments.filter(doc => doc.status === status);
    }

    // Sort by created_at descending
    filteredDocuments.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

    return NextResponse.json({
      success: true,
      documents: filteredDocuments
    });

  } catch (error) {
    console.error('Fetch documents error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Helper functions
function getFileTypeFromExtension(filename: string): string {
  const ext = path.extname(filename).toLowerCase();
  const typeMap: { [key: string]: string } = {
    '.pdf': 'application/pdf',
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.tiff': 'image/tiff',
    '.tif': 'image/tiff',
    '.dwg': 'application/dwg',
    '.dxf': 'application/dxf',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.csv': 'text/csv'
  };
  return typeMap[ext] || 'application/octet-stream';
}

function isCADFile(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  return ['.dwg', '.dxf'].includes(ext);
}

function isImageFile(filename: string): boolean {
  const ext = path.extname(filename).toLowerCase();
  return ['.jpg', '.jpeg', '.png', '.tiff', '.tif'].includes(ext);
}

