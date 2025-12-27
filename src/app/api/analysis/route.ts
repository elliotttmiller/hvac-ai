import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/local-db';

const PYTHON_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const incomingFile = (formData.get('file') || formData.get('image')) as File | null;
    const projectId = formData.get('projectId') as string | null;
    const category = (formData.get('category') as string | null) || 'blueprint';
    const coords = formData.get('coords') as string | null;
    const prompt = formData.get('prompt') as string | null;
    const returnTopK = formData.get('return_top_k') as string | null;

    if (!incomingFile) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    if (!projectId) {
      return NextResponse.json({ error: 'No projectId provided' }, { status: 400 });
    }
    
    // Validate file size (max 500MB)
    if (incomingFile.size > 500 * 1024 * 1024) {
      return NextResponse.json({ error: 'File size exceeds 500MB limit' }, { status: 400 });
    }

    // Create document entry with 'processing' status
    const document = db.createDocument(projectId, {
      name: incomingFile.name,
      type: incomingFile.type || 'application/octet-stream',
      size: incomingFile.size,
      url: `/api/analysis/${Date.now()}/${incomingFile.name}`,
    });

    // Build python form data for analysis
    const pythonFormData = new FormData();
    pythonFormData.append('image', incomingFile);

    let targetUrl = '';

    // Detect whether the client requested a streaming response (SSE).
    const { searchParams } = new URL(request.url);
    const wantsStream = searchParams.get('stream') === '1' || (request.headers.get('accept') || '').includes('text/event-stream');

    if (coords || prompt) {
      // Segment request
      if (coords) {
        pythonFormData.append('coords', coords as string);
      } else if (prompt) {
        try {
          const promptObj = JSON.parse(prompt as string);
          if (promptObj && promptObj.coords) {
            if (Array.isArray(promptObj.coords)) {
              pythonFormData.append('coords', `${promptObj.coords[0]},${promptObj.coords[1]}`);
            } else if (typeof promptObj.coords === 'object' && 'x' in promptObj.coords && 'y' in promptObj.coords) {
              pythonFormData.append('coords', `${promptObj.coords.x},${promptObj.coords.y}`);
            }
          } else {
            pythonFormData.append('prompt', JSON.stringify(promptObj));
          }
        } catch (e) {
          pythonFormData.append('prompt', prompt as string);
        }
      }
      if (returnTopK) pythonFormData.append('return_top_k', returnTopK as string);
      targetUrl = `${PYTHON_SERVICE_URL}/api/hvac/analyze`;
    } else {
      // Count request
      const gridSize = formData.get('grid_size') as string | null;
      if (gridSize) pythonFormData.append('grid_size', gridSize);
      targetUrl = `${PYTHON_SERVICE_URL}/api/hvac/analyze`;
    }

    // If streaming is requested, return the stream directly
    if (wantsStream) {
      try {
        const upstream = await fetch(
          `${PYTHON_SERVICE_URL}/api/hvac/analyze?stream=1`,
          {
            method: 'POST',
            body: pythonFormData,
            headers: { 'ngrok-skip-browser-warning': '69420' },
          }
        );
        
        if (!upstream.ok) {
          db.updateDocument(document.id, { status: 'error' });
          const errorText = await upstream.text().catch(() => 'Upstream service error');
          return NextResponse.json(
            { error: errorText }, 
            { status: upstream.status }
          );
        }
        
        // Return stream with document info
        return new Response(upstream.body, { 
          status: upstream.status, 
          headers: { 
            'content-type': upstream.headers.get('content-type') || 'text/event-stream',
            'cache-control': 'no-cache',
            'connection': 'keep-alive',
          } 
        });
      } catch (error) {
        db.updateDocument(document.id, { status: 'error' });
        console.error('Streaming proxy error:', error);
        return NextResponse.json(
          { error: 'Failed to connect to analysis service' }, 
          { status: 503 }
        );
      }
    }

    // Non-streaming: send to analysis service
    const response = await fetch(targetUrl, {
      method: 'POST',
      body: pythonFormData,
      headers: { 'ngrok-skip-browser-warning': '69420' }
    });

    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');

    if (!response.ok) {
      const raw = isJson ? await response.json().catch(() => null) : await response.text().catch(() => null);
      const message = raw && typeof raw === 'object' ? (raw.detail || JSON.stringify(raw)) : String(raw || 'Analysis failed');
      console.error('Python service error', response.status, message);
      db.updateDocument(document.id, { status: 'error' });
      return NextResponse.json({ error: message }, { status: response.status });
    }

    const analysisData = isJson ? await response.json() : await response.text();
    
    // Update document with analysis results
    const extractedText = analysisData?.extracted_text || analysisData?.raw || '';
    const confidence = analysisData?.confidence || undefined;
    
    const updatedDoc = db.updateDocument(document.id, {
      status: 'completed',
      extractedText,
      confidence,
    });

    return NextResponse.json({ 
      document: updatedDoc,
      analysis: analysisData 
    });

  } catch (error) {
    console.error('Blueprint analysis error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const projectId = searchParams.get('projectId');

    if (!projectId) {
      return NextResponse.json({ error: 'Missing projectId' }, { status: 400 });
    }

    // Return only the documents actually uploaded for this project
    const documents = db.getDocumentsByProjectId(projectId);

    return NextResponse.json({
      projectId,
      documents,
    });
  } catch (error) {
    console.error('Analysis GET error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

