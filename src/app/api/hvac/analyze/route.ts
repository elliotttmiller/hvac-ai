import { NextRequest, NextResponse } from 'next/server';

const PYTHON_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const projectId = formData.get('projectId') as string || 'default';
    const location = formData.get('location') as string || '';

    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }

    // Forward request to Python service
    const pythonFormData = new FormData();
    pythonFormData.append('file', file);
    
    const queryParams = new URLSearchParams({
      project_id: projectId,
      ...(location && { location })
    });

    const response = await fetch(
      `${PYTHON_SERVICE_URL}/api/analyze/blueprint?${queryParams}`,
      {
        method: 'POST',
        body: pythonFormData,
      }
    );

    // Read response safely: some endpoints (or ngrok) may return HTML on error
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');

    if (!response.ok) {
      const raw = isJson ? await response.json().catch(() => null) : await response.text().catch(() => null);
      const message = raw && typeof raw === 'object' ? (raw.detail || JSON.stringify(raw)) : String(raw || 'Analysis failed');
      console.error('Python service error', response.status, message);
      return NextResponse.json(
        { error: message },
        { status: response.status }
      );
    }

    const data = isJson ? await response.json() : await response.text();
    // If we got text back (HTML), include it under `raw` so frontend can show diagnostic info
    return NextResponse.json(
      isJson ? data : { raw: data }
    );

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
    const analysisId = searchParams.get('analysisId');

    if (!analysisId) {
      return NextResponse.json(
        { error: 'Analysis ID required' },
        { status: 400 }
      );
    }

    const response = await fetch(
      `${PYTHON_SERVICE_URL}/api/analyze/${analysisId}`
    );

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || 'Analysis not found' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Get analysis error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
