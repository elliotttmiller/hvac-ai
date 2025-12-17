import { NextRequest, NextResponse } from 'next/server';

const PYTHON_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    // Accept flexible form fields from the frontend and translate to the
    // Python service's API. Frontend callers may POST to this Next.js route
    // with either 'image' (used by the client) or 'file' (legacy). We map:
    // - click/segment requests (contain 'coords' or 'coords') -> /api/v1/segment
    // - full-image count requests -> /api/v1/count

    const formData = await request.formData();
    const incomingFile = (formData.get('file') || formData.get('image')) as File | null;
    const coords = formData.get('coords') as string | null;
    const prompt = formData.get('prompt') as string | null;
    const returnTopK = formData.get('return_top_k') as string | null;

    if (!incomingFile) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Build python form data
    const pythonFormData = new FormData();
    // Python service expects the file under the 'image' field
    pythonFormData.append('image', incomingFile);

    let targetUrl = '';

    if (coords || prompt) {
      // Segment request — backend expects 'coords' as a string "x,y"
      if (coords) {
        pythonFormData.append('coords', coords as string);
      } else if (prompt) {
        // If the frontend sent a prompt that includes coords, try to extract
        try {
          const promptObj = JSON.parse(prompt as string);
          if (promptObj && promptObj.coords) {
            // support either {coords: [x,y]} or {coords: {x,y}}
            if (Array.isArray(promptObj.coords)) {
              pythonFormData.append('coords', `${promptObj.coords[0]},${promptObj.coords[1]}`);
            } else if (typeof promptObj.coords === 'object' && 'x' in promptObj.coords && 'y' in promptObj.coords) {
              pythonFormData.append('coords', `${promptObj.coords.x},${promptObj.coords.y}`);
            }
          } else {
            // As a fallback, forward the whole prompt under 'prompt' in case backend handles it
            pythonFormData.append('prompt', JSON.stringify(promptObj));
          }
        } catch (e) {
          // If prompt isn't JSON, forward as-is
          pythonFormData.append('prompt', prompt as string);
        }
      }

      if (returnTopK) pythonFormData.append('return_top_k', returnTopK as string);

      targetUrl = `${PYTHON_SERVICE_URL}/api/v1/segment`;
    } else {
      // Count request — backend expects optional 'grid_size' form field
      const gridSize = formData.get('grid_size') as string | null;
      if (gridSize) pythonFormData.append('grid_size', gridSize);
      targetUrl = `${PYTHON_SERVICE_URL}/api/v1/count`;
    }

  const response = await fetch(targetUrl, { method: 'POST', body: pythonFormData, headers: { 'ngrok-skip-browser-warning': '69420' } });

    // Read response safely: some endpoints (or ngrok) may return HTML on error
    // Read response safely: some endpoints (or ngrok) may return HTML on error
    const contentType = response.headers.get('content-type') || '';
    const isJson = contentType.includes('application/json');

    if (!response.ok) {
      const raw = isJson ? await response.json().catch(() => null) : await response.text().catch(() => null);
      const message = raw && typeof raw === 'object' ? (raw.detail || JSON.stringify(raw)) : String(raw || 'Analysis failed');
      console.error('Python service error', response.status, message);
      return NextResponse.json({ error: message }, { status: response.status });
    }

    const data = isJson ? await response.json() : await response.text();
    return NextResponse.json(isJson ? data : { raw: data });

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
      `${PYTHON_SERVICE_URL}/api/analyze/${analysisId}`,
      { headers: { 'ngrok-skip-browser-warning': '69420' } }
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
