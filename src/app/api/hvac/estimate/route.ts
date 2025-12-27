import { NextRequest, NextResponse } from 'next/server';

/**
 * DEPRECATED: Estimation is now integrated with the /api/hvac/analyze endpoint.
 * The analyze endpoint returns a 'quote' field in the response containing the pricing.
 * 
 * This route is maintained for backward compatibility but returns a note to use analyze endpoint.
 */

export async function POST(request: NextRequest) {
  return NextResponse.json(
    {
      error: 'Deprecated endpoint',
      detail: 'Estimation is now integrated with /api/hvac/analyze. Call that endpoint instead and extract the "quote" field from the response.',
      example: 'POST /api/hvac/analyze with your image → response includes "quote" with pricing'
    },
    { status: 410 } // 410 Gone - the resource is no longer available
  );
}

export async function GET(request: NextRequest) {
  return NextResponse.json(
    {
      error: 'Deprecated endpoint',
      detail: 'Estimation is now integrated with /api/hvac/analyze. Use the /api/hvac/analyze endpoint with a file upload.'
    },
    { status: 410 } // 410 Gone
  );
}
