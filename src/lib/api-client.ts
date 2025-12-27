/**
 * API Client for Quote Generation
 * 
 * NOTE: Quote generation is now integrated directly into the /api/hvac/analyze endpoint.
 * The analyze endpoint returns a response with structure:
 * {
 *   detections: [...],
 *   quote: { ... },    // Pricing is embedded in the response
 *   image_shape: [...]
 * }
 * 
 * The quote is automatically extracted and stored in the pricing-store.
 */

import type { Quote, QuoteSettings } from './pricing-store';

export interface GenerateQuoteRequest {
  project_id: string;
  location: string;
  analysis_data: {
    total_objects: number;
    counts_by_category: Record<string, number>;
  };
  settings?: {
    margin_percent?: number;
    tax_rate?: number;
    labor_hourly_rate?: number;
  };
}

export interface GenerateQuoteResponse {
  quote_id: string;
  currency: string;
  summary: {
    subtotal_materials: number;
    subtotal_labor: number;
    total_cost: number;
    final_price: number;
  };
  line_items: Array<{
    category: string;
    count: number;
    unit_material_cost: number;
    unit_labor_hours: number;
    total_line_cost: number;
    sku_name?: string;
    unit?: string;
  }>;
}

/**
 * @deprecated Quote generation is now integrated into /api/hvac/analyze endpoint.
 * The quote is returned as part of the analyze response under the 'quote' field.
 * This function is kept for backward compatibility but should not be used.
 * 
 * Use the quote from the analyze response instead:
 * const response = await fetch('/api/hvac/analyze', ...)
 * const quote = response.quote;  // Extract quote from response
 */
export async function generateQuote(request: GenerateQuoteRequest): Promise<GenerateQuoteResponse> {
  throw new Error(
    'generateQuote() is deprecated. ' +
    'Quote generation is now integrated into /api/hvac/analyze endpoint. ' +
    'Extract the quote from the analyze response instead: response.quote'
  );
}

/**
 * Check whether the pricing subsystem is available on the backend.
 * Returns { available: boolean, reason?: string }
 */
export async function checkPricingAvailable(): Promise<{ available: boolean; reason?: string }> {
  const PYTHON_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8000';
  try {
    // Check the health endpoint which includes pricing_enabled status
    const res = await fetch(`${PYTHON_SERVICE_URL}/health`, {
      method: 'GET',
      headers: { 'ngrok-skip-browser-warning': '69420' }
    });

    if (!res.ok) {
      return { available: false, reason: `HTTP ${res.status}` };
    }

    const json = await res.json().catch(() => ({ pricing_enabled: false }));
    return { available: !!json.pricing_enabled, reason: json.reason };
  } catch (err) {
    return { available: false, reason: String(err) };
  }
}

/**
 * Export quote to CSV format with proper RFC 4180 compliant escaping
 */
export function exportQuoteToCSV(quote: Quote, projectId: string): void {
  // Helper to escape CSV cells
  const escapeCell = (cell: string | number): string => {
    const str = String(cell);
    // If cell contains comma, quote, or newline, wrap in quotes and escape internal quotes
    if (str.includes(',') || str.includes('"') || str.includes('\n')) {
      return `"${str.replace(/"/g, '""')}"`;
    }
    return str;
  };
  
  const rows = [
    ['Project ID', projectId],
    ['Quote ID', quote.quote_id],
    ['Currency', quote.currency],
    [''],
    ['Category', 'SKU Name', 'Quantity', 'Unit', 'Material Cost/Unit', 'Labor Hours/Unit', 'Total Cost'],
    ...quote.line_items.map(item => [
      item.category,
      item.sku_name || '-',
      item.count.toString(),
      item.unit || 'each',
      `$${item.unit_material_cost.toFixed(2)}`,
      item.unit_labor_hours.toFixed(2),
      `$${item.total_line_cost.toFixed(2)}`
    ]),
    [''],
    ['Summary', '', '', '', '', '', ''],
    ['Subtotal Materials', '', '', '', '', '', `$${quote.summary.subtotal_materials.toFixed(2)}`],
    ['Subtotal Labor', '', '', '', '', '', `$${quote.summary.subtotal_labor.toFixed(2)}`],
    ['Total Cost', '', '', '', '', '', `$${quote.summary.total_cost.toFixed(2)}`],
    ['Final Price (with margin)', '', '', '', '', '', `$${quote.summary.final_price.toFixed(2)}`]
  ];
  
  const csvContent = rows.map(row => row.map(escapeCell).join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);
  
  link.setAttribute('href', url);
  link.setAttribute('download', `quote-${quote.quote_id}-${Date.now()}.csv`);
  link.style.visibility = 'hidden';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
