/**
 * API Client for Quote Generation
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
 * Generate a quote from analysis data
 */
export async function generateQuote(request: GenerateQuoteRequest): Promise<GenerateQuoteResponse> {
  const PYTHON_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || 'http://localhost:8000';
  
  const response = await fetch(`${PYTHON_SERVICE_URL}/api/v1/quote/generate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': '69420'
    },
    body: JSON.stringify(request)
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Quote generation failed' }));
    throw new Error(error.detail || 'Failed to generate quote');
  }
  
  return response.json();
}

/**
 * Export quote to CSV format
 */
export function exportQuoteToCSV(quote: Quote, projectId: string): void {
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
  
  const csvContent = rows.map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
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
