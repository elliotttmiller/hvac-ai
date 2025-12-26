/**
 * Zustand store for quote/pricing state management
 */

import { create } from 'zustand';

export interface LineItem {
  category: string;
  count: number;
  unit_material_cost: number;
  unit_labor_hours: number;
  total_line_cost: number;
  sku_name?: string;
  unit?: string;
}

export interface QuoteSummary {
  subtotal_materials: number;
  subtotal_labor: number;
  total_cost: number;
  final_price: number;
}

export interface Quote {
  quote_id: string;
  currency: string;
  summary: QuoteSummary;
  line_items: LineItem[];
}

export interface QuoteSettings {
  margin_percent: number;
  tax_rate: number;
  labor_hourly_rate: number;
}

interface QuoteStore {
  // State
  quote: Quote | null;
  isGenerating: boolean;
  error: string | null;
  settings: QuoteSettings;
  
  // Local overrides (for client-side recalculation)
  localOverrides: Record<string, { count?: number; unit_cost?: number }>;
  
  // Actions
  setQuote: (quote: Quote) => void;
  setGenerating: (isGenerating: boolean) => void;
  setError: (error: string | null) => void;
  updateSettings: (settings: Partial<QuoteSettings>) => void;
  setLocalOverride: (category: string, override: { count?: number; unit_cost?: number }) => void;
  clearLocalOverrides: () => void;
  getRecalculatedQuote: () => Quote | null;
  reset: () => void;
}

const DEFAULT_SETTINGS: QuoteSettings = {
  margin_percent: 20,
  tax_rate: 0,
  labor_hourly_rate: 85,
};

export const useQuoteStore = create<QuoteStore>((set, get) => ({
  // Initial state
  quote: null,
  isGenerating: false,
  error: null,
  settings: DEFAULT_SETTINGS,
  localOverrides: {},
  
  // Actions
  setQuote: (quote) => set({ quote, error: null }),
  
  setGenerating: (isGenerating) => set({ isGenerating }),
  
  setError: (error) => set({ error, isGenerating: false }),
  
  updateSettings: (newSettings) => set((state) => ({
    settings: { ...state.settings, ...newSettings }
  })),
  
  setLocalOverride: (category, override) => set((state) => ({
    localOverrides: {
      ...state.localOverrides,
      [category]: { ...state.localOverrides[category], ...override }
    }
  })),
  
  clearLocalOverrides: () => set({ localOverrides: {} }),
  
  getRecalculatedQuote: () => {
    const { quote, localOverrides, settings } = get();
    if (!quote) return null;
    
    // Recalculate line items with local overrides
    const updatedLineItems = quote.line_items.map((item) => {
      const override = localOverrides[item.category];
      if (!override) return item;
      
      const count = override.count ?? item.count;
      const unitMaterialCost = override.unit_cost ?? item.unit_material_cost;
      const unitLaborCost = item.unit_labor_hours * settings.labor_hourly_rate;
      const totalLineCost = (unitMaterialCost + unitLaborCost) * count;
      
      return {
        ...item,
        count,
        unit_material_cost: unitMaterialCost,
        total_line_cost: totalLineCost
      };
    });
    
    // Recalculate summary
    const subtotalMaterials = updatedLineItems.reduce(
      (sum, item) => sum + (item.unit_material_cost * item.count), 
      0
    );
    const subtotalLabor = updatedLineItems.reduce(
      (sum, item) => sum + (item.unit_labor_hours * settings.labor_hourly_rate * item.count), 
      0
    );
    const totalCost = subtotalMaterials + subtotalLabor;
    const marginAmount = totalCost * (settings.margin_percent / 100);
    const finalPrice = totalCost + marginAmount;
    
    return {
      ...quote,
      line_items: updatedLineItems,
      summary: {
        subtotal_materials: Number(subtotalMaterials.toFixed(2)),
        subtotal_labor: Number(subtotalLabor.toFixed(2)),
        total_cost: Number(totalCost.toFixed(2)),
        final_price: Number(finalPrice.toFixed(2))
      }
    };
  },
  
  reset: () => set({
    quote: null,
    isGenerating: false,
    error: null,
    settings: DEFAULT_SETTINGS,
    localOverrides: {}
  })
}));
