'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Download,
  DollarSign,
  TrendingUp,
  Percent,
  Edit3,
  Check,
  X
} from 'lucide-react';

export interface QuoteData {
  quote_id?: string;
  line_items: Array<{
    category: string;
    sku_name?: string;
    count: number;
    unit_material_cost: number;
    total_line_cost: number;
  }>;
  summary: {
    subtotal_materials: number;
    subtotal_labor: number;
    total_cost: number;
    final_price: number;
  };
}

export interface QuoteSettings {
  margin_percent: number;
  labor_hourly_rate: number;
}

export interface QuoteBuilderProps {
  /** The quote data to display */
  data?: QuoteData | null;
  /** Current settings for calculations */
  settings?: QuoteSettings;
  /** Callback when settings change */
  onSettingsChange?: (settings: Partial<QuoteSettings>) => void;
  /** Callback when a line item is modified */
  onLineItemUpdate?: (category: string, updates: { count?: number; unit_cost?: number }) => void;
  /** Callback when export is requested */
  onExport?: (data: QuoteData) => void;
  /** Callback when category is hovered */
  onCategoryHover?: (category: string | null) => void;
  /** Custom title for the component */
  title?: string;
  /** Custom export button text */
  exportButtonText?: string;
  /** Whether line items are editable */
  editable?: boolean;
}

const DEFAULT_SETTINGS: QuoteSettings = {
  margin_percent: 15,
  labor_hourly_rate: 75,
};

export function QuoteBuilder({
  data,
  settings = DEFAULT_SETTINGS,
  onSettingsChange,
  onLineItemUpdate,
  onExport,
  onCategoryHover,
  title = "Quote",
  exportButtonText = "Export",
  editable = true,
}: QuoteBuilderProps) {
  const [editingCategory, setEditingCategory] = useState<string | null>(null);
  const [editValues, setEditValues] = useState<{ count?: number; unit_cost?: number }>({});

  const handleStartEdit = (category: string, count: number, unitCost: number) => {
    if (!editable) return;
    setEditingCategory(category);
    setEditValues({ count, unit_cost: unitCost });
  };

  const handleSaveEdit = () => {
    if (editingCategory && editValues) {
      onLineItemUpdate?.(editingCategory, editValues);
    }
    setEditingCategory(null);
    setEditValues({});
  };

  const handleCancelEdit = () => {
    setEditingCategory(null);
    setEditValues({});
  };

  const handleExport = () => {
    if (data && onExport) {
      onExport(data);
    }
  };

  if (!data) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          <CardDescription>Generate analysis to view pricing</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <div className="h-full flex flex-col space-y-4 overflow-hidden">
      {/* Header */}
      <Card className="border-slate-200">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl font-bold">{title}</CardTitle>
              {data.quote_id && (
                <CardDescription className="font-mono text-xs mt-1">
                  {data.quote_id}
                </CardDescription>
              )}
            </div>
            {onExport && (
              <Button onClick={handleExport} variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                {exportButtonText}
              </Button>
            )}
          </div>
        </CardHeader>
      </Card>

      {/* Settings Controls */}
      {onSettingsChange && (
        <Card className="border-slate-200">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-xs">Margin: {settings.margin_percent}%</Label>
                <Percent className="h-3 w-3 text-slate-400" />
              </div>
              <Slider
                value={[settings.margin_percent]}
                onValueChange={(value) => onSettingsChange({ margin_percent: value[0] })}
                min={0}
                max={50}
                step={1}
                className="w-full"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-xs">Labor Rate ($/hr)</Label>
              <Input
                type="number"
                value={settings.labor_hourly_rate}
                onChange={(e) => onSettingsChange({ labor_hourly_rate: Number(e.target.value) })}
                className="h-8 text-sm font-mono"
                min={0}
                step={5}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Line Items Table */}
      <Card className="flex-1 overflow-hidden border-slate-200">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Line Items</CardTitle>
        </CardHeader>
        <CardContent className="overflow-y-auto h-full pb-20">
          <div className="space-y-2">
            {data.line_items.map((item) => {
              const isEditing = editingCategory === item.category;

              return (
                <div
                  key={item.category}
                  className="p-3 rounded-lg border border-slate-200 hover:border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
                  onMouseEnter={() => onCategoryHover?.(item.category)}
                  onMouseLeave={() => onCategoryHover?.(null)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium">{item.sku_name || item.category}</p>
                      </div>
                      <p className="text-xs text-slate-500">{item.category}</p>
                    </div>

                    {!isEditing && editable && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0"
                        onClick={() => handleStartEdit(item.category, item.count, item.unit_material_cost)}
                      >
                        <Edit3 className="h-3 w-3" />
                      </Button>
                    )}
                  </div>

                  {isEditing ? (
                    <div className="space-y-2 mt-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <Label className="text-xs">Quantity</Label>
                          <Input
                            type="number"
                            value={editValues.count ?? 0}
                            onChange={(e) => {
                              const count = Math.max(0, parseInt(e.target.value) || 0);
                              setEditValues({ ...editValues, count });
                            }}
                            className="h-8 text-sm font-mono"
                            min={0}
                          />
                        </div>
                        <div>
                          <Label className="text-xs">Material Cost</Label>
                          <Input
                            type="number"
                            value={editValues.unit_cost ?? 0}
                            onChange={(e) => {
                              const unit_cost = Math.max(0, parseFloat(e.target.value) || 0);
                              setEditValues({ ...editValues, unit_cost });
                            }}
                            className="h-8 text-sm font-mono"
                            min={0}
                            step={0.01}
                          />
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button onClick={handleSaveEdit} size="sm" className="h-7 text-xs flex-1">
                          <Check className="h-3 w-3 mr-1" />
                          Save
                        </Button>
                        <Button onClick={handleCancelEdit} size="sm" variant="outline" className="h-7 text-xs flex-1">
                          <X className="h-3 w-3 mr-1" />
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                      <div>
                        <span className="text-slate-500">Qty:</span> {item.count}
                      </div>
                      <div>
                        <span className="text-slate-500">Mat:</span> ${item.unit_material_cost.toFixed(2)}
                      </div>
                      <div className="text-right font-semibold">
                        ${item.total_line_cost.toFixed(2)}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Summary */}
      <Card className="border-slate-200 bg-slate-50">
        <CardContent className="pt-6 space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Materials</span>
            <span className="font-mono">${data.summary.subtotal_materials.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Labor</span>
            <span className="font-mono">${data.summary.subtotal_labor.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-600">Subtotal</span>
            <span className="font-mono">${data.summary.total_cost.toFixed(2)}</span>
          </div>
          <Separator className="my-2" />
          <div className="flex justify-between text-lg font-bold">
            <span className="flex items-center gap-2">
              <DollarSign className="h-5 w-5 text-green-600" />
              Final Price
            </span>
            <span className="font-mono text-green-600">
              ${data.summary.final_price.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-slate-500 text-center">
            Includes {settings.margin_percent}% margin
          </p>
        </CardContent>
      </Card>
    </div>
  );
}