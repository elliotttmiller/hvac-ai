'use client';

/**
 * AnnotationSidebar Component
 * Virtualized list for displaying and filtering annotations
 */

import React, { useMemo, useCallback, useState } from 'react';
import type { EditableAnnotation } from '@/types/deep-zoom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  Filter,
  Search,
  Trash2,
  Edit3,
  CheckCircle2,
  AlertCircle,
  Download,
} from 'lucide-react';

interface AnnotationSidebarProps {
  annotations: Map<string, EditableAnnotation>;
  filteredAnnotations: EditableAnnotation[];
  selectedId: string | null;
  confidenceThreshold: number;
  onConfidenceChange: (value: number) => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onReclassify: (id: string, newLabel: string) => void;
  hasUnsavedChanges: boolean;
  onSave: () => void;
}

// Class colors
const CLASS_COLORS: Record<string, string> = {
  valve: '#ef4444',
  instrument: '#3b82f6',
  sensor: '#10b981',
  duct: '#f59e0b',
  vent: '#8b5cf6',
  default: '#6b7280',
};

function getColorForLabel(label: string): string {
  const lower = label.toLowerCase();
  if (lower.includes('valve')) return CLASS_COLORS.valve;
  if (lower.includes('instrument')) return CLASS_COLORS.instrument;
  if (lower.includes('sensor')) return CLASS_COLORS.sensor;
  if (lower.includes('duct')) return CLASS_COLORS.duct;
  if (lower.includes('vent')) return CLASS_COLORS.vent;
  return CLASS_COLORS.default;
}

export default function AnnotationSidebar({
  annotations,
  filteredAnnotations,
  selectedId,
  confidenceThreshold,
  onConfidenceChange,
  onSelect,
  onDelete,
  onReclassify,
  hasUnsavedChanges,
  onSave,
}: AnnotationSidebarProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'confidence' | 'label' | 'position'>('confidence');

  // Filter and sort annotations
  const displayedAnnotations = useMemo(() => {
    let filtered = filteredAnnotations;

    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((ann) => ann.label.toLowerCase().includes(query));
    }

    // Sort
    const sorted = [...filtered];
    switch (sortBy) {
      case 'confidence':
        sorted.sort((a, b) => b.score - a.score);
        break;
      case 'label':
        sorted.sort((a, b) => a.label.localeCompare(b.label));
        break;
      case 'position':
        sorted.sort((a, b) => {
          const [ax, ay] = a.bbox;
          const [bx, by] = b.bbox;
          return ay - by || ax - bx; // top to bottom, left to right
        });
        break;
    }

    return sorted;
  }, [filteredAnnotations, searchQuery, sortBy]);

  // Compute category counts
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    filteredAnnotations.forEach((ann) => {
      counts[ann.label] = (counts[ann.label] || 0) + 1;
    });
    return counts;
  }, [filteredAnnotations]);

  // Row renderer for virtualized list
  const Row = useCallback(
    ({ index, style }: { index: number; style: React.CSSProperties }) => {
      const ann = displayedAnnotations[index];
      if (!ann) return null;
      
      const isSelected = ann.id === selectedId;
      const color = getColorForLabel(ann.label);

      return (
        <div
          style={style}
          className={`px-3 py-2 border-b border-slate-200 cursor-pointer transition-colors ${
            isSelected ? 'bg-blue-50 border-blue-300' : 'hover:bg-slate-50'
          }`}
          onClick={() => onSelect(ann.id)}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <div
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: color }}
                />
                <span className="font-medium text-sm truncate">{ann.label}</span>
                {ann.isDirty && (
                  <AlertCircle className="h-3 w-3 text-orange-500 flex-shrink-0" />
                )}
                {ann.isNew && (
                  <Badge variant="secondary" className="text-xs">
                    NEW
                  </Badge>
                )}
              </div>
              <div className="text-xs text-slate-500">
                Confidence: {Math.round(ann.score * 100)}%
              </div>
            </div>
            <div className="flex gap-1">
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={(e) => {
                  e.stopPropagation();
                  const newLabel = prompt('Enter new label:', ann.label);
                  if (newLabel) {
                    onReclassify(ann.id, newLabel);
                  }
                }}
              >
                <Edit3 className="h-3 w-3" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-red-600 hover:text-red-700"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Delete annotation "${ann.label}"?`)) {
                    onDelete(ann.id);
                  }
                }}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </div>
      );
    },
    [displayedAnnotations, selectedId, onSelect, onReclassify, onDelete]
  );

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between mb-4">
          <CardTitle className="text-lg">Detections</CardTitle>
          <Badge variant="secondary">{filteredAnnotations.length} items</Badge>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-2 top-2.5 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search annotations..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-8"
          />
        </div>

        {/* Confidence Threshold Slider */}
        <div className="space-y-2 mt-4">
          <div className="flex items-center justify-between">
            <Label className="text-sm">Confidence Threshold</Label>
            <span className="text-sm font-medium">{Math.round(confidenceThreshold * 100)}%</span>
          </div>
          <Slider
            value={[confidenceThreshold * 100]}
            onValueChange={([value]) => onConfidenceChange(value / 100)}
            min={0}
            max={100}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-slate-500">
            Showing {filteredAnnotations.length} of {annotations.size} detections
          </p>
        </div>

        {/* Sort Options */}
        <div className="flex gap-2 mt-4">
          <Button
            variant={sortBy === 'confidence' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSortBy('confidence')}
            className="flex-1"
          >
            Confidence
          </Button>
          <Button
            variant={sortBy === 'label' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSortBy('label')}
            className="flex-1"
          >
            Label
          </Button>
          <Button
            variant={sortBy === 'position' ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSortBy('position')}
            className="flex-1"
          >
            Position
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0 overflow-hidden">
        {/* Category Summary */}
        <div className="px-3 py-2 bg-slate-50 border-y border-slate-200">
          <div className="text-xs font-semibold text-slate-600 mb-2">Categories</div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(categoryCounts)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 5)
              .map(([label, count]) => (
                <Badge
                  key={label}
                  variant="secondary"
                  className="text-xs"
                  style={{
                    backgroundColor: getColorForLabel(label) + '20',
                    color: getColorForLabel(label),
                    borderColor: getColorForLabel(label),
                  }}
                >
                  {label}: {count}
                </Badge>
              ))}
          </div>
        </div>

        {/* Simple Scrollable List (virtualization for large datasets) */}
        <div className="flex-1 h-full overflow-y-auto max-h-[500px]">
          {displayedAnnotations.length > 0 ? (
            <div>
              {displayedAnnotations.map((ann, index) => (
                <Row key={ann.id} index={index} style={{}} />
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-slate-400">
              <div className="text-center">
                <Filter className="h-12 w-12 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No annotations match filters</p>
              </div>
            </div>
          )}
        </div>

        {/* Save Button */}
        {hasUnsavedChanges && (
          <div className="px-3 py-3 border-t border-slate-200 bg-orange-50">
            <Button onClick={onSave} className="w-full" size="sm">
              <CheckCircle2 className="mr-2 h-4 w-4" />
              Save Changes
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
