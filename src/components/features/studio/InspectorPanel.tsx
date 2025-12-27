'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { 
  Info, 
  Ruler, 
  DollarSign, 
  Flag, 
  Download, 
  ExternalLink,
  ChevronDown,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Badge } from '@/components/ui/badge';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';

export interface ComponentDetails {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  rotation?: number;
  ocrText?: string;
  ocrConfidence?: number;
  estimatedCost?: {
    material: number;
    labor: number;
    total: number;
    laborHours: number;
  };
}

interface InspectorPanelProps {
  component: ComponentDetails | null;
  onAction?: (action: string, componentId: string) => void;
  className?: string;
}

export function InspectorPanel({
  component,
  onAction,
  className,
}: InspectorPanelProps) {
  if (!component) {
    return (
      <div className={cn('flex items-center justify-center h-full p-6', className)}>
        <div className="text-center text-muted-foreground">
          <Info className="w-12 h-12 mx-auto mb-3 opacity-20" />
          <p className="text-sm">Select a component to view details</p>
        </div>
      </div>
    );
  }

  const { bbox, rotation } = component;
  const width = bbox[2] - bbox[0];
  const height = bbox[3] - bbox[1];

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="studio-panel-header justify-between">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4" />
          <span className="font-semibold text-sm">Inspector</span>
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="p-4 space-y-4"
        >
          {/* Component Header */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">{component.label}</h3>
            <Badge variant="secondary" className="text-xs">
              {Math.round(component.confidence * 100)}% Confidence
            </Badge>
          </div>

          <Separator />

          {/* Accordion Sections */}
          <Accordion type="multiple" defaultValue={['properties', 'geometry', 'cost']} className="w-full">
            {/* Properties Section */}
            <AccordionItem value="properties">
              <AccordionTrigger className="text-sm font-semibold">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4" />
                  Properties
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3 pt-2">
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Class Label:</span>
                    <span className="col-span-2 font-medium">{component.label}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Confidence:</span>
                    <span className="col-span-2 font-medium">
                      {(component.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Component ID:</span>
                    <span className="col-span-2 font-mono text-xs truncate" title={component.id}>
                      {component.id}
                    </span>
                  </div>
                  {component.ocrText && (
                    <>
                      <Separator />
                      <div className="space-y-2">
                        <span className="text-muted-foreground text-sm">OCR Text:</span>
                        <div className="bg-secondary/50 p-2 rounded text-sm font-mono">
                          {component.ocrText}
                        </div>
                        {component.ocrConfidence && (
                          <span className="text-xs text-muted-foreground">
                            OCR Confidence: {(component.ocrConfidence * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Geometry Section */}
            <AccordionItem value="geometry">
              <AccordionTrigger className="text-sm font-semibold">
                <div className="flex items-center gap-2">
                  <Ruler className="w-4 h-4" />
                  Geometry
                </div>
              </AccordionTrigger>
              <AccordionContent>
                <div className="space-y-3 pt-2">
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">X:</span>
                    <span className="col-span-2 font-medium">{bbox[0].toFixed(2)}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Y:</span>
                    <span className="col-span-2 font-medium">{bbox[1].toFixed(2)}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Width:</span>
                    <span className="col-span-2 font-medium">{width.toFixed(2)}</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <span className="text-muted-foreground">Height:</span>
                    <span className="col-span-2 font-medium">{height.toFixed(2)}</span>
                  </div>
                  {rotation !== undefined && (
                    <div className="grid grid-cols-3 gap-2 text-sm">
                      <span className="text-muted-foreground">Rotation:</span>
                      <span className="col-span-2 font-medium">{rotation.toFixed(2)}°</span>
                    </div>
                  )}
                </div>
              </AccordionContent>
            </AccordionItem>

            {/* Cost Estimate Section */}
            <AccordionItem value="cost">
              <AccordionTrigger className="text-sm font-semibold">
                <div className="flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  Cost Estimate
                </div>
              </AccordionTrigger>
              <AccordionContent>
                {component.estimatedCost ? (
                  <div className="space-y-3 pt-2">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <span className="text-muted-foreground">Material Cost:</span>
                      <span className="font-medium text-right">
                        ${component.estimatedCost.material.toFixed(2)}
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <span className="text-muted-foreground">Labor Hours:</span>
                      <span className="font-medium text-right">
                        {component.estimatedCost.laborHours.toFixed(1)} hrs
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <span className="text-muted-foreground">Labor Cost:</span>
                      <span className="font-medium text-right">
                        ${component.estimatedCost.labor.toFixed(2)}
                      </span>
                    </div>
                    <Separator />
                    <div className="grid grid-cols-2 gap-2 text-base font-semibold">
                      <span>Total Estimate:</span>
                      <span className="text-primary text-right">
                        ${component.estimatedCost.total.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground pt-2">
                    No cost estimate available
                  </p>
                )}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </motion.div>
      </ScrollArea>

      <Separator />

      {/* Action Buttons */}
      <div className="p-4 space-y-2">
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start"
          onClick={() => onAction?.('flag', component.id)}
        >
          <Flag className="w-4 h-4 mr-2" />
          Flag for Review
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start"
          onClick={() => onAction?.('export', component.id)}
        >
          <Download className="w-4 h-4 mr-2" />
          Export Component Data
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="w-full justify-start"
          onClick={() => onAction?.('catalog', component.id)}
        >
          <ExternalLink className="w-4 h-4 mr-2" />
          View in Catalog
        </Button>
      </div>
    </div>
  );
}
