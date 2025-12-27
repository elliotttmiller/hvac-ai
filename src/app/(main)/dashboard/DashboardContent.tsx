'use client';

import React, { useState } from 'react';
import { FileUploader } from '@/components/features/ingestion/FileUploader';
import { InteractiveViewer, OverlayItem } from '@/components/features/visualization/InteractiveViewer';
import { ViewportControls } from '@/components/features/visualization/ViewportControls';
import { QuoteBuilder, QuoteData } from '@/components/features/estimation/QuoteBuilder';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import {
  Upload,
  Eye,
  Calculator,
  FileText,
  ArrowLeft
} from 'lucide-react';

interface DashboardContentProps {
  projectId?: string;
  initialFile?: File;
  initialAnalysisData?: {
    detections?: OverlayItem[];
    quote?: QuoteData;
  };
  onBack?: () => void;
  mode?: 'dashboard' | 'workspace';
}

export default function DashboardContent({
  projectId,
  initialFile,
  initialAnalysisData,
  onBack,
  mode = 'dashboard'
}: DashboardContentProps) {
  const [file, setFile] = useState<File | null>(initialFile || null);
  const [analysisData, setAnalysisData] = useState<{
    detections?: OverlayItem[];
    quote?: QuoteData;
  } | null>(initialAnalysisData || null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState<'upload' | 'analyze' | 'quote'>(
    initialFile ? 'analyze' : 'upload'
  );

  const handleUpload = (uploadedFile: File) => {
    setFile(uploadedFile);
    setIsProcessing(true);
    setCurrentStep('analyze');

    // Simulate processing
    setTimeout(() => {
      setIsProcessing(false);
      // Mock analysis data
      setAnalysisData({
        detections: [
          {
            id: '1',
            label: 'valve',
            score: 0.95,
            bbox: [100, 100, 150, 150],
            textContent: 'V-101',
            textConfidence: 0.89
          },
          {
            id: '2',
            label: 'duct',
            score: 0.87,
            bbox: [200, 200, 300, 250]
          }
        ],
        quote: {
          quote_id: projectId ? `QUOTE-${projectId}` : 'QUOTE-2024-001',
          line_items: [
            {
              category: 'valve',
              sku_name: 'Control Valve V-101',
              count: 1,
              unit_material_cost: 250.00,
              total_line_cost: 250.00
            },
            {
              category: 'duct',
              sku_name: 'HVAC Duct Section',
              count: 1,
              unit_material_cost: 150.00,
              total_line_cost: 150.00
            }
          ],
          summary: {
            subtotal_materials: 400.00,
            subtotal_labor: 120.00,
            total_cost: 520.00,
            final_price: 572.00
          }
        }
      });
      setCurrentStep('quote');
    }, 2000);
  };

  const steps = [
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'analyze', label: 'Analyze', icon: Eye },
    { id: 'quote', label: 'Quote', icon: Calculator },
  ];

  if (!file) {
    return (
      <div className="h-full flex flex-col">
        {/* Header */}
        {mode === 'workspace' && projectId && (
          <div className="flex items-center justify-between p-6 border-b border-slate-200">
            <div className="flex items-center gap-4">
              {onBack && (
                <Button variant="ghost" size="sm" onClick={onBack}>
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back
                </Button>
              )}
              <div>
                <h1 className="text-2xl font-bold">Project {projectId}</h1>
                <p className="text-slate-600">HVAC Analysis & Quoting</p>
              </div>
            </div>
          </div>
        )}

        {/* Upload Area */}
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="max-w-md w-full">
            <FileUploader
              onUpload={handleUpload}
              isProcessing={isProcessing}
              title={mode === 'workspace' ? "Upload Blueprint" : "Upload Blueprint or Document"}
              subtitle={mode === 'workspace'
                ? "Drop your HVAC blueprint here or click to browse"
                : "Supports PDF, PNG, JPG, DWG/DXF files (Max 50MB)"
              }
              processButtonText="Analyze Document"
              showProcessControls={mode === 'dashboard'}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      {mode === 'workspace' && projectId && (
        <div className="flex items-center justify-between p-6 border-b border-slate-200">
          <div className="flex items-center gap-4">
            {onBack && (
              <Button variant="ghost" size="sm" onClick={onBack}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
            )}
            <div>
              <h1 className="text-2xl font-bold">Project {projectId}</h1>
              <p className="text-slate-600">HVAC Analysis & Quoting</p>
            </div>
          </div>

          {/* Step Indicator */}
          <div className="flex items-center gap-2">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isActive = step.id === currentStep;
              const isCompleted = steps.findIndex(s => s.id === currentStep) > index;

              return (
                <React.Fragment key={step.id}>
                  <div className={`flex items-center gap-2 px-3 py-1 rounded-lg ${
                    isActive ? 'bg-blue-100 text-blue-900' :
                    isCompleted ? 'bg-green-100 text-green-900' :
                    'bg-slate-100 text-slate-600'
                  }`}>
                    <Icon className="h-4 w-4" />
                    <span className="text-sm font-medium">{step.label}</span>
                  </div>
                  {index < steps.length - 1 && (
                    <div className={`w-8 h-px ${
                      isCompleted ? 'bg-green-400' : 'bg-slate-300'
                    }`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 h-full">
          {/* Left: Visual Analysis */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Eye className="h-5 w-5" />
                  Visual Analysis
                </CardTitle>
              </CardHeader>
              <div className="p-4">
                <div className="aspect-video bg-slate-900 rounded-lg overflow-hidden">
                  <InteractiveViewer
                    sourceUrl={URL.createObjectURL(file)}
                    overlays={analysisData?.detections}
                    className="w-full h-full"
                  />
                </div>
              </div>
            </Card>

            {/* Analysis Summary */}
            {analysisData?.detections && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Analysis Summary
                  </CardTitle>
                </CardHeader>
                <div className="p-4 space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-600">Objects Detected</span>
                    <Badge variant="secondary">{analysisData.detections.length}</Badge>
                  </div>
                  <Separator />
                  <div className="space-y-2">
                    {analysisData.detections.reduce((acc, detection) => {
                      const existing = acc.find(item => item.category === detection.label);
                      if (existing) {
                        existing.count++;
                      } else {
                        acc.push({ category: detection.label, count: 1 });
                      }
                      return acc;
                    }, [] as { category: string; count: number }[]).map(({ category, count }) => (
                      <div key={category} className="flex justify-between items-center">
                        <span className="text-sm capitalize">{category.replace(/_/g, ' ')}</span>
                        <Badge variant="outline">{count}</Badge>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            )}
          </div>

          {/* Right: Quote Builder */}
          <div className="space-y-4">
            <Card className="h-full">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calculator className="h-5 w-5" />
                  Quote Builder
                </CardTitle>
              </CardHeader>
              <div className="p-4 h-[calc(100%-80px)]">
                <QuoteBuilder
                  data={analysisData?.quote}
                  title="Cost Estimate"
                  exportButtonText="Export Quote"
                />
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}

