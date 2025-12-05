'use client';

import React from 'react';
import HVACBlueprintUploader from '@/components/hvac/HVACBlueprintUploader';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Wind, FileText, DollarSign, CheckCircle2 } from 'lucide-react';

export default function DocumentsPage() {
  return (
    <div className="container mx-auto py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">HVAC Blueprint Analysis</h1>
        <p className="text-muted-foreground">Upload blueprints for AI-powered HVAC system analysis and cost estimation</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <Wind className="h-8 w-8 text-blue-500 mb-2" />
            <CardTitle className="text-sm">AI Detection</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Automatically identify HVAC components using computer vision</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <FileText className="h-8 w-8 text-green-500 mb-2" />
            <CardTitle className="text-sm">Multi-Format</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Support for PDF, DWG, DXF, and image files</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <DollarSign className="h-8 w-8 text-purple-500 mb-2" />
            <CardTitle className="text-sm">Cost Estimation</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">Regional pricing with material and labor calculations</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CheckCircle2 className="h-8 w-8 text-orange-500 mb-2" />
            <CardTitle className="text-sm">Compliance</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-xs text-muted-foreground">ASHRAE and building code compliance checking</p>
          </CardContent>
        </Card>
      </div>

      <HVACBlueprintUploader onAnalysisComplete={(result) => console.log('Analysis complete:', result)} />

      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
          <CardDescription>Simple 3-step process for HVAC blueprint analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">1</div>
              <div>
                <h3 className="font-semibold mb-1">Upload Blueprint</h3>
                <p className="text-sm text-muted-foreground">Drag and drop or select your HVAC blueprint file.</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">2</div>
              <div>
                <h3 className="font-semibold mb-1">AI Analysis</h3>
                <p className="text-sm text-muted-foreground">Our AI engine detects HVAC components and extracts specifications.</p>
              </div>
            </div>
            <div className="flex gap-4">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center font-bold">3</div>
              <div>
                <h3 className="font-semibold mb-1">Generate Estimate</h3>
                <p className="text-sm text-muted-foreground">Review components and generate cost estimates with regional adjustments.</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
