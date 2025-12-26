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

      {/* Feature cards removed for a cleaner, more focused UI */}

      <HVACBlueprintUploader onAnalysisComplete={(result) => console.log('Analysis complete:', result)} />

      {/* 'How It Works' section removed per UX request to streamline the page */}
    </div>
  );
}
