'use client';

import React, { useState, useEffect } from 'react';
import Uploader from '@/components/uploader/Uploader';
import AnalysisDashboard from '@/components/hvac/AnalysisDashboard';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Wind, FileText, DollarSign, CheckCircle2 } from 'lucide-react';
import type { AnalysisResult } from '@/types/analysis';
import { checkPricingAvailable } from '@/lib/api-client';

export default function DocumentsPage() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [pricingAvailable, setPricingAvailable] = useState<boolean | undefined>(undefined);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await checkPricingAvailable();
        if (mounted) setPricingAvailable(!!res.available);
      } catch (err) {
        if (mounted) setPricingAvailable(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="container mx-auto py-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">HVAC Blueprint Analysis</h1>
        <p className="text-muted-foreground">Upload blueprints for AI-powered HVAC system analysis and cost estimation</p>
      </div>

      {!analysisResult ? (
        <Uploader
          action="analyze"
          showMetadataFields={{ projectId: true, location: true }}
          onFileSelected={(f) => setImageFile(f)}
          onAnalyzeComplete={(result) => setAnalysisResult(result)}
        />
      ) : (
        // Render the studio as a fixed full-screen surface so the dashboard
        // fully owns the viewport and the page body doesn't scroll.
        <div>
          <FullScreenStudio
            projectId={analysisResult.analysis_id || 'HVAC-PROJ-001'}
            location={''}
            analysisResult={analysisResult}
            imageFile={imageFile}
            pricingAvailable={pricingAvailable}
            onBackToUpload={() => {
              setAnalysisResult(null);
              setImageFile(null);
            }}
          />
        </div>
      )}
    </div>
  );
}

// Small full-screen wrapper component to lock body scroll while the
// studio is open and render the AnalysisDashboard as a fixed viewport.
function FullScreenStudio(props: {
  projectId: string;
  location: string;
  analysisResult: AnalysisResult;
  imageFile: File | null;
  pricingAvailable?: boolean;
  onBackToUpload?: () => void;
}) {
  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = prev; };
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-white">
      <AnalysisDashboard {...props} />
    </div>
  );
}
