"use client";

import React, { useState, useEffect } from 'react';
import { FileUploader } from '@/components/features/ingestion/FileUploader';
import { StudioLayout } from '@/components/features/studio/StudioLayout';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

interface DetectionItem {
  label: string;
  conf: number;
  box: [number, number, number, number];
  obb?: {
    x_center: number;
    y_center: number;
    width: number;
    height: number;
    rotation: number;
  };
  textContent?: string;
  textConfidence?: number;
}

interface AnalysisResult {
  detections?: DetectionItem[];
  extracted_text?: string;
  confidence?: number;
  [key: string]: unknown;
}

interface StudioProps {
  projectId: string;
  testMode?: boolean;
}

export default function HVACStudio({ projectId, testMode = false }: StudioProps) {
  const router = useRouter();

  // --- State Machine ---
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  // Test mode setup
  useEffect(() => {
    if (testMode) {
      // Mock data for UI testing with OBB data
      setImageUrl('https://images.unsplash.com/photo-1581094794329-c8112a89af12?w=1200&h=800&fit=crop');
      setAnalysisData({
        detections: [
          { 
            label: 'VAV Box', 
            conf: 0.95, 
            box: [100, 100, 300, 200],
            obb: { x_center: 200, y_center: 150, width: 200, height: 100, rotation: 0 }
          },
          { 
            label: 'VAV Box', 
            conf: 0.92, 
            box: [400, 150, 600, 250],
            obb: { x_center: 500, y_center: 200, width: 200, height: 100, rotation: 15 }
          },
          { 
            label: 'Ductwork', 
            conf: 0.87, 
            box: [400, 150, 600, 250],
            textContent: 'SUPPLY-12"',
            textConfidence: 0.85
          },
          { 
            label: 'Sensor', 
            conf: 0.92, 
            box: [200, 350, 280, 400],
          },
          { 
            label: 'Valve', 
            conf: 0.89, 
            box: [500, 400, 580, 460],
          },
        ],
        extracted_text: 'Sample HVAC blueprint analysis',
        confidence: 0.91
      });
    }
  }, [testMode]);

  const handleUpload = async (uploadedFile: File) => {
    setFile(uploadedFile);
    const url = URL.createObjectURL(uploadedFile);
    setImageUrl(url);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('project_id', projectId);
      formData.append('generate_quote', 'false');

      const response = await fetch('/api/analysis', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data = await response.json();
      setAnalysisData(data);
      toast.success('Blueprint analyzed successfully');
    } catch (error) {
      console.error(error);
      toast.error('Failed to analyze blueprint');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleGenerateQuote = () => {
    toast.info('Quote generation coming soon');
    // TODO: Navigate to quote generation or trigger quote modal
  };

  // --- Render: Empty State (Upload) ---
  if (!file || !imageUrl) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center bg-slate-50 dark:bg-slate-950 p-6">
        <div className="w-full max-w-xl space-y-4">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold">HVAC Studio</h1>
            <p className="text-muted-foreground">Upload a blueprint to enter the workspace</p>
          </div>
          <FileUploader 
            onUpload={handleUpload} 
            isProcessing={isProcessing}
            title="Import Blueprint"
            subtitle="PDF, PNG, JPG supported"
          />
        </div>
      </div>
    );
  }

  // --- Render: The Studio (3-Panel Layout) ---
  return (
    <StudioLayout
      projectId={projectId}
      imageUrl={imageUrl}
      detections={analysisData?.detections}
      isAnalyzing={isProcessing}
      onGenerateQuote={handleGenerateQuote}
    />
  );
}
