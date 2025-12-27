"use client";

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { FileUploader } from '@/components/features/ingestion/FileUploader';
import { ViewportControls } from '@/components/features/visualization/ViewportControls';
import { QuoteBuilder } from '@/components/features/estimation/QuoteBuilder';
import { Button } from '@/components/ui/button';
import { PanelRight, Maximize2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { toast } from 'sonner';

// Dynamic import to avoid SSR issues with OpenSeadragon
const InteractiveViewer = dynamic(
  () => import('@/components/features/visualization/InteractiveViewer').then(mod => ({ default: mod.InteractiveViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="h-full w-full flex items-center justify-center bg-slate-800">
        <div className="text-white">Loading viewer...</div>
      </div>
    )
  }
);

interface DetectionItem {
  label: string;
  conf: number;
  box: [number, number, number, number];
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
  
  // UI State
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [selectedDetection, setSelectedDetection] = useState<DetectionItem | null>(null);

  // Test mode setup
  useEffect(() => {
    if (testMode) {
      // Mock data for UI testing
      setImageUrl('https://images.unsplash.com/photo-1581094794329-c8112a89af12?w=1200&h=800&fit=crop');
      setAnalysisData({
        detections: [
          { label: 'HVAC Unit', conf: 0.95, box: [100, 100, 300, 200] },
          { label: 'Ductwork', conf: 0.87, box: [400, 150, 600, 250] },
          { label: 'Vent', conf: 0.92, box: [200, 350, 280, 400] },
          { label: 'Thermostat', conf: 0.89, box: [500, 400, 580, 460] },
        ],
        extracted_text: 'Sample HVAC blueprint analysis',
        confidence: 0.91
      });
    }
  }, [testMode]);

  const handleDetectionSelect = (detection: DetectionItem) => {
    setSelectedDetection(detection);
    // Could add viewport focusing logic here
  };

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

  // --- Render: Empty State (Upload) ---
  if (!file || !imageUrl) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center bg-slate-50 p-6">
        <div className="w-full max-w-xl space-y-4">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-slate-900">HVAC Studio</h1>
            <p className="text-slate-500">Upload a blueprint to enter the workspace</p>
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

  // --- Render: The Studio (Viewer + Panels) ---
  return (
    <div className="flex h-full w-full overflow-hidden bg-slate-900 relative">
      <div className={`flex-1 relative transition-all duration-300 ${isPanelOpen ? 'mr-[400px]' : 'mr-0'}`}>
        <InteractiveViewer 
          sourceUrl={imageUrl} 
          detections={analysisData?.detections} 
        />
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
          <ViewportControls />
        </div>

        {!isPanelOpen && (
          <Button 
            variant="secondary" 
            size="icon" 
            className="absolute top-4 right-4 shadow-lg"
            onClick={() => setIsPanelOpen(true)}
          >
            <PanelRight size={18} />
          </Button>
        )}
      </div>

      <div 
        className={`
          fixed right-0 top-16 bottom-0 w-[400px] bg-white border-l shadow-2xl 
          transition-transform duration-300 ease-in-out z-20 flex flex-col
          ${isPanelOpen ? 'translate-x-0' : 'translate-x-full'}
        `}
      >
        <div className="h-12 border-b flex items-center justify-between px-4 bg-slate-50">
          <span className="font-semibold text-sm">Estimation & Takeoff</span>
          <Button variant="ghost" size="icon" onClick={() => setIsPanelOpen(false)}>
            <Maximize2 size={16} />
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <QuoteBuilder 
            data={null}
            analysisData={analysisData}
            onDetectionSelect={handleDetectionSelect}
          />
        </div>
      </div>
    </div>
  );
}
