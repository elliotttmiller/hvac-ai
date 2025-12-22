'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Upload, CheckCircle, Loader2, RefreshCw, Maximize2 } from 'lucide-react';
import DrawingViewer from '@/components/viewer/DrawingViewer';

type AppState = 'IDLE' | 'UPLOADING' | 'PROCESSING' | 'COMPLETE';

interface Detection {
  label: string;
  conf: number;
  box: [number, number, number, number]; // [x, y, w, h]
}

const InfiniteWorkspace = () => {
  const [appState, setAppState] = useState<AppState>('IDLE');
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  
  // Cleanup object URL on unmount to prevent memory leaks
  useEffect(() => {
    return () => {
      if (imageSrc) {
        URL.revokeObjectURL(imageSrc);
      }
    };
  }, [imageSrc]);
  
  // Mock function to simulate backend inference
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // 1. Transition to Uploading
    setAppState('UPLOADING');
    const objectUrl = URL.createObjectURL(file);
    setImageSrc(objectUrl);

    // 2. Simulate Processing (Replace this with your fetch/axios call)
    setTimeout(() => {
      setAppState('PROCESSING');
      
      // Simulate API delay
      setTimeout(() => {
        // MOCK DATA: Replace with actual YOLOv11 response
        const mockYoloResponse: Detection[] = [
          { label: 'Vent', conf: 0.92, box: [100, 100, 150, 100] }, // [x, y, w, h]
          { label: 'Thermostat', conf: 0.88, box: [300, 250, 50, 50] },
          { label: 'Duct Work', conf: 0.75, box: [50, 400, 600, 50] },
        ];
        
        setDetections(mockYoloResponse);
        setAppState('COMPLETE');
      }, 2000);
    }, 800);
  };

  const resetApp = () => {
    setAppState('IDLE');
    if (imageSrc) {
      URL.revokeObjectURL(imageSrc);
    }
    setImageSrc(null);
    setDetections([]);
  };

  return (
    <div className="h-screen w-screen bg-slate-950 text-white overflow-hidden flex flex-col relative">
      
      {/* Header - Disappears or becomes transparent when viewing results */}
      <header className={`absolute top-0 w-full p-6 z-50 transition-all duration-500 ${appState === 'COMPLETE' ? 'opacity-0 hover:opacity-100' : 'opacity-100'}`}>
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold tracking-tighter bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent">
            HVAC<span className="font-light text-slate-400">.AI</span>
          </h1>
          {appState === 'COMPLETE' && (
            <button onClick={resetApp} className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors">
              <RefreshCw size={16} /> Upload New
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 flex items-center justify-center relative">
        
        {/* State 1: IDLE / Upload Area */}
        {appState === 'IDLE' && (
          <div className="animate-fade-in text-center p-12 border-2 border-dashed border-slate-700 rounded-3xl bg-slate-900/50 backdrop-blur-sm hover:border-blue-500/50 transition-all duration-300 group">
            <input 
              type="file" 
              accept="image/*,.pdf" 
              onChange={handleFileUpload} 
              className="hidden" 
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
              <div className="p-6 bg-slate-800 rounded-full group-hover:scale-110 transition-transform duration-300">
                <Upload size={32} className="text-blue-400" />
              </div>
              <div>
                <h2 className="text-xl font-semibold">Upload HVAC Drawing</h2>
                <p className="text-slate-400 mt-2">Supports JPG, PNG, PDF</p>
              </div>
            </label>
          </div>
        )}

        {/* State 2: Processing Overlay */}
        {(appState === 'UPLOADING' || appState === 'PROCESSING') && (
          <div className="absolute inset-0 z-40 bg-slate-950 flex flex-col items-center justify-center">
            <div className="relative">
              <div className="absolute inset-0 bg-blue-500 blur-xl opacity-20 animate-pulse"></div>
              <Loader2 size={64} className="text-blue-500 animate-spin relative z-10" />
            </div>
            <p className="mt-8 text-lg font-light tracking-widest uppercase text-slate-400 animate-pulse">
              {appState === 'UPLOADING' ? 'Uploading...' : 'Analyzing Geometry'}
            </p>
          </div>
        )}

        {/* State 3: Result Viewer (The Viewport Optimized View) */}
        {appState === 'COMPLETE' && imageSrc && (
          <DrawingViewer imageSrc={imageSrc} detections={detections} />
        )}

      </main>
    </div>
  );
};

export default InfiniteWorkspace;
