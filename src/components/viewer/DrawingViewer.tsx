'use client';

import React, { useRef, useEffect, useState } from 'react';

interface Detection {
  label: string;
  conf: number;
  box: [number, number, number, number]; // [x, y, w, h]
}

interface DrawingViewerProps {
  imageSrc: string;
  detections: Detection[];
}

const DrawingViewer: React.FC<DrawingViewerProps> = ({ imageSrc, detections }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Handle Resize & Canvas Drawing
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && imgRef.current && canvasRef.current) {
        // 1. Get the actual rendered dimensions of the image
        const { clientWidth, clientHeight } = imgRef.current;
        const { naturalWidth, naturalHeight } = imgRef.current;
        
        // 2. Set canvas to match rendered image exactly
        canvasRef.current.width = clientWidth;
        canvasRef.current.height = clientHeight;
        
        // 3. Calculate Scale Factor (Rendered Size / Actual Size)
        const scaleX = clientWidth / naturalWidth;
        const scaleY = clientHeight / naturalHeight;

        // 4. Draw Detections
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;
        
        ctx.clearRect(0, 0, clientWidth, clientHeight);

        detections.forEach(det => {
          const [x, y, w, h] = det.box;
          
          // Scale coordinates
          const sx = x * scaleX;
          const sy = y * scaleY;
          const sw = w * scaleX;
          const sh = h * scaleY;

          // Sleek Box Style
          ctx.strokeStyle = '#00f0ff'; // Cyan
          ctx.lineWidth = 0.8;
          ctx.shadowColor = '#00f0ff';
          ctx.shadowBlur = 10;
          
          // Draw Box
          ctx.strokeRect(sx, sy, sw, sh);

          // Draw Label Background
          ctx.fillStyle = 'rgba(0, 240, 255, 0.2)';
          ctx.fillRect(sx, sy, sw, sh);

          // Draw Text Label
          ctx.fillStyle = '#00f0ff';
          ctx.font = 'bold 12px Inter, sans-serif';
          ctx.fillText(`${det.label} ${(det.conf * 100).toFixed(0)}%`, sx, sy - 6);
        });
      }
    };

    // Attach listener
    window.addEventListener('resize', handleResize);
    
    // Trigger once on mount (with slight delay to ensure image load)
    const img = imgRef.current;
    if (img && img.complete) {
      handleResize();
    } else if (img) {
      img.addEventListener('load', handleResize);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (img) {
        img.removeEventListener('load', handleResize);
      }
    };
  }, [detections, imageSrc]);

  return (
    <div 
      ref={containerRef} 
      className="w-full h-full flex items-center justify-center bg-slate-900 overflow-hidden relative"
    >
      {/* 
         The image is limited by max-w and max-h to ensure it fits 
         the viewport perfectly without scrolling.
      */}
      <div className="relative shadow-2xl shadow-black/50">
        <img 
          ref={imgRef}
          src={imageSrc} 
          alt="HVAC Drawing" 
          className="max-w-full max-h-[90vh] object-contain block select-none"
        />
        <canvas 
          ref={canvasRef}
          className="absolute top-0 left-0 pointer-events-none"
        />
      </div>

      {/* Floating Info Panel */}
      <div className="absolute bottom-8 right-8 bg-slate-950/80 backdrop-blur-md border border-slate-800 p-4 rounded-xl text-sm text-slate-300">
        <p className="font-bold text-white mb-2">Detection Summary</p>
        <ul className="space-y-1">
          <li className="flex justify-between gap-8">
            <span>Ducts Found:</span> <span className="text-cyan-400">{detections.filter(d => d.label.toLowerCase().includes('duct')).length}</span>
          </li>
          <li className="flex justify-between gap-8">
            <span>Vents Found:</span> <span className="text-cyan-400">{detections.filter(d => d.label.toLowerCase().includes('vent')).length}</span>
          </li>
          <li className="flex justify-between gap-8">
            <span>Total Items:</span> <span className="text-cyan-400">{detections.length}</span>
          </li>
          <li className="flex justify-between gap-8">
            <span>Confidence:</span> <span className="text-green-400">High</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default DrawingViewer;
