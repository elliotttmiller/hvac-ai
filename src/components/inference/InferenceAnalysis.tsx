'use client';

import React, { useState, useCallback, useRef, useEffect } from 'react';
import type { Segment, CountResult } from '@/types/analysis';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Upload,
  Scan,
  Download,
  Loader2,
  CheckCircle2,
  AlertCircle,
  X,
  FileBarChart,
  Eye,
} from 'lucide-react';
import { toast } from 'sonner';

// --- Types ---
type AnalysisState = 'idle' | 'analyzing';
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || '';

// --- Color Palette for Classes ---
const CLASS_COLORS: Record<string, string> = {
  valve: '#ef4444', // Red
  instrument: '#3b82f6', // Blue
  sensor: '#10b981', // Emerald
  duct: '#f59e0b', // Amber
  default: '#8b5cf6', // Violet
};

function getColorForLabel(label: string) {
  const lower = label.toLowerCase();
  if (lower.includes('valve')) return CLASS_COLORS.valve;
  if (lower.includes('instrument') || lower.includes('computer') || lower.includes('plc')) return CLASS_COLORS.instrument;
  if (lower.includes('sensor')) return CLASS_COLORS.sensor;
  if (lower.includes('duct')) return CLASS_COLORS.duct;
  return CLASS_COLORS.default;
}

export default function SAMAnalysis() {
  // --- State ---
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [analysisState, setAnalysisState] = useState<AnalysisState>('idle');
  const [segments, setSegments] = useState<Segment[]>([]);
  const [countResult, setCountResult] = useState<CountResult | null>(null);
  
  // Visualization Options
  const [showLabels, setShowLabels] = useState(true);
  const [showFill, setShowFill] = useState(true);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // --- 1. Image Initialization ---
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    imageRef.current = img;
  }, []);

  // --- 2. The Vector Drawing Engine ---
  const drawCanvasContent = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imageRef.current;
    if (!canvas || !img || !img.src) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // A. Draw Base Image
    // We render at natural resolution for sharpness, CSS handles display size
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // B. Draw Segments
    segments.forEach((segment, index) => {
      const isHovered = index === hoveredIndex;
      const baseColor = getColorForLabel(segment.label);
      
      // If we have vector polygon data (Preferred)
      if (segment.polygon && segment.polygon.length > 0) {
        // Normalize polygon shape: support both [ [x,y], ... ] and [ [[x,y], ...], [[x,y], ...] ]
        let poly2d: number[][] | null = null;
        try {
          const first = segment.polygon[0];

          const isPoint = (p: unknown): p is number[] => {
            return Array.isArray(p) && typeof (p as unknown[])[0] === 'number';
          };

          const isPolyOfPolys = (p: unknown): p is number[][][] => {
            return Array.isArray(p) && Array.isArray((p as unknown[])[0]) && Array.isArray(((p as unknown[])[0] as unknown[])[0]);
          };

          if (isPoint(first)) {
            poly2d = segment.polygon as number[][];
          } else if (isPolyOfPolys(segment.polygon)) {
            poly2d = (segment.polygon as number[][][])[0];
          }
        } catch (e) {
          poly2d = null;
        }

        if (poly2d && poly2d.length > 0) {
          ctx.beginPath();
          // Move to first point
          ctx.moveTo(poly2d[0][0] as number, poly2d[0][1] as number);
          // Draw lines to subsequent points
          for (let i = 1; i < poly2d.length; i++) {
            ctx.lineTo(poly2d[i][0] as number, poly2d[i][1] as number);
          }
          ctx.closePath();
        }

        // 1. Stroke (Outline)
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = isHovered ? 4 : 2; // Thicker on hover
        ctx.stroke();

        // 2. Fill (Subtle)
        if (showFill || isHovered) {
          ctx.fillStyle = baseColor;
          // Very transparent normally (0.1), slightly more on hover (0.3)
          ctx.globalAlpha = isHovered ? 0.3 : 0.1; 
          ctx.fill();
          ctx.globalAlpha = 1.0; // Reset
        }
      } 
      // Fallback for BBox only if no polygon
      else {
        const [x, y, x2, y2] = segment.bbox;
        const w = x2 - x;
        const h = y2 - y;
        
        ctx.strokeStyle = baseColor;
        ctx.lineWidth = isHovered ? 4 : 2;
        ctx.strokeRect(x, y, w, h);
      }

      // C. Draw Labels (Optimized)
      // Only draw if enabled globally OR if this specific item is hovered
      if (showLabels || isHovered) {
        const [x, y] = segment.bbox;
        const labelText = `${segment.label} ${Math.round(segment.score * 100)}%`;
        
        ctx.font = isHovered ? 'bold 16px Inter, sans-serif' : '12px Inter, sans-serif';
        const textMetrics = ctx.measureText(labelText);
        
        const pad = 4;
        const textW = textMetrics.width + (pad * 2);
        const textH = isHovered ? 24 : 18;

        // Draw Label Background
        ctx.fillStyle = isHovered ? baseColor : 'rgba(0,0,0,0.6)';
        ctx.fillRect(x, y - textH, textW, textH);

        // Draw Label Text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(labelText, x + pad, y - 6);
      }
    });
  }, [segments, hoveredIndex, showLabels, showFill]);

  // --- 3. Interaction Handlers ---

  // Handle Mouse Move for Hover Effects
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    // Calculate mouse position relative to actual image coordinates
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Find the first segment containing the mouse
    // We iterate backwards to find the "top-most" item visually
    let foundIndex: number | null = null;
    for (let i = segments.length - 1; i >= 0; i--) {
      const [x1, y1, x2, y2] = segments[i].bbox;
      if (mouseX >= x1 && mouseX <= x2 && mouseY >= y1 && mouseY <= y2) {
        foundIndex = i;
        break;
      }
    }

    if (foundIndex !== hoveredIndex) {
      setHoveredIndex(foundIndex);
    }
  }, [segments, hoveredIndex]);

  // Handle Image Load
  useEffect(() => {
    const img = imageRef.current;
    const canvas = canvasRef.current;
    if (!img) return;

    const handleLoad = () => {
      if (canvas) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        drawCanvasContent();
      }
    };
    img.addEventListener('load', handleLoad);
    
    if (uploadedImage) {
      const url = URL.createObjectURL(uploadedImage);
      img.src = url;
      return () => { URL.revokeObjectURL(url); img.removeEventListener('load', handleLoad); };
    }
  }, [uploadedImage, drawCanvasContent]);

  // Redraw when interaction state changes
  useEffect(() => {
    requestAnimationFrame(drawCanvasContent);
  }, [drawCanvasContent]);


  // --- 4. API Call ---
  const handleAnalyze = async () => {
    if (!uploadedImage) return;
    setAnalysisState('analyzing');
    setSegments([]);
    setCountResult(null);

    const formData = new FormData();
    formData.append('image', uploadedImage);
    // We set a balanced threshold of 0.50 as discussed
    formData.append('conf_threshold', '0.50'); 

    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Analysis failed');
      
      const data = await res.json();
      setSegments(data.segments);
      setCountResult(data);
      toast.success(`Found ${data.total_objects_found} components`);
    } catch (e) {
      toast.error('Failed to analyze image');
      console.error(e);
    } finally {
      setAnalysisState('idle');
    }
  };

  const handleDrop = useCallback((files: File[]) => {
    if (files.length > 0) setUploadedImage(files[0]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
    multiple: false
  });

  // --- Render ---
  return (
    <div className="container mx-auto py-8 space-y-6">
      
      {/* Upload Area */}
      <Card>
        <CardHeader>
          <CardTitle>HVAC Blueprint Analysis</CardTitle>
          <CardDescription>Upload a P&ID or Floor Plan to detect symbols</CardDescription>
        </CardHeader>
        <CardContent>
          {!uploadedImage ? (
            <div {...getRootProps()} className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
              <input {...getInputProps()} />
              <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
              <p className="text-lg font-medium text-gray-700">Drop blueprint here</p>
              <p className="text-sm text-gray-500">Supports PNG, JPG (High Res recommended)</p>
            </div>
          ) : (
            <div className="flex items-center justify-between bg-slate-50 p-4 rounded-lg border">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 bg-blue-100 rounded flex items-center justify-center text-blue-600">
                  <Scan className="h-5 w-5" />
                </div>
                <div>
                  <p className="font-medium text-slate-900">{uploadedImage.name}</p>
                  <p className="text-xs text-slate-500">{(uploadedImage.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button onClick={handleAnalyze} disabled={analysisState === 'analyzing'}>
                  {analysisState === 'analyzing' ? <Loader2 className="animate-spin mr-2 h-4 w-4" /> : <Scan className="mr-2 h-4 w-4" />}
                  Analyze Diagram
                </Button>
                <Button variant="ghost" size="icon" onClick={() => { setUploadedImage(null); setSegments([]); setCountResult(null); }}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Visualization Canvas */}
      {uploadedImage && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          
          {/* Main Canvas Area */}
          <Card className="lg:col-span-3 overflow-hidden">
            <CardHeader className="flex flex-row items-center justify-between py-4">
              <CardTitle className="text-base">Visual Inspection</CardTitle>
              
              {/* Visualization Controls */}
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch id="show-labels" checked={showLabels} onCheckedChange={setShowLabels} />
                  <Label htmlFor="show-labels" className="text-sm cursor-pointer">Labels</Label>
                </div>
                <div className="flex items-center gap-2">
                  <Switch id="show-fill" checked={showFill} onCheckedChange={setShowFill} />
                  <Label htmlFor="show-fill" className="text-sm cursor-pointer">Fill</Label>
                </div>
              </div>
            </CardHeader>
            
            <CardContent className="p-0 bg-slate-900 relative min-h-[500px] flex items-center justify-center overflow-auto">
              <canvas 
                ref={canvasRef}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => setHoveredIndex(null)}
                className="max-w-full h-auto shadow-2xl"
                style={{ cursor: hoveredIndex !== null ? 'pointer' : 'default' }}
              />
              {/* Hover Tooltip Overlay */}
              {hoveredIndex !== null && segments[hoveredIndex] && (
                <div 
                  className="absolute bottom-4 left-4 bg-black/80 backdrop-blur text-white p-3 rounded-lg shadow-xl border border-white/10 z-10"
                >
                  <p className="font-bold text-sm text-blue-400">{segments[hoveredIndex].label}</p>
                  <p className="text-xs text-gray-300">Confidence: {(segments[hoveredIndex].score * 100).toFixed(1)}%</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Sidebar Stats */}
          <Card className="lg:col-span-1 h-fit">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <FileBarChart className="h-4 w-4" /> Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              {countResult ? (
                <div className="space-y-4">
                  <div className="text-center p-4 bg-slate-50 rounded-lg border">
                    <div className="text-3xl font-bold text-slate-900">{countResult.total_objects_found}</div>
                    <div className="text-xs text-slate-500 uppercase tracking-wide font-semibold">Components Found</div>
                  </div>
                  
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                    {Object.entries(countResult.counts_by_category)
                      .sort(([,a], [,b]) => b - a)
                      .map(([label, count]) => (
                      <div key={label} className="flex items-center justify-between text-sm p-2 hover:bg-slate-50 rounded group">
                        <span className="truncate max-w-[140px] text-slate-700" title={label}>{label}</span>
                        <Badge variant="secondary" className="group-hover:bg-blue-100 group-hover:text-blue-700 transition-colors">
                          {count}
                        </Badge>
                      </div>
                    ))}
                  </div>
                  
                  <Button variant="outline" className="w-full text-xs" onClick={() => toast.info("Export feature coming soon")}>
                    <Download className="mr-2 h-3 w-3" /> Export CSV
                  </Button>
                </div>
              ) : (
                <div className="text-center py-12 text-gray-400 text-sm">
                  Run analysis to see component breakdown
                </div>
              )}
            </CardContent>
          </Card>

        </div>
      )}
    </div>
  );
}