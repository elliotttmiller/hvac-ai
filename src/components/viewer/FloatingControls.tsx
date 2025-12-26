'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  RotateCcw,
  Layers,
  Eye,
  EyeOff
} from 'lucide-react';

interface FloatingControlsProps {
  onZoomIn?: () => void;
  onZoomOut?: () => void;
  onReset?: () => void;
  onFullscreen?: () => void;
  onToggleLayers?: () => void;
  showLabels?: boolean;
  onToggleLabels?: () => void;
}

export default function FloatingControls({
  onZoomIn,
  onZoomOut,
  onReset,
  onFullscreen,
  onToggleLayers,
  showLabels = true,
  onToggleLabels
}: FloatingControlsProps) {
  return (
    <motion.div
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40"
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <div className="bg-white/80 backdrop-blur-md border border-slate-200 rounded-full shadow-lg px-4 py-2 flex items-center gap-2">
        {/* Zoom In */}
        <Button
          variant="ghost"
          size="sm"
          className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
          onClick={onZoomIn}
          title="Zoom In"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
        
        {/* Zoom Out */}
        <Button
          variant="ghost"
          size="sm"
          className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
          onClick={onZoomOut}
          title="Zoom Out"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        
        {/* Separator */}
        <div className="w-px h-6 bg-slate-300" />
        
        {/* Reset View */}
        <Button
          variant="ghost"
          size="sm"
          className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
          onClick={onReset}
          title="Reset View"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
        
        {/* Fullscreen */}
        {onFullscreen && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onFullscreen}
            title="Fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
        )}
        
        {/* Separator */}
        <div className="w-px h-6 bg-slate-300" />
        
        {/* Toggle Layers */}
        {onToggleLayers && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onToggleLayers}
            title="Toggle Layers"
          >
            <Layers className="h-4 w-4" />
          </Button>
        )}
        
        {/* Toggle Labels */}
        {onToggleLabels && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onToggleLabels}
            title={showLabels ? 'Hide Labels' : 'Show Labels'}
          >
            {showLabels ? (
              <Eye className="h-4 w-4" />
            ) : (
              <EyeOff className="h-4 w-4" />
            )}
          </Button>
        )}
      </div>
    </motion.div>
  );
}
