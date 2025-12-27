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

export interface ViewportControlsProps {
  /** Callback for zoom in action */
  onZoomIn?: () => void;
  /** Callback for zoom out action */
  onZoomOut?: () => void;
  /** Callback for reset view action */
  onReset?: () => void;
  /** Callback for fullscreen toggle */
  onFullscreen?: () => void;
  /** Callback for toggling overlay layers */
  onToggleLayers?: () => void;
  /** Whether labels are currently shown */
  showLabels?: boolean;
  /** Callback for toggling labels */
  onToggleLabels?: () => void;
  /** Custom className for positioning */
  className?: string;
  /** Whether controls are visible */
  visible?: boolean;
}

export function ViewportControls({
  onZoomIn,
  onZoomOut,
  onReset,
  onFullscreen,
  onToggleLayers,
  showLabels = true,
  onToggleLabels,
  className = "fixed bottom-6 left-1/2 -translate-x-1/2 z-40",
  visible = true,
}: ViewportControlsProps) {
  if (!visible) return null;

  return (
    <motion.div
      className={className}
      initial={{ y: 100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
    >
      <div className="bg-white/80 backdrop-blur-md border border-slate-200 rounded-full shadow-lg px-4 py-2 flex items-center gap-2">
        {/* Zoom Controls */}
        {onZoomIn && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onZoomIn}
            title="Zoom In"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        )}

        {onZoomOut && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onZoomOut}
            title="Zoom Out"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
        )}

        {/* Separator */}
        {(onZoomIn || onZoomOut) && (onReset || onFullscreen) && (
          <div className="w-px h-6 bg-slate-300" />
        )}

        {/* View Controls */}
        {onReset && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onReset}
            title="Reset View"
            aria-label="Reset view"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        )}

        {onFullscreen && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onFullscreen}
            title="Fullscreen"
            aria-label="Toggle fullscreen"
          >
            <Maximize2 className="h-4 w-4" />
          </Button>
        )}

        {/* Separator */}
        {(onReset || onFullscreen) && (onToggleLayers || onToggleLabels) && (
          <div className="w-px h-6 bg-slate-300" />
        )}

        {/* Overlay Controls */}
        {onToggleLayers && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onToggleLayers}
            title="Toggle Layers"
            aria-label="Toggle overlay layers"
          >
            <Layers className="h-4 w-4" />
          </Button>
        )}

        {onToggleLabels && (
          <Button
            variant="ghost"
            size="sm"
            className="rounded-full h-9 w-9 p-0 hover:bg-slate-100"
            onClick={onToggleLabels}
            title={showLabels ? 'Hide Labels' : 'Show Labels'}
            aria-label={showLabels ? 'Hide labels' : 'Show labels'}
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