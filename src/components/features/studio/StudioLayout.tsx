'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  PanelLeftClose, 
  PanelLeftOpen, 
  PanelRightClose, 
  PanelRightOpen,
  Zap,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useStudioStore } from '@/lib/studio-store';
import { Button } from '@/components/ui/button';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import { ComponentTree, ComponentData } from './ComponentTree';
import { InspectorPanel, ComponentDetails } from './InspectorPanel';
import { 
  ComponentTreeSkeleton, 
  InspectorPanelSkeleton,
  ViewerSkeleton,
} from './SkeletonLoaders';
import { InteractiveViewer, OverlayItem } from '@/components/features/visualization/InteractiveViewer';
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

interface StudioLayoutProps {
  projectId: string;
  imageUrl?: string;
  detections?: DetectionItem[];
  isAnalyzing?: boolean;
  onGenerateQuote?: () => void;
  className?: string;
}

export function StudioLayout({
  projectId,
  imageUrl,
  detections = [],
  isAnalyzing = false,
  onGenerateQuote,
  className,
}: StudioLayoutProps) {
  const {
    navigatorPanel,
    inspectorPanel,
    selectedComponentId,
    hoveredComponentId,
    componentVisibility,
    setNavigatorCollapsed,
    setInspectorCollapsed,
    setNavigatorSize,
    setInspectorSize,
    setSelectedComponent,
    setHoveredComponent,
  } = useStudioStore();

  // Convert detections to component data
  const components: ComponentData[] = useMemo(() => {
    return detections.map((det, index) => ({
      id: `comp-${index}`,
      label: det.label,
      confidence: det.conf,
      bbox: det.box,
      className: det.label, // Group by label
    }));
  }, [detections]);

  // Convert detections to overlays for the viewer
  const overlays: OverlayItem[] = useMemo(() => {
    return detections.map((det, index) => {
      const id = `comp-${index}`;
      const isVisible = componentVisibility[det.label] ?? true;
      
      if (!isVisible) return null;
      
      return {
        id,
        label: det.label,
        score: det.conf,
        bbox: det.box,
        obb: det.obb,
        textContent: det.textContent,
        textConfidence: det.textConfidence,
      };
    }).filter(Boolean) as OverlayItem[];
  }, [detections, componentVisibility]);

  // Get selected component details
  const selectedComponent: ComponentDetails | null = useMemo(() => {
    if (!selectedComponentId) return null;
    
    const index = Number.parseInt(selectedComponentId.split('-')[1]);
    const detection = detections[index];
    
    if (!detection) return null;
    
    return {
      id: selectedComponentId,
      label: detection.label,
      confidence: detection.conf,
      bbox: detection.box,
      rotation: detection.obb?.rotation,
      ocrText: detection.textContent,
      ocrConfidence: detection.textConfidence,
    };
  }, [selectedComponentId, detections]);

  const handleComponentClick = (id: string) => {
    setSelectedComponent(id);
    // TODO: Pan and zoom canvas to focus on component
  };

  const handleComponentHover = (id: string | null) => {
    setHoveredComponent(id);
  };

  const handleInspectorAction = (action: string, componentId: string) => {
    switch (action) {
      case 'flag':
        toast.info('Component flagged for review');
        break;
      case 'export':
        toast.info('Exporting component data...');
        break;
      case 'catalog':
        toast.info('Opening component catalog...');
        break;
    }
  };

  return (
    <div className={cn('h-screen w-screen flex flex-col bg-background', className)}>
      {/* Top Bar */}
      <div className="h-14 border-b border-border bg-background flex items-center justify-between px-4 z-10">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold">HVAC Studio</h1>
          {projectId && (
            <span className="text-sm text-muted-foreground">
              Project: {projectId}
            </span>
          )}
          {components.length > 0 && (
            <span className="text-xs text-muted-foreground">
              {components.length} components detected
            </span>
          )}
        </div>
        
        <div className="flex items-center gap-2">
          {onGenerateQuote && components.length > 0 && (
            <Button onClick={onGenerateQuote} className="gap-2">
              <Zap className="w-4 h-4" />
              Generate Quote
            </Button>
          )}
        </div>
      </div>

      {/* 3-Panel Layout */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup orientation="horizontal">
          {/* Navigator Panel (Left) */}
          <AnimatePresence>
            {!navigatorPanel.isCollapsed && (
              <ResizablePanel
                defaultSize={navigatorPanel.size}
                minSize={15}
                maxSize={35}
                onResize={(size) => {
                  if (typeof size === 'number') {
                    setNavigatorSize(size);
                  }
                }}
                className="studio-panel"
              >
                <motion.div
                  initial={{ x: -20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -20, opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  className="h-full"
                >
                  {isAnalyzing ? (
                    <ComponentTreeSkeleton />
                  ) : (
                    <ComponentTree
                      components={components}
                      onComponentClick={handleComponentClick}
                      onComponentHover={handleComponentHover}
                    />
                  )}
                </motion.div>
              </ResizablePanel>
            )}
          </AnimatePresence>

          {!navigatorPanel.isCollapsed && <ResizableHandle withHandle />}

          {/* Workspace Panel (Center) */}
          <ResizablePanel defaultSize={100 - navigatorPanel.size - inspectorPanel.size} minSize={30}>
            <div className="relative h-full w-full bg-slate-950">
              {/* Toggle Buttons */}
              <div className="absolute top-4 left-4 z-10 flex gap-2">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setNavigatorCollapsed(!navigatorPanel.isCollapsed)}
                  className="glassmorphism"
                >
                  {navigatorPanel.isCollapsed ? (
                    <PanelLeftOpen className="w-4 h-4" />
                  ) : (
                    <PanelLeftClose className="w-4 h-4" />
                  )}
                </Button>
              </div>

              <div className="absolute top-4 right-4 z-10">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setInspectorCollapsed(!inspectorPanel.isCollapsed)}
                  className="glassmorphism"
                >
                  {inspectorPanel.isCollapsed ? (
                    <PanelRightOpen className="w-4 h-4" />
                  ) : (
                    <PanelRightClose className="w-4 h-4" />
                  )}
                </Button>
              </div>

              {/* Canvas/Viewer */}
              {isAnalyzing ? (
                <ViewerSkeleton />
              ) : imageUrl ? (
                <InteractiveViewer
                  sourceUrl={imageUrl}
                  overlays={overlays}
                  selectedId={selectedComponentId}
                  hoveredId={hoveredComponentId}
                  onSelect={setSelectedComponent}
                  onHover={setHoveredComponent}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-muted-foreground">No image loaded</p>
                </div>
              )}
            </div>
          </ResizablePanel>

          {!inspectorPanel.isCollapsed && <ResizableHandle withHandle />}

          {/* Inspector Panel (Right) */}
          <AnimatePresence>
            {!inspectorPanel.isCollapsed && (
              <ResizablePanel
                defaultSize={inspectorPanel.size}
                minSize={20}
                maxSize={40}
                onResize={(size) => {
                  if (typeof size === 'number') {
                    setInspectorSize(size);
                  }
                }}
                className="studio-panel border-l"
              >
                <motion.div
                  initial={{ x: 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: 20, opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  className="h-full"
                >
                  {isAnalyzing ? (
                    <InspectorPanelSkeleton />
                  ) : (
                    <InspectorPanel
                      component={selectedComponent}
                      onAction={handleInspectorAction}
                    />
                  )}
                </motion.div>
              </ResizablePanel>
            )}
          </AnimatePresence>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}
