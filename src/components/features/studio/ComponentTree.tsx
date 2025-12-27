'use client';

import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { 
  ChevronDown, 
  ChevronRight, 
  Eye, 
  EyeOff,
  Box,
  Circle,
  Square,
  Hexagon,
  Triangle,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useStudioStore } from '@/lib/studio-store';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';

export interface ComponentData {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  className: string; // Normalized class name for grouping
}

interface ComponentTreeProps {
  components: ComponentData[];
  onComponentClick?: (id: string) => void;
  onComponentHover?: (id: string | null) => void;
  className?: string;
}

// Map component class names to appropriate icons
const getIconForClass = (className: string) => {
  const lower = className.toLowerCase();
  if (lower.includes('vav') || lower.includes('box')) return Box;
  if (lower.includes('duct')) return Square;
  if (lower.includes('valve')) return Circle;
  if (lower.includes('sensor')) return Hexagon;
  if (lower.includes('vent')) return Triangle;
  return Circle;
};

interface ComponentGroup {
  className: string;
  displayName: string;
  items: ComponentData[];
  count: number;
  avgConfidence: number;
}

export function ComponentTree({
  components = [],
  onComponentClick,
  onComponentHover,
  className,
}: ComponentTreeProps) {
  const {
    selectedComponentId,
    hoveredComponentId,
    componentVisibility,
    setSelectedComponent,
    setHoveredComponent,
    toggleComponentVisibility,
  } = useStudioStore();

  // Group components by class name
  const groupedComponents = useMemo(() => {
    const groups = new Map<string, ComponentGroup>();
    
    components.forEach((comp) => {
      const className = comp.className || comp.label;
      
      if (!groups.has(className)) {
        groups.set(className, {
          className,
          displayName: className,
          items: [],
          count: 0,
          avgConfidence: 0,
        });
      }
      
      const group = groups.get(className)!;
      group.items.push(comp);
      group.count++;
    });
    
    // Calculate average confidence for each group
    groups.forEach((group) => {
      const totalConf = group.items.reduce((sum, item) => sum + item.confidence, 0);
      group.avgConfidence = totalConf / group.count;
    });
    
    return Array.from(groups.values()).sort((a, b) => b.count - a.count);
  }, [components]);

  const [expandedGroups, setExpandedGroups] = React.useState<Set<string>>(
    new Set(groupedComponents.map(g => g.className))
  );

  const toggleGroup = (className: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(className)) {
        next.delete(className);
      } else {
        next.add(className);
      }
      return next;
    });
  };

  const handleComponentClick = (id: string) => {
    setSelectedComponent(id);
    onComponentClick?.(id);
  };

  const handleComponentHover = (id: string | null) => {
    setHoveredComponent(id);
    onComponentHover?.(id);
  };

  if (components.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-full p-6', className)}>
        <div className="text-center text-muted-foreground">
          <Box className="w-12 h-12 mx-auto mb-3 opacity-20" />
          <p className="text-sm">No components detected</p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="studio-panel-header justify-between">
        <div className="flex items-center gap-2">
          <Box className="w-4 h-4" />
          <span className="font-semibold text-sm">Components</span>
          <span className="text-xs text-muted-foreground">
            ({components.length})
          </span>
        </div>
      </div>

      {/* Component Tree */}
      <ScrollArea className="flex-1">
        <div className="p-2 space-y-1">
          {groupedComponents.map((group) => {
            const isExpanded = expandedGroups.has(group.className);
            const isVisible = componentVisibility[group.className] ?? true;
            const Icon = getIconForClass(group.className);

            return (
              <div key={group.className} className="space-y-1">
                {/* Group Header */}
                <div className="flex items-center gap-1 group">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 px-2 flex-1 justify-start hover:bg-secondary/80"
                    onClick={() => toggleGroup(group.className)}
                  >
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 mr-1" />
                    ) : (
                      <ChevronRight className="w-4 h-4 mr-1" />
                    )}
                    <Icon className="w-4 h-4 mr-2 text-muted-foreground" />
                    <span className="text-sm font-medium flex-1 text-left">
                      {group.displayName}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {group.count}
                    </span>
                  </Button>
                  
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 w-8 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={() => toggleComponentVisibility(group.className)}
                  >
                    {isVisible ? (
                      <Eye className="w-4 h-4" />
                    ) : (
                      <EyeOff className="w-4 h-4 text-muted-foreground" />
                    )}
                  </Button>
                </div>

                {/* Group Items */}
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="ml-4 space-y-0.5"
                  >
                    {group.items.map((item) => {
                      const isSelected = item.id === selectedComponentId;
                      const isHovered = item.id === hoveredComponentId;

                      return (
                        <button
                          key={item.id}
                          className={cn(
                            'w-full text-left px-3 py-2 rounded text-sm transition-colors',
                            'hover:bg-secondary/60',
                            isSelected && 'bg-primary/20 border-l-2 border-primary',
                            isHovered && !isSelected && 'bg-secondary/40'
                          )}
                          onClick={() => handleComponentClick(item.id)}
                          onMouseEnter={() => handleComponentHover(item.id)}
                          onMouseLeave={() => handleComponentHover(null)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="truncate flex-1">
                              {item.label}
                            </span>
                            <span className="text-xs text-muted-foreground ml-2">
                              {Math.round(item.confidence * 100)}%
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </motion.div>
                )}
              </div>
            );
          })}
        </div>
      </ScrollArea>

      <Separator />
      
      {/* Footer Stats */}
      <div className="p-3 text-xs text-muted-foreground space-y-1">
        <div className="flex justify-between">
          <span>Total Groups:</span>
          <span className="font-medium">{groupedComponents.length}</span>
        </div>
        <div className="flex justify-between">
          <span>Total Components:</span>
          <span className="font-medium">{components.length}</span>
        </div>
      </div>
    </div>
  );
}
