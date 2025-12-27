'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Info, 
  TrendingUp, 
  Bell,
  Activity,
} from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useStudioStore } from '@/lib/studio-store';
import { InspectorPanel } from '@/components/features/studio/InspectorPanel';

/**
 * Inspector Panel - Context-Aware Right Sidebar
 * - On Dashboard: Shows high-level stats or "What's New" feed
 * - On Workspace (No Selection): Shows document summary
 * - On Workspace (Component Selected): Shows component properties
 */
export default function InspectorPanelWrapper() {
  const pathname = usePathname();
  const { inspectorPanel, selectedComponentId } = useStudioStore();
  
  const isWorkspace = pathname?.startsWith('/workspace');
  const isDashboard = pathname === '/dashboard' || pathname === '/';

  // Collapsed state - hide the panel
  if (inspectorPanel.isCollapsed) {
    return null;
  }

  return (
    <div className="h-full flex flex-col bg-slate-900 border-l border-slate-800">
      <AnimatePresence mode="wait">
        {isWorkspace ? (
          <motion.div
            key="workspace-inspector"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {/* Workspace Inspector - Component Details or Document Summary */}
            <InspectorPanel component={null} />
          </motion.div>
        ) : (
          <motion.div
            key="dashboard-inspector"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.2 }}
            className="h-full flex flex-col"
          >
            {/* Dashboard Inspector - Stats & What's New */}
            <div className="studio-panel-header">
              <Activity className="w-4 h-4 mr-2" />
              <span className="font-semibold text-sm">Activity</span>
            </div>
            
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-4">
                {/* Stats Card */}
                <Card className="bg-slate-800 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <TrendingUp className="w-4 h-4 text-emerald-500" />
                      Overview
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Active Projects</span>
                      <span className="font-semibold text-white">0</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Total Documents</span>
                      <span className="font-semibold text-white">0</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Components Detected</span>
                      <span className="font-semibold text-white">0</span>
                    </div>
                  </CardContent>
                </Card>

                {/* What's New Card */}
                <Card className="bg-slate-800 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Bell className="w-4 h-4 text-blue-500" />
                      What's New
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="space-y-2">
                      <div className="text-xs text-slate-400">
                        <p className="font-semibold text-slate-300 mb-1">New Studio Layout</p>
                        <p>Experience the unified 3-panel workspace for seamless project management.</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Quick Actions */}
                <Card className="bg-slate-800 border-slate-700">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center gap-2">
                      <Info className="w-4 h-4 text-purple-500" />
                      Quick Actions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2 text-xs text-slate-400">
                      <p>• Create a new project</p>
                      <p>• Upload a blueprint</p>
                      <p>• Generate a quote</p>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </ScrollArea>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
