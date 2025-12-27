'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Folder, 
  FileText, 
  Users, 
  BookOpen,
  Home,
  Box,
} from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { useStudioStore } from '@/lib/studio-store';
import { ComponentTree } from '@/components/features/studio/ComponentTree';

/**
 * Navigator Panel - Context-Aware Left Sidebar
 * - On Dashboard: Shows high-level navigation (Teams, Libraries, Templates)
 * - On Workspace: Shows component tree for active document
 */
export default function NavigatorPanel() {
  const pathname = usePathname();
  const { navigatorPanel } = useStudioStore();
  
  const isWorkspace = pathname?.startsWith('/workspace');
  const isDashboard = pathname === '/dashboard' || pathname === '/';

  // Collapsed state - hide the panel
  if (navigatorPanel.isCollapsed) {
    return null;
  }

  return (
    <div className="h-full flex flex-col bg-slate-900 border-r border-slate-800">
      <AnimatePresence mode="wait">
        {isWorkspace ? (
          <motion.div
            key="workspace-navigator"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {/* Workspace Navigator - Component Tree */}
            <div className="studio-panel-header">
              <Box className="w-4 h-4 mr-2" />
              <span className="font-semibold text-sm">Document Structure</span>
            </div>
            <div className="flex-1 overflow-hidden">
              <ComponentTree components={[]} />
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="dashboard-navigator"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
            className="h-full flex flex-col"
          >
            {/* Dashboard Navigator - High-Level Navigation */}
            <div className="studio-panel-header">
              <Home className="w-4 h-4 mr-2" />
              <span className="font-semibold text-sm">Navigation</span>
            </div>
            
            <ScrollArea className="flex-1">
              <div className="p-4 space-y-6">
                {/* Teams Section */}
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2">
                    Teams
                  </h3>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-slate-300 hover:text-white hover:bg-slate-800"
                  >
                    <Users className="w-4 h-4 mr-3" />
                    My Team
                  </Button>
                </div>

                <Separator className="bg-slate-800" />

                {/* Libraries Section */}
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2">
                    Libraries
                  </h3>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-slate-300 hover:text-white hover:bg-slate-800"
                  >
                    <Folder className="w-4 h-4 mr-3" />
                    Component Library
                  </Button>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-slate-300 hover:text-white hover:bg-slate-800"
                  >
                    <BookOpen className="w-4 h-4 mr-3" />
                    Documentation
                  </Button>
                </div>

                <Separator className="bg-slate-800" />

                {/* Templates Section */}
                <div className="space-y-2">
                  <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-2">
                    Templates
                  </h3>
                  <Button
                    variant="ghost"
                    className="w-full justify-start text-slate-300 hover:text-white hover:bg-slate-800"
                  >
                    <FileText className="w-4 h-4 mr-3" />
                    Project Templates
                  </Button>
                </div>
              </div>
            </ScrollArea>

            {/* Footer */}
            <div className="p-4 border-t border-slate-800">
              <p className="text-xs text-slate-500 text-center">
                HVAC Studio Navigator
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
