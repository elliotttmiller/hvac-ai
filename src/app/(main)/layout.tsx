'use client';

import React from 'react';
import { usePathname } from 'next/navigation';
import AuthGuard from '@/components/shared/AuthGuard';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import { useStudioStore } from '@/lib/studio-store';
import { Button } from '@/components/ui/button';
import { 
  PanelLeftClose, 
  PanelLeftOpen, 
  PanelRightClose, 
  PanelRightOpen,
} from 'lucide-react';

interface MainLayoutProps {
  children: React.ReactNode;
  navigator: React.ReactNode;
  inspector: React.ReactNode;
}

export default function MainLayout({ children, navigator, inspector }: MainLayoutProps) {
  const pathname = usePathname();
  const {
    navigatorPanel,
    inspectorPanel,
    setNavigatorCollapsed,
    setInspectorCollapsed,
  } = useStudioStore();

  return (
    <AuthGuard>
      <div className="h-screen w-screen flex flex-col overflow-hidden bg-slate-950">
        {/* Main 3-Panel Layout */}
        <div className="flex-1 overflow-hidden">
          <ResizablePanelGroup direction="horizontal">
            {/* Left Panel - Navigator */}
            {!navigatorPanel.isCollapsed && (
              <>
                <ResizablePanel
                  defaultSize={navigatorPanel.size}
                  minSize={15}
                  maxSize={35}
                  className="relative"
                >
                  {navigator}
                </ResizablePanel>
                <ResizableHandle className="w-px bg-slate-800 hover:bg-slate-700 transition-colors" />
              </>
            )}

            {/* Center Panel - Main Content */}
            <ResizablePanel defaultSize={100 - navigatorPanel.size - inspectorPanel.size} minSize={30}>
              <div className="relative h-full w-full bg-slate-950">
                {/* Toggle Buttons - Floating on top of content */}
                <div className="absolute top-4 left-4 z-50 flex gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setNavigatorCollapsed(!navigatorPanel.isCollapsed)}
                    className="glassmorphism shadow-lg"
                  >
                    {navigatorPanel.isCollapsed ? (
                      <PanelLeftOpen className="w-4 h-4" />
                    ) : (
                      <PanelLeftClose className="w-4 h-4" />
                    )}
                  </Button>
                </div>

                <div className="absolute top-4 right-4 z-50">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setInspectorCollapsed(!inspectorPanel.isCollapsed)}
                    className="glassmorphism shadow-lg"
                  >
                    {inspectorPanel.isCollapsed ? (
                      <PanelRightOpen className="w-4 h-4" />
                    ) : (
                      <PanelRightClose className="w-4 h-4" />
                    )}
                  </Button>
                </div>

                {/* Main Content Area */}
                <div className="h-full w-full overflow-auto">
                  {children}
                </div>
              </div>
            </ResizablePanel>

            {/* Right Panel - Inspector */}
            {!inspectorPanel.isCollapsed && (
              <>
                <ResizableHandle className="w-px bg-slate-800 hover:bg-slate-700 transition-colors" />
                <ResizablePanel
                  defaultSize={inspectorPanel.size}
                  minSize={20}
                  maxSize={40}
                  className="relative"
                >
                  {inspector}
                </ResizablePanel>
              </>
            )}
          </ResizablePanelGroup>
        </div>
      </div>
    </AuthGuard>
  );
}