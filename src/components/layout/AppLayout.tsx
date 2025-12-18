"use client";

import { ReactNode, useState, useEffect } from "react";
import MainNavigation from "./MainNavigation";
import TopHeader from "./TopHeader";

interface AppLayoutProps {
  children: ReactNode;
}

export default function AppLayout({ children }: AppLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile viewport and auto-collapse on small screens
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768; // md breakpoint
      setIsMobile(mobile);
      // Auto-collapse on mobile for more space
      if (mobile && !collapsed) {
        setCollapsed(true);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, [collapsed]);

  return (
    <div className="h-screen flex overflow-hidden bg-background">
      {/* Sidebar Navigation */}
      <MainNavigation collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />

      {/* Main Content Area - Responsive margins with smooth transitions */}
      <div 
        className={`
          flex-1 flex flex-col 
          transition-all duration-300 ease-in-out
          ${isMobile ? 'ml-0' : (collapsed ? 'ml-20' : 'ml-64')}
        `}
      >
        {/* Top Header */}
        <TopHeader />

        {/* Page Content - Responsive padding and smooth scrolling */}
        <main className="flex-1 overflow-auto scroll-smooth">
          <div className="p-4 sm:p-6 lg:p-8 max-w-[100vw] transition-all duration-300">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}
