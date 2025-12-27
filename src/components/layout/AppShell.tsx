'use client';

import { ReactNode, useState, useEffect } from 'react';
import { Navigation } from './Navigation';
import { Header } from './Header';

import { LucideIcon } from 'lucide-react';

interface AppShellProps {
  children: ReactNode;
  /** Custom navigation items */
  navigation?: Array<{
    name: string;
    href: string;
    icon: LucideIcon;
    description?: string;
  }>;
  /** Custom header content */
  headerContent?: ReactNode;
  /** Whether sidebar should be collapsible */
  collapsible?: boolean;
  /** Default collapsed state */
  defaultCollapsed?: boolean;
}

export function AppShell({
  children,
  navigation,
  headerContent,
  collapsible = true,
  defaultCollapsed = false,
}: AppShellProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [isMobile, setIsMobile] = useState(false);

  // Detect mobile viewport and auto-collapse on small screens
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768; // md breakpoint
      setIsMobile(mobile);
      // Auto-collapse on mobile for more space
      if (mobile && !collapsed && collapsible) {
        setCollapsed(true);
      }
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, [collapsed, collapsible]);

  return (
    <div className="h-screen flex overflow-hidden bg-background">
      {/* Sidebar Navigation */}
      <Navigation
        collapsed={collapsed}
        onToggle={collapsible ? () => setCollapsed((c) => !c) : undefined}
        navigation={navigation}
      />

      {/* Main Content Area - Responsive margins with smooth transitions */}
      <div
        className={`
          flex-1 flex flex-col
          transition-all duration-300 ease-in-out
          ${isMobile ? 'ml-0' : (collapsed ? 'ml-20' : 'ml-64')}
        `}
      >
        {/* Top Header */}
        <Header content={headerContent} />

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