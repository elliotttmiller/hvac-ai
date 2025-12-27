'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Menu, ChevronLeft, ChevronRight } from 'lucide-react';
import { LucideIcon } from 'lucide-react';

interface NavigationItem {
  name: string;
  href: string;
  icon: LucideIcon;
  description?: string;
}

interface NavigationProps {
  collapsed?: boolean;
  onToggle?: () => void;
  navigation?: NavigationItem[];
}

import { LayoutDashboard, Building } from 'lucide-react';

const DEFAULT_NAVIGATION: NavigationItem[] = [
  {
    name: 'Dashboard',
    href: '/',
    icon: LayoutDashboard,
    description: 'Overview and metrics',
  },
  {
    name: 'Projects',
    href: '/projects',
    icon: Building,
    description: 'View all projects',
  },
];

export function Navigation({
  collapsed = false,
  onToggle,
  navigation = DEFAULT_NAVIGATION
}: NavigationProps) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const NavLinks = ({ mobile = false }: { mobile?: boolean }) => (
    <div className="space-y-1">
      {navigation.map((item) => {
        const isActive = pathname === item.href;
        const IconComponent = item.icon;
        return (
          <Link
            key={item.name}
            href={item.href}
            onClick={() => mobile && setMobileOpen(false)}
            className={cn(
              'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200',
              'hover:bg-accent hover:text-accent-foreground',
              isActive
                ? 'bg-accent text-accent-foreground shadow-sm'
                : 'text-muted-foreground',
              collapsed && !mobile && 'justify-center px-2'
            )}
            title={collapsed && !mobile ? item.name : undefined}
          >
            <IconComponent className="h-5 w-5 flex-shrink-0" />
            {(!collapsed || mobile) && (
              <span className="truncate">{item.name}</span>
            )}
          </Link>
        );
      })}
    </div>
  );

  return (
    <>
      {/* Desktop Sidebar */}
      <div
        className={cn(
          'hidden md:flex md:flex-col bg-card border-r transition-all duration-300',
          collapsed ? 'w-20' : 'w-64'
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          {!collapsed && (
            <h2 className="text-lg font-semibold">Navigation</h2>
          )}
          {onToggle && (
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggle}
              className="h-8 w-8"
            >
              {collapsed ? (
                <ChevronRight className="h-4 w-4" />
              ) : (
                <ChevronLeft className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 p-4">
          <NavLinks />
        </nav>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden">
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="fixed top-4 left-4 z-50 md:hidden"
            >
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="left" className="w-64 p-0">
            <div className="p-4 border-b">
              <h2 className="text-lg font-semibold">Navigation</h2>
            </div>
            <nav className="p-4">
              <NavLinks mobile />
            </nav>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}