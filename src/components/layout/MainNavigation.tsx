"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import {
  LayoutDashboard,
  FileText,
  Building,
  Wind,
  DollarSign,
  Menu,
} from "lucide-react";

const navigation = [
  {
    name: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    description: "Overview and metrics",
  },
  {
    name: "Projects",
    href: "/projects",
    icon: Building,
    description: "View all projects",
  },
];

interface MainNavigationProps {
  collapsed?: boolean;
  onToggle?: () => void;
}

export default function MainNavigation({ collapsed = false, onToggle }: MainNavigationProps) {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  const NavLinks = () => (
    <div className="space-y-1">
      {navigation.map((item) => {
        const isActive = pathname === item.href;
        return (
          <Link
            key={item.name}
            href={item.href}
            onClick={() => setMobileOpen(false)}
            className={cn(
              "flex items-center rounded-lg",
              "transition-all duration-200 ease-in-out",
              "hover:bg-accent hover:text-accent-foreground hover:scale-[1.02]",
              "active:scale-[0.98]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              collapsed ? 'justify-center px-2 py-3 w-14 mx-auto' : 'gap-3 px-3 py-2.5 text-sm font-medium',
              isActive 
                ? "bg-accent text-accent-foreground shadow-sm" 
                : "text-muted-foreground hover:text-foreground"
            )}
            title={collapsed ? item.name : undefined}
            aria-label={item.name}
            aria-current={isActive ? 'page' : undefined}
          >
            <item.icon className={cn(
              "flex-shrink-0 transition-all duration-200",
              collapsed ? "h-5 w-5" : "h-4 w-4"
            )} />
            <span className={cn(
              "transition-all duration-300 whitespace-nowrap",
              collapsed ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100'
            )}>
              {item.name}
            </span>
          </Link>
        );
      })}
    </div>
  );

  return (
    <>
      {/* Desktop Navigation - Industry-standard sidebar with smooth transitions */}
      <aside 
        className={`
          hidden md:flex md:flex-col 
          ${collapsed ? 'md:w-20' : 'md:w-64'} 
          md:fixed md:inset-y-0 md:border-r md:bg-background/95 md:backdrop-blur-sm
          transition-all duration-300 ease-in-out
          shadow-lg md:shadow-none
          z-40
        `}
        aria-label="Main navigation"
      >
        <div className="flex flex-col flex-1 min-h-0 pt-3 pb-4">
          {/* Header with logo and collapse toggle */}
          <div className={`flex items-center flex-shrink-0 mb-4 transition-all duration-300 ${collapsed ? 'px-2 justify-center' : 'px-4 justify-between'}`}>
            <div className={`flex items-center overflow-hidden transition-all duration-300 ${collapsed ? 'gap-0' : 'gap-2'}`}>
              <Wind className={`flex-shrink-0 text-primary transition-all duration-300 ${collapsed ? 'h-8 w-8' : 'h-7 w-7'}`} />
              <h1 
                className={`
                  font-bold whitespace-nowrap text-fluid-lg
                  transition-all duration-300
                  ${collapsed ? 'w-0 opacity-0' : 'w-auto opacity-100'}
                `}
              >
                HVAC AI
              </h1>
            </div>
            {/* Collapse Toggle Button */}
            {!collapsed && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => onToggle && onToggle()} 
                aria-label="Collapse sidebar"
                className="transition-all duration-200 hover:bg-accent"
              >
                <Menu className="h-4 w-4" />
              </Button>
            )}
            {collapsed && (
              <Button 
                variant="ghost" 
                size="icon" 
                onClick={() => onToggle && onToggle()} 
                aria-label="Expand sidebar"
                className="absolute top-3 left-1/2 -translate-x-1/2 transition-all duration-200 hover:bg-accent"
              >
                <Menu className="h-4 w-4" />
              </Button>
            )}
          </div>
          
          {/* Navigation Links */}
          <nav className="flex-1 flex flex-col overflow-y-auto px-2 scroll-smooth" aria-label="Primary navigation">
            <NavLinks />
          </nav>
        </div>
      </aside>

      {/* Mobile Navigation - Slide-out sheet with smooth backdrop */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 border-b bg-background/95 backdrop-blur-md shadow-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <Wind className="h-6 w-6 text-primary flex-shrink-0" />
            <h1 className="text-lg font-bold text-fluid-base">HVAC AI</h1>
          </div>
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild>
              <Button 
                variant="ghost" 
                size="icon"
                aria-label="Open navigation menu"
                className="transition-all duration-200 hover:bg-accent"
              >
                <Menu className="h-6 w-6" />
              </Button>
            </SheetTrigger>
            <SheetContent 
              side="left" 
              className="w-72 sm:w-80 transition-all duration-300"
            >
              <div className="flex items-center gap-2 mb-6">
                <Wind className="h-8 w-8 text-primary" />
                <h1 className="text-xl font-bold">HVAC AI</h1>
              </div>
              <nav aria-label="Mobile navigation">
                <NavLinks />
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {/* Mobile header spacer - exact height match for smooth layout */}
      <div className="md:hidden h-16" aria-hidden="true" />
    </>
  );
}
