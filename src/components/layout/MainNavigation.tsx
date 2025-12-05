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
  Upload,
} from "lucide-react";

const navigation = [
  {
    name: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    description: "Overview and metrics",
  },
  {
    name: "Upload Blueprint",
    href: "/documents",
    icon: Upload,
    description: "Upload and analyze HVAC blueprints",
  },
  {
    name: "Projects",
    href: "/projects",
    icon: Building,
    description: "View all projects",
  },
  {
    name: "BIM Viewer",
    href: "/bim",
    icon: Wind,
    description: "3D visualization",
  },
];

export default function MainNavigation() {
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
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all hover:bg-accent",
              isActive
                ? "bg-accent text-accent-foreground"
                : "text-muted-foreground"
            )}
          >
            <item.icon className="h-4 w-4" />
            <span>{item.name}</span>
          </Link>
        );
      })}
    </div>
  );

  return (
    <>
      {/* Desktop Navigation */}
      <div className="hidden md:flex md:flex-col md:w-64 md:fixed md:inset-y-0 md:border-r md:bg-background">
        <div className="flex flex-col flex-1 min-h-0 pt-5 pb-4">
          <div className="flex items-center flex-shrink-0 px-4 mb-5">
            <Wind className="h-8 w-8 text-primary mr-2" />
            <h1 className="text-xl font-bold">HVAC AI</h1>
          </div>
          <div className="flex-1 flex flex-col overflow-y-auto px-3">
            <NavLinks />
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden fixed top-0 left-0 right-0 z-50 border-b bg-background">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center">
            <Wind className="h-6 w-6 text-primary mr-2" />
            <h1 className="text-lg font-bold">HVAC AI</h1>
          </div>
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="h-6 w-6" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64">
              <div className="flex items-center mb-5">
                <Wind className="h-8 w-8 text-primary mr-2" />
                <h1 className="text-xl font-bold">HVAC AI</h1>
              </div>
              <NavLinks />
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {/* Spacer for mobile */}
      <div className="md:hidden h-16" />
    </>
  );
}
