"use client";

import { useEffect, useState, useRef } from "react";
import { usePathname } from "next/navigation";
import AppLayout from "@/components/layout/AppLayout";
import AuthGuard from "@/components/auth/AuthGuard";
import { Toaster } from "@/components/ui/sonner";
import { initializeProduction } from "@/lib/production-config";
import GlobalLoader from "@/components/ui/GlobalLoader";

export default function ClientBody({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  // Navigation loader timing configuration (ms)
  const NAV_LOADING_DELAY = 150; // delay before showing the overlay
  const NAV_MIN_VISIBLE = 500; // minimum time the overlay remains visible

  const [isNavigating, setIsNavigating] = useState(false);
  const prevPathname = useRef<string | null>(null);

  const navDelayRef = useRef<number | null>(null);
  const navMinRef = useRef<number | null>(null);
  const navShownAt = useRef<number | null>(null);

  // When pathname changes, schedule showing the loader after a short delay.
  // Once shown, keep it visible for at least NAV_MIN_VISIBLE to avoid flashes.
  useEffect(() => {
    // initialize previous pathname on first run
    if (prevPathname.current == null) {
      prevPathname.current = pathname ?? null;
      return;
    }

    if (prevPathname.current === pathname) return;

    // clear any existing timers
    if (navDelayRef.current) {
      clearTimeout(navDelayRef.current);
      navDelayRef.current = null;
    }
    if (navMinRef.current) {
      clearTimeout(navMinRef.current);
      navMinRef.current = null;
    }

    // start delay to show loader
    navDelayRef.current = window.setTimeout(() => {
      navDelayRef.current = null;
      navShownAt.current = Date.now();
      setIsNavigating(true);

      // ensure minimum visible time
      navMinRef.current = window.setTimeout(() => {
        navMinRef.current = null;
        setIsNavigating(false);
        navShownAt.current = null;
      }, NAV_MIN_VISIBLE);
    }, NAV_LOADING_DELAY);

    prevPathname.current = pathname ?? null;

    return () => {
      if (navDelayRef.current) {
        clearTimeout(navDelayRef.current);
        navDelayRef.current = null;
      }
      if (navMinRef.current) {
        clearTimeout(navMinRef.current);
        navMinRef.current = null;
      }
    };
  }, [pathname]);

  // Initialize production environment and remove extension classes
  useEffect(() => {
    // This runs only on the client after hydration
    document.body.className = "antialiased";

    // Initialize production configuration and monitoring
    initializeProduction();
  }, []);

  // Check if current route is a public auth route
  const isAuthRoute = pathname?.startsWith('/auth/');

  return (
    <div className="antialiased">
      <AuthGuard>
        {isAuthRoute ? (
          // Don't wrap auth pages with AppLayout
          children
        ) : (
          // Wrap main app with AppLayout
          <AppLayout>
            {children}
          </AppLayout>
        )}
      </AuthGuard>
      {isNavigating && <GlobalLoader message="Loading…" />}
      <Toaster />
    </div>
  );
}
