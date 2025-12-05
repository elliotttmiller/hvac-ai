"use client";

import { useSession } from "next-auth/react";
import { useRouter, usePathname } from "next/navigation";
import { useEffect, useState, useRef, ReactNode } from "react";
import { Loader2, Zap } from "lucide-react";
import GlobalLoader from "@/components/ui/GlobalLoader";

interface AuthGuardProps {
  children: ReactNode;
}

interface DirectUser {
  id: string;
  email: string;
  name: string;
  role: string;
  department: string;
  permissions: string[];
}

export default function AuthGuard({ children }: AuthGuardProps) {
  const { data: session, status } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  const [directUser, setDirectUser] = useState<DirectUser | null>(null);
  const [isCheckingDirectAuth, setIsCheckingDirectAuth] = useState(true);

  // Public routes that don't require authentication
  const publicRoutes = ['/auth/signup', '/auth/error'];
  const isPublicRoute = publicRoutes.includes(pathname);

  // Check for direct authentication on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        const storedUser = localStorage.getItem('user');
        const authMethod = localStorage.getItem('authMethod');

        if (storedUser && authMethod === 'direct') {
          const user = JSON.parse(storedUser);
          setDirectUser(user);
          console.log('🔐 Direct auth user found:', user.email);
        }
      } catch (error) {
        console.error('Error reading direct auth from localStorage:', error);
        // Clear invalid data
        localStorage.removeItem('user');
        localStorage.removeItem('authMethod');
      }
    }
    setIsCheckingDirectAuth(false);
  }, []);

  // Determine if user is authenticated (either NextAuth or direct)
  const isAuthenticated = session || directUser;
  const isLoading = status === 'loading' || isCheckingDirectAuth;

  useEffect(() => {
    if (isLoading) return; // Still loading

    // Only enforce redirects when explicitly enabled (e.g., in production).
    // Use NEXT_PUBLIC_ENFORCE_AUTH=true to enable enforcement.
    const enforceAuth = process.env.NEXT_PUBLIC_ENFORCE_AUTH === 'true';

    if (!isAuthenticated && !isPublicRoute) {
      if (enforceAuth) {
        // Redirect to home page if not authenticated
        console.log('🔒 Not authenticated, redirecting to home (enforced)');
        router.push('/');
      } else {
        console.log('🔒 Not authenticated, auth enforcement disabled - not redirecting');
      }
    } else if (isAuthenticated && isPublicRoute) {
      if (enforceAuth) {
        // Redirect to dashboard if already authenticated and on public route
        console.log('✅ Already authenticated, redirecting to dashboard (enforced)');
        router.push('/');
      }
    }
  }, [isAuthenticated, isLoading, router, pathname, isPublicRoute]);

  // Show loading screen while checking authentication
  // Loader timing: delay before showing and minimum visible time to avoid flashes
  const AUTH_LOADING_DELAY = 120; // ms
  const AUTH_MIN_VISIBLE = 420; // ms

  const [showAuthLoader, setShowAuthLoader] = useState(false);
  const authDelayRef = useRef<number | null>(null);
  const authMinRef = useRef<number | null>(null);

  useEffect(() => {
    if (isLoading) {
      // start delay to show loader
      if (authDelayRef.current) clearTimeout(authDelayRef.current);
      authDelayRef.current = window.setTimeout(() => {
        authDelayRef.current = null;
        setShowAuthLoader(true);
        // ensure minimum visible time
        if (authMinRef.current) clearTimeout(authMinRef.current);
        authMinRef.current = window.setTimeout(() => {
          authMinRef.current = null;
          // will be hidden when isLoading becomes false
        }, AUTH_MIN_VISIBLE);
      }, AUTH_LOADING_DELAY);
    } else {
      // hide loader but respect minimum visible time
      if (authDelayRef.current) {
        clearTimeout(authDelayRef.current);
        authDelayRef.current = null;
        setShowAuthLoader(false);
      } else if (showAuthLoader) {
        // if currently shown, wait for min visible timer or hide immediately
        if (authMinRef.current) {
          // schedule hide after remaining time
          const remaining = AUTH_MIN_VISIBLE; // min timer ensures it's not too quick
          if (authMinRef.current) {
            clearTimeout(authMinRef.current);
            authMinRef.current = window.setTimeout(() => {
              authMinRef.current = null;
              setShowAuthLoader(false);
            }, remaining);
          }
        } else {
          setShowAuthLoader(false);
        }
      }
    }

    return () => {
      if (authDelayRef.current) {
        clearTimeout(authDelayRef.current);
        authDelayRef.current = null;
      }
      if (authMinRef.current) {
        clearTimeout(authMinRef.current);
        authMinRef.current = null;
      }
    };
  }, [isLoading, showAuthLoader]);

  if (showAuthLoader) {
    return <GlobalLoader message="Loading HVACAI..." />;
  }

  // During development we no longer redirect to a sign-in page.
  // Render children even when not authenticated to make local development faster.
  if (!isAuthenticated && !isPublicRoute) {
    return <>{children}</>;
  }

  return <>{children}</>;
}
