'use client';
import AuthGuard from '@/components/shared/AuthGuard';
import { Button } from '@/components/ui/button';
import { ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthGuard>
      <div className="h-screen w-full bg-background flex flex-col">
        {/* Simple header for navigation */}
        <header className="h-16 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 flex items-center px-4">
          <Link href="/projects">
            <Button variant="ghost" size="sm">
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Projects
            </Button>
          </Link>
          <div className="ml-auto">
            <h1 className="text-lg font-semibold">HVAC Studio</h1>
          </div>
        </header>

        {/* Main content area */}
        <div className="flex-1 overflow-hidden">
          {children}
        </div>
      </div>
    </AuthGuard>
  );
}