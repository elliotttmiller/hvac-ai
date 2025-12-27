'use client';

import dynamic from 'next/dynamic';

// Dynamically import the dashboard content to avoid SSR issues
const DashboardContent = dynamic(() => import('./DashboardContent'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-screen bg-slate-950">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
    </div>
  ),
});

export default function DashboardPage() {
  return <DashboardContent />;
}

