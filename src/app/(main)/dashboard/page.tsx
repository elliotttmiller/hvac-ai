'use client';

import dynamic from 'next/dynamic';
import { AppShell } from '@/components/layout/AppShell';
import { LayoutDashboard, Building, FileText, Wind, DollarSign } from 'lucide-react';

// Dynamically import the dashboard content to avoid SSR issues
const DashboardContent = dynamic(() => import('./DashboardContent'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-64">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
  ),
});

const navigation = [
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
  {
    name: 'Analysis',
    href: '/analysis',
    icon: Wind,
    description: 'AI-powered analysis',
  },
  {
    name: 'Quotes',
    href: '/quotes',
    icon: DollarSign,
    description: 'Pricing and estimates',
  },
];

export default function DashboardPage() {
  return (
    <AppShell navigation={navigation}>
      <DashboardContent />
    </AppShell>
  );
}

