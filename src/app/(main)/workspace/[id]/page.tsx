'use client';

import dynamic from 'next/dynamic';

const DashboardContent = dynamic(() => import('../../dashboard/DashboardContent'), {
  ssr: false,
});

interface WorkspacePageProps {
  params: {
    id: string;
  };
}

export default function WorkspacePage({ params }: WorkspacePageProps) {
  const { id } = params;

  // TODO: Add authentication check
  // TODO: Add project validation

  return <DashboardContent projectId={id} mode="workspace" />;
}