import HVACStudio from '@/components/features/HVACStudio';
import { Suspense } from 'react';

interface PageProps {
  params: Promise<{ id: string }>;
  searchParams?: Promise<{ [key: string]: string | string[] | undefined }>;
}

async function WorkspaceContent({ params, searchParams }: PageProps) {
  const resolvedParams = await params;
  const resolvedSearchParams = await searchParams;
  const testMode = resolvedSearchParams?.test === 'true';

  return (
    <div className="h-full w-full">
      <HVACStudio projectId={resolvedParams.id} testMode={testMode} />
    </div>
  );
}

export default function WorkspacePage(props: PageProps) {
  return (
    <Suspense fallback={<div className="h-full w-full flex items-center justify-center">Loading workspace...</div>}>
      <WorkspaceContent {...props} />
    </Suspense>
  );
}