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
    <div className="h-full w-full overflow-hidden">
      <HVACStudio projectId={resolvedParams.id} testMode={testMode} />
    </div>
  );
}

export default function WorkspacePage(props: PageProps) {
  return (
    <Suspense 
      fallback={
        <div className="h-full w-full flex items-center justify-center bg-slate-950">
          <div className="text-center space-y-3">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="text-slate-400 text-sm">Loading workspace...</p>
          </div>
        </div>
      }
    >
      <WorkspaceContent {...props} />
    </Suspense>
  );
}