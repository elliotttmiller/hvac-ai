'use client';

import dynamic from 'next/dynamic';

const DeepZoomInferenceAnalysis = dynamic(
  () => import('@/components/inference/DeepZoomInferenceAnalysis'),
  { 
    ssr: false,
    loading: () => (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4" />
          <p className="text-slate-600">Loading Deep-Zoom Analysis...</p>
        </div>
      </div>
    ),
  }
);

export default function DeepZoomAnalysisPage() {
  return <DeepZoomInferenceAnalysis />;
}


