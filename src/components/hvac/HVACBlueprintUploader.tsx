"use client";

import React from 'react';
import Uploader from '@/components/uploader/Uploader';

interface HVACBlueprintUploaderProps {
  onAnalysisComplete?: (result: any) => void;
}

export default function HVACBlueprintUploader({ onAnalysisComplete }: HVACBlueprintUploaderProps) {
  return (
    <Uploader
      action="analyze"
      showMetadataFields={{ projectId: true, location: true }}
      onAnalyzeComplete={onAnalysisComplete}
    />
  );
}