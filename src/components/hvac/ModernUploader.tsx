'use client';

import React from 'react';
import Uploader from '@/components/uploader/Uploader';

interface ModernUploaderProps {
  onFileSelect?: (file: File) => void;
}

export default function ModernUploader({ onFileSelect }: ModernUploaderProps) {
  return (
    <Uploader
      action="analyze"
      onAnalyzeComplete={(res) => { if (onFileSelect && res?.file_name) { /* no-op: adapter */ } }}
      showMetadataFields={false}
    />
  );
}
