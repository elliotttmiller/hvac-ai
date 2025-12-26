'use client';

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, Image as ImageIcon } from 'lucide-react';

interface ModernUploaderProps {
  onFileSelect: (file: File) => void;
  acceptedFileTypes?: Record<string, string[]>;
  maxSize?: number;
}

export default function ModernUploader({
  onFileSelect,
  acceptedFileTypes = {
    'image/*': ['.png', '.jpg', '.jpeg', '.webp'],
    'application/pdf': ['.pdf'],
  },
  maxSize = 500 * 1024 * 1024 // 500MB
}: ModernUploaderProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxSize,
    multiple: false
  });

  return (
    <div
      {...getRootProps()}
      className={`
        relative overflow-hidden rounded-2xl border-2 border-dashed cursor-pointer
        transition-all duration-300 ease-out
        ${isDragActive
          ? 'border-blue-500 bg-blue-50/50 scale-[1.02]'
          : 'border-slate-200 bg-slate-50/50 hover:border-slate-300 hover:bg-slate-100/50'
        }
      `}
    >
      <input {...getInputProps()} />
      
      {/* Glassmorphism Background */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/80 to-slate-100/80 backdrop-blur-sm" />
      
      {/* Content */}
      <div className="relative p-12 text-center">
        {/* Icon */}
        <motion.div
          className="mx-auto w-20 h-20 mb-6 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-lg"
          animate={{
            scale: isDragActive ? [1, 1.1, 1] : 1,
          }}
          transition={{ duration: 0.3 }}
        >
          {isDragActive ? (
            <ImageIcon className="h-10 w-10 text-white" />
          ) : (
            <Upload className="h-10 w-10 text-white" />
          )}
        </motion.div>
        
        {/* Text */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h3 className="text-xl font-semibold text-slate-800 mb-2">
            {isDragActive ? 'Drop it like it\'s hot!' : 'Upload Blueprint'}
          </h3>
          <p className="text-sm text-slate-600 mb-4">
            Drag & drop or click to browse
          </p>
          <div className="flex items-center justify-center gap-4 text-xs text-slate-500">
            <div className="flex items-center gap-1">
              <FileText className="h-3 w-3" />
              <span>PDF, DWG, DXF</span>
            </div>
            <div className="flex items-center gap-1">
              <ImageIcon className="h-3 w-3" />
              <span>PNG, JPG, WEBP</span>
            </div>
          </div>
        </motion.div>
      </div>
      
      {/* Animated Border Gradient on Drag */}
      {isDragActive && (
        <motion.div
          className="absolute inset-0 rounded-2xl"
          style={{
            background: 'linear-gradient(90deg, #3b82f6, #8b5cf6, #3b82f6)',
            backgroundSize: '200% 100%',
          }}
          animate={{
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          }}
          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: 'linear' }}
        />
      )}
    </div>
  );
}
