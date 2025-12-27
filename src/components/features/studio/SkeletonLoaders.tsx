'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className, ...props }: SkeletonProps) {
  return (
    <div
      className={cn('animate-pulse rounded-md bg-muted', className)}
      {...props}
    />
  );
}

export function ComponentTreeSkeleton() {
  return (
    <div className="p-2 space-y-2">
      {[1, 2, 3, 4].map((group) => (
        <div key={group} className="space-y-1">
          {/* Group Header Skeleton */}
          <div className="flex items-center gap-2 p-2">
            <Skeleton className="h-4 w-4" />
            <Skeleton className="h-4 flex-1" />
            <Skeleton className="h-4 w-6" />
          </div>
          
          {/* Group Items Skeleton */}
          <div className="ml-6 space-y-1">
            {[1, 2, 3].map((item) => (
              <div key={item} className="flex items-center gap-2 p-2">
                <Skeleton className="h-3 flex-1" />
                <Skeleton className="h-3 w-10" />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function InspectorPanelSkeleton() {
  return (
    <div className="p-4 space-y-4">
      {/* Header Skeleton */}
      <div className="space-y-2">
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-5 w-24" />
      </div>
      
      {/* Sections Skeleton */}
      {[1, 2, 3].map((section) => (
        <div key={section} className="space-y-2">
          <Skeleton className="h-8 w-full" />
          <div className="space-y-2 pl-4">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <Skeleton className="h-4 w-4/6" />
          </div>
        </div>
      ))}
      
      {/* Action Buttons Skeleton */}
      <div className="space-y-2 pt-4">
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </div>
    </div>
  );
}

export function ViewerSkeleton() {
  return (
    <div className="relative w-full h-full bg-slate-900 flex items-center justify-center">
      <motion.div
        animate={{
          scale: [1, 1.1, 1],
          opacity: [0.5, 0.8, 0.5],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="text-center"
      >
        <div className="relative w-24 h-24 mx-auto mb-4">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{
              duration: 2,
              repeat: Infinity,
              ease: 'linear',
            }}
            className="absolute inset-0 rounded-full border-4 border-primary/20 border-t-primary"
          />
        </div>
        <p className="text-sm text-muted-foreground">Analyzing blueprint...</p>
      </motion.div>
    </div>
  );
}
