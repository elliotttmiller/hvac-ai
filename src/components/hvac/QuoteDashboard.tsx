'use client';

import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuoteStore } from '@/lib/pricing-store';
import { generateQuote } from '@/lib/api-client';
import InteractiveInvoice from './InteractiveInvoice';
import InferenceAnalysis from '@/components/inference/InferenceAnalysis';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  ChevronRight,
  ChevronLeft,
  Loader2,
  AlertCircle,
  DollarSign
} from 'lucide-react';
import type { AnalysisResult } from '@/types/analysis';

interface QuoteDashboardProps {
  projectId: string;
  location: string;
  analysisResult: AnalysisResult;
  imageFile: File | null;
}

export default function QuoteDashboard({
  projectId,
  location,
  analysisResult,
  imageFile
}: QuoteDashboardProps) {
  const { quote, isGenerating, error, setQuote, setGenerating, setError } = useQuoteStore();
  const [viewerExpanded, setViewerExpanded] = useState(true);
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  
  const hasAttemptedGeneration = useRef(false);
  
  // Auto-generate quote on mount
  useEffect(() => {
    if (!hasAttemptedGeneration.current && !quote && analysisResult && !isGenerating) {
      hasAttemptedGeneration.current = true;
      handleGenerateQuote();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  const handleGenerateQuote = async () => {
    setGenerating(true);
    setError(null);
    
    try {
      const response = await generateQuote({
        project_id: projectId,
        location: location || 'Atlanta, GA',
        analysis_data: {
          total_objects: analysisResult.total_objects_found || 0,
          counts_by_category: analysisResult.counts_by_category || {}
        }
      });
      
      setQuote(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate quote');
      console.error('Quote generation error:', err);
    } finally {
      setGenerating(false);
    }
  };
  
  return (
    <div className="h-screen w-full flex flex-col bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header Bar */}
      <div className="bg-white border-b border-slate-200 px-6 py-4 shadow-sm">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <DollarSign className="h-6 w-6 text-green-600" />
            <div>
              <h1 className="text-xl font-bold">Quote Dashboard</h1>
              <p className="text-sm text-slate-500">Project: {projectId}</p>
            </div>
          </div>
          
          {!quote && !isGenerating && (
            <Button onClick={handleGenerateQuote} className="gap-2">
              <DollarSign className="h-4 w-4" />
              Generate Quote
            </Button>
          )}
        </div>
      </div>
      
      {/* Main Content - Split View */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Side - Blueprint Visualizer */}
        <motion.div
          className="relative bg-slate-900 overflow-hidden"
          initial={{ width: '60%' }}
          animate={{ width: viewerExpanded ? '60%' : '40%' }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
        >
          {imageFile && analysisResult.segments && (
            <InferenceAnalysis
              initialImage={imageFile}
              initialSegments={analysisResult.segments}
              initialCount={{
                total_objects_found: analysisResult.total_objects_found || 0,
                counts_by_category: analysisResult.counts_by_category || {}
              }}
            />
          )}
          
          {/* Expand/Collapse Toggle */}
          <motion.button
            className="absolute right-0 top-1/2 -translate-y-1/2 z-50 bg-white/90 backdrop-blur-md hover:bg-white shadow-lg rounded-l-lg p-2 border border-r-0 border-slate-200"
            onClick={() => setViewerExpanded(!viewerExpanded)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {viewerExpanded ? (
              <ChevronLeft className="h-5 w-5 text-slate-700" />
            ) : (
              <ChevronRight className="h-5 w-5 text-slate-700" />
            )}
          </motion.button>
        </motion.div>
        
        {/* Right Side - Interactive Invoice */}
        <motion.div
          className="flex-1 overflow-hidden p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          {isGenerating && (
            <Card className="h-full flex items-center justify-center">
              <div className="text-center space-y-4">
                <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto" />
                <p className="text-slate-600">Generating quote...</p>
              </div>
            </Card>
          )}
          
          {error && !isGenerating && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          
          {!isGenerating && !error && (
            <InteractiveInvoice
              projectId={projectId}
              onCategoryHover={setHoveredCategory}
            />
          )}
        </motion.div>
      </div>
    </div>
  );
}
