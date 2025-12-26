'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuoteStore } from '@/lib/pricing-store';
import { generateQuote } from '@/lib/api-client';
import InteractiveInvoice from './InteractiveInvoice';
import InferenceAnalysis from '@/components/inference/InferenceAnalysis';
import { getColorForLabel } from '@/lib/label-colors';
import { Button } from '@/components/ui/button';
import { exportQuoteToCSV } from '@/lib/api-client';
import type { Quote } from '@/lib/pricing-store';
import { FileText } from 'lucide-react';
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
  pricingAvailable?: boolean;
  onBackToUpload?: () => void;
}

export default function AnalysisDashboard({
  projectId,
  location,
  analysisResult,
  imageFile,
  pricingAvailable,
  onBackToUpload
}: QuoteDashboardProps) {
  const { quote, isGenerating, error, setQuote, setGenerating, setError } = useQuoteStore();
  const [viewerExpanded, setViewerExpanded] = useState(true);
  const [hoveredCategory, setHoveredCategory] = useState<string | null>(null);
  // Unified right panel mode: 'quote' shows InteractiveInvoice, 'components' shows component analysis
  const [panelMode, setPanelMode] = useState<'quote' | 'components'>('quote');
  // Sidebar width in pixels (resizable)
  const [sidebarWidth, setSidebarWidth] = useState<number>(420);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(true);
  const isResizing = useRef(false);
  const containerRef = useRef<HTMLDivElement | null>(null);
  type InferenceControl = {
    focusOnCategory: (category: string) => void;
    setFilterCategory: (category: string | null) => void;
    setFilterCategories: (categories: string[] | null) => void;
    setHighlightedCategory: (category: string | null) => void;
  };
  const inferenceRef = useRef<InferenceControl | null>(null);
  // locally track selected filters (multi-select)
  const [selectedFilters, setSelectedFilters] = useState<string[]>([]);

  const startResize = (e: React.MouseEvent) => {
    isResizing.current = true;
    e.preventDefault();
  };

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    // calculate width based on distance from right edge
    const newWidth = Math.max(280, Math.min(800, rect.right - e.clientX));
    setSidebarWidth(newWidth);
  }, []);

  const stopResize = useCallback(() => {
    isResizing.current = false;
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', stopResize);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', stopResize);
    };
  }, [onMouseMove, stopResize]);

  const hasAttemptedGeneration = useRef(false);

  // NOTE: We DO NOT auto-generate quotes on mount. Quoting can be heavy and may
  // rely on a backend pricing subsystem which may not be present in all dev
  // environments. The parent component should pass `pricingAvailable` to
  // control whether the Generate button is enabled.

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

  // Accessibility: keyboard handler for the expand toggle
  const handleToggleKey = (e: React.KeyboardEvent<HTMLButtonElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      setViewerExpanded(v => !v);
    }
  };

  return (
    // Parent decides full-screen behavior. This component is accessible and
    // includes a sticky toolbar/header so controls remain visible while
    // internal content scrolls.
    <div className="flex flex-col h-full w-full bg-gradient-to-br from-slate-50 to-slate-100" role="region" aria-label="HVAC Analysis Studio">
      {/* Sticky Toolbar/Header */}
      <header className="sticky top-0 z-40 bg-white border-b border-slate-200 px-6 py-3 shadow-sm" role="toolbar" aria-label="Analysis Controls">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
                {/* Back to upload / start new analysis */}
                {onBackToUpload && (
                  <button
                    className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-800 mr-3"
                    onClick={onBackToUpload}
                    aria-label="Back to upload"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    <span>Upload another</span>
                  </button>
                )}
            <DollarSign className="h-6 w-6 text-green-600" aria-hidden="true" />
            <div>
              <h1 className="text-xl font-bold">Analyze Dashboard</h1>
              <p className="text-sm text-slate-500">Project: {projectId}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Top-level function executor / quick actions */}
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  // If panel is closed, open it first. Otherwise toggle mode.
                  if (!sidebarOpen) {
                    setSidebarOpen(true);
                    setPanelMode('quote');
                  } else {
                    setPanelMode(m => m === 'quote' ? 'components' : 'quote');
                  }
                }}
                aria-label="Toggle right panel"
              >
                <FileText className="h-4 w-4" />
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  if (quote) exportQuoteToCSV(quote as Quote, projectId);
                }}
                disabled={!quote}
                aria-disabled={!quote}
                aria-label="Export quote CSV"
              >
                Export
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - Split View */}
      <main className="flex-1 flex overflow-hidden" role="main">
        {/* Left Side - Blueprint Visualizer (flexible) */}
        <motion.section
          className="relative bg-slate-900 overflow-auto"
          aria-label="Blueprint Visualizer"
          style={{ flex: '1 1 auto' }}
        >
          {imageFile && analysisResult.segments && (
            <InferenceAnalysis
              ref={inferenceRef}
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
            onKeyDown={handleToggleKey}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            aria-pressed={viewerExpanded}
            aria-label={viewerExpanded ? 'Collapse visualizer' : 'Expand visualizer'}
          >
            {viewerExpanded ? (
              <ChevronLeft className="h-5 w-5 text-slate-700" />
            ) : (
              <ChevronRight className="h-5 w-5 text-slate-700" />
            )}
          </motion.button>

            {/* Open panel pill when sidebar is closed */}
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                aria-label="Open right panel"
                className="absolute right-2 top-1/2 -translate-y-1/2 z-50 h-10 w-10 rounded-full bg-white/90 shadow flex items-center justify-center border border-slate-200"
              >
                <ChevronLeft className="h-4 w-4 text-slate-700" />
              </button>
            )}
        </motion.section>

          {/* Divider / Resizer (only shown when sidebar open). Removed hover background to avoid interfering with the viewer pointer events. */}
          {sidebarOpen && (
            <div
              role="separator"
              aria-orientation="vertical"
              aria-label="Resize panel"
              onMouseDown={startResize}
              className="w-2 cursor-col-resize bg-transparent z-40"
              style={{ cursor: 'col-resize' }}
            />
          )}

        {/* Right Side - Unified Switchable Panel (resizable) */}
        {sidebarOpen && (
          <aside
            className="overflow-auto bg-white p-0"
            style={{ width: sidebarWidth, minWidth: 280, maxWidth: 800 }}
            aria-label="Right panel"
          >
          <div className="p-4 border-b border-slate-200 flex items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <button
                className={`px-3 py-1 rounded ${panelMode === 'quote' ? 'bg-slate-100' : 'hover:bg-slate-50'}`}
                onClick={() => setPanelMode('quote')}
                aria-pressed={panelMode === 'quote'}
                aria-label="Show Quote Panel"
              >
                <DollarSign className="h-4 w-4 inline-block mr-2" />
                Quote
              </button>

              <button
                className={`px-3 py-1 rounded ${panelMode === 'components' ? 'bg-slate-100' : 'hover:bg-slate-50'}`}
                onClick={() => setPanelMode('components')}
                aria-pressed={panelMode === 'components'}
                aria-label="Show Components Panel"
              >
                <FileText className="h-4 w-4 inline-block mr-2" />
                Components
              </button>
            </div>

            <div className="flex items-center gap-2">
              {/* Generate button kept here for convenience */}
              {!quote && !isGenerating && (
                <Button
                  onClick={handleGenerateQuote}
                  className="gap-2"
                  disabled={typeof pricingAvailable !== 'undefined' ? !pricingAvailable : false}
                  aria-disabled={typeof pricingAvailable !== 'undefined' ? !pricingAvailable : undefined}
                  aria-label={pricingAvailable === false ? 'Pricing unavailable' : 'Generate quote'}
                >
                  <DollarSign className="h-4 w-4" aria-hidden="true" />
                  {pricingAvailable === false ? 'Pricing Unavailable' : 'Generate Quote'}
                </Button>
              )}
              {/* Collapse panel (floating hide) button - closes the right panel */}
              <button
                onClick={() => setSidebarOpen(false)}
                aria-label="Close right panel"
                title="Close panel"
                className="ml-2 inline-flex items-center justify-center h-8 w-8 rounded bg-white hover:bg-slate-50 border border-slate-100"
              >
                <ChevronRight className="h-4 w-4 text-slate-600" />
              </button>
            </div>
          </div>

          <div className="p-4">
            {isGenerating && (
              <Card className="h-40 flex items-center justify-center">
                <div className="text-center space-y-4">
                  <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto" aria-hidden="true" />
                  <p className="text-slate-600">Generating quote...</p>
                </div>
              </Card>
            )}

            {error && !isGenerating && (
              <Alert variant="destructive" role="alert">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {pricingAvailable === false && (
              <Alert variant="destructive" role="status">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>Pricing subsystem unavailable — quoting is disabled in this environment.</AlertDescription>
              </Alert>
            )}

            {/* Panel body */}
            {panelMode === 'quote' ? (
              <div className="mt-4">
                <InteractiveInvoice projectId={projectId} onCategoryHover={setHoveredCategory} />
              </div>
            ) : (
              <div className="mt-2 space-y-2">
                <h3 className="text-sm font-semibold">Detected Components</h3>
                <div className="text-sm text-slate-600">Summary of detected categories and counts</div>
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-sm text-slate-500">Choose one or more to filter the visualizer</div>
                    {selectedFilters.length > 0 && (
                      <button
                        className="text-xs text-slate-600 hover:text-slate-900 underline"
                        onClick={() => {
                          setSelectedFilters([]);
                          inferenceRef.current?.setFilterCategories(null);
                        }}
                        aria-label="Clear filters"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  {(analysisResult.counts_by_category && Object.keys(analysisResult.counts_by_category).length > 0) ? (
                    <ul className="divide-y divide-slate-100">
                      {Object.entries(analysisResult.counts_by_category).map(([cat, count]) => {
                        const selected = selectedFilters.includes(cat);
                        return (
                          <motion.li
                            key={cat}
                            className="py-2 flex justify-between items-center"
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0, scale: selected ? 1.01 : 1 }}
                            whileHover={{ scale: 1.02 }}
                          >
                            <button
                              className={`flex items-center gap-3 text-left w-full relative transition-all duration-150 ${selected ? 'bg-slate-50 rounded-md' : 'hover:bg-slate-50'}`}
                              onClick={() => {
                                setSelectedFilters(prev => {
                                  let next: string[];
                                  if (prev.includes(cat)) {
                                    next = prev.filter(p => p !== cat);
                                  } else {
                                    next = [...prev, cat];
                                  }
                                  // inform visualizer of the new active set (null === no filter)
                                  inferenceRef.current?.setFilterCategories(next.length ? next : null);
                                  // if newly selected, focus the visualizer on that category
                                  if (!prev.includes(cat)) inferenceRef.current?.focusOnCategory(cat);
                                  return next;
                                });
                              }}
                              onMouseEnter={() => inferenceRef.current?.setHighlightedCategory(cat)}
                              onMouseLeave={() => inferenceRef.current?.setHighlightedCategory(null)}
                              aria-pressed={selected}
                            >
                              <span style={{ width: 10, height: 10, backgroundColor: getColorForLabel(cat), borderRadius: 2 }} />
                              <span className="capitalize">{cat.replace(/_/g, ' ')}</span>
                              {selected && (
                                <span className="absolute inset-0 bg-gradient-to-r from-transparent to-transparent pointer-events-none" aria-hidden />
                              )}
                            </button>
                            <span className="font-mono">{count}</span>
                          </motion.li>
                        );
                      })}
                    </ul>
                  ) : (
                    <div className="text-sm text-slate-500">No component counts available.</div>
                  )}
                </div>
              </div>
            )}
          </div>
          </aside>
        )}
      </main>
    </div>
  );
}
