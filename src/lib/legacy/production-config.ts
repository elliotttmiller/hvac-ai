// Minimal production-config shim for legacy blueprint analyzer
// The real implementation may be project-specific; this shim provides
// the exported functions/types used by the legacy analyzer so TypeScript
// checks and imports succeed.

export function getConfig() {
  return {
    features: {
      realTimeOCR: false,
      advancedCV: false,
      progressiveEnhancement: false,
      blueprintAnalysis: false,
    }
  };
}

export const PerformanceMonitor = {
  startTimer: (_: string) => undefined as unknown,
  endTimer: (_: string) => 0,
  logPerformance: (_: string, __: number, ___: object) => undefined,
};

export const ErrorTracker = {
  trackError: (_: string, __: Error, ___?: object) => undefined,
};

export const CacheManager = {
  get: <T>(_key: string): T | null => null,
  set: (_key: string, _value: unknown, _ttl?: number) => undefined,
};

export function getPerformanceSetting(_: string) {
  // sensible defaults
  return 1024 * 1024 * 50; // 50MB
}

export function getQualitySetting(_: string) {
  return 'standard';
}

const _productionConfigShim = {};
export default _productionConfigShim;
