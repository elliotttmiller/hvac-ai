# Legacy Code

This directory contains code that is currently unused but preserved for potential future reference.

## Files

### Blueprint Analyzers

- `blueprint-analyzer.ts` - Original blueprint analysis interfaces and types
- `blueprint-analyzer-production.ts` - Production-ready blueprint analyzer implementation

These files are not currently imported or used in the application. They may be useful as reference for future blueprint analysis features or can be safely removed if not needed.

## Maintenance

Files in this directory should be:
- Reviewed periodically for relevance
- Removed if no longer needed
- Moved back to active codebase if required for new features

To use any of these files:
1. Move them back to `src/lib/`
2. Update imports in consuming components
3. Test thoroughly before deploying
