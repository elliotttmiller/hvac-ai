# Professional UI Redesign - Interactive Blueprint Viewer

## Overview
Redesigned the HVAC Blueprint Analysis viewer (`DashboardContent.tsx`) from a basic 2-column grid layout to an industry-standard professional viewer with full-page no-scroll layout, sidebar tabs, and on-demand quote generation.

## Architecture Changes

### Layout Transformation
**Before:** 2-column grid (70% viewport + 30% quote builder)
- Simple card-based layout
- Scrollable content
- Auto-generating quotes
- Step-indicator flow

**After:** Professional full-page viewer
- Left panel: Interactive image viewport (70% of screen width)
- Right sidebar: Tabbed interface (30% of screen width, fixed position)
- No scrolling - all content fits in viewport
- Manual button-triggered quote generation
- Professional dark theme with slate/emerald color scheme

### Component Architecture
```
DashboardContent (main container - full h-screen, no scroll)
├── Upload Screen (if no file)
│   ├── Header (minimal, gradient background)
│   └── Upload Area (centered FileUploader)
│
└── Analysis Screen (if file selected)
    ├── Top Bar (minimal, dark theme)
    │   ├── Back button + File info
    │   ├── Generate Quote button (when detections exist)
    │   └── New Upload button
    │
    └── Main Content (full-height flex container)
        ├── Left Panel: Viewport (70%)
        │   └── InteractiveViewer (full height, black background)
        │
        └── Right Sidebar: Tabs (30%, fixed)
            ├── TabsList (4 tabs)
            │   ├── Viewport (Eye icon)
            │   ├── Analysis (BarChart icon)
            │   ├── Quote (Calculator icon)
            │   └── Settings (Gear icon) [TBD]
            │
            └── TabsContent panels
                ├── Viewport Tab
                │   ├── File info (name, size, type)
                │   └── Viewport controls
                │
                ├── Analysis Tab
                │   ├── Detection summary (count)
                │   └── Component list (with confidence scores, count badges)
                │
                └── Quote Tab
                    └── QuoteBuilder component (only shown when quote generated)
```

## State Management

### Key State Variables
```typescript
// Upload management
const [file, setFile] = useState<File | null>();
const [isProcessing, setIsProcessing] = useState(false);

// Analysis data (detections & quote)
const [analysisData, setAnalysisData] = useState<{
  detections?: OverlayItem[];
  quote?: QuoteData;
} | null>();

// UI tabs & quote visibility
const [activeTab, setActiveTab] = useState<string>('viewport');
const [showQuoteEngine, setShowQuoteEngine] = useState(false);
```

## Event Flows

### Upload Flow
1. User uploads file via FileUploader
2. `handleUpload()` called with File object
3. FormData sent to `/api/analysis` POST endpoint
4. Backend analyzes image, returns detections
5. Parse detections into OverlayItem[] format
6. Update analysisData with detections
7. Auto-switch to 'viewport' tab (shows image + overlays)
8. Quote engine remains hidden (showQuoteEngine = false)

### Quote Generation Flow
1. User clicks "Generate Quote" button (enabled when detections exist)
2. Button triggers `handleGenerateQuote()` function
3. Creates QuoteData from analysisData.detections
4. Updates analysisData with quote object
5. Sets showQuoteEngine = true
6. Sets activeTab = 'quote'
7. QuoteBuilder rendered in Quote tab with cost estimate

### Tab Navigation Flow
1. User clicks tab in sidebar (Viewport, Analysis, Quote, Settings)
2. setActiveTab(tabValue) updates state
3. Tabs component shows/hides TabsContent based on activeTab
4. Only active tab rendered (uses data-[state=inactive]:hidden)

## Styling Strategy

### Color Scheme
- **Dark Theme**: Slate-50 to Slate-950 (dark UI following modern conventions)
- **Accent Color**: Emerald-600 (for buttons, highlights)
- **Borders**: Slate-800 (subtle divisions)
- **Text**: Slate-200 (light text on dark backgrounds)

### Layout Constraints
- **Fixed Viewport Heights**: 
  - Header: py-4 (fixed height)
  - Main content: flex-1 (fills remaining space)
  - Sidebar: Tabs with fixed widths and auto-scrolling tabs content
  
- **No Scrolling**: 
  - Use `overflow-hidden` on parent containers
  - Individual tab contents use `overflow-auto` for their content only
  - Sidebar tabs scroll independently

- **Responsive Design**:
  - Sidebar fixed at 360px width
  - Viewport takes remaining width
  - Tabs use grid layout for responsive list
  - All text uses appropriate text-xs/sm/base sizing

### Component Theming
- **Cards**: bg-slate-800 with slate-950 headers
- **Badges**: secondary variant (teal backgrounds)
- **Progress Bars**: emerald-500 for fill, slate-700 for track
- **Buttons**: 
  - Primary actions: emerald-600 hover:emerald-700
  - Secondary: variant="outline" with slate colors
  - Ghost: transparent with slate text

## UI Elements

### Upload Screen
- **Hero Section**: Centered, max-width container
- **FileUploader**: Drag-and-drop interface with process controls
- **Gradient Background**: Subtle from-slate-50 to-slate-100

### Analysis Screen - Top Bar
- **Left Section**: Back button (if onBack provided) + file name + detection count
- **Right Section**: 
  - "Generate Quote" button (emerald, shows Zap icon, disabled until quote generated)
  - "New Upload" button (outline variant)

### Analysis Screen - Viewport Panel
- **Container**: Black background (bg-black) rounded corners
- **Content**: Full-height InteractiveViewer with detection overlays
- **Interaction**: Supports pan, zoom via ViewportControls (accessed in sidebar)

### Analysis Screen - Sidebar Tabs
- **Tab List**: 
  - 4 tabs with icons + labels
  - Dark background (bg-slate-800)
  - Active tab highlighted (emerald border)
  
- **Viewport Tab**: 
  - File metadata (name, size, type)
  - ViewportControls component
  
- **Analysis Tab**:
  - Detection summary card (emerald background, shows count)
  - Scrollable component list (max-height with overflow)
  - Each component shows:
    - Label (capitalized)
    - Count badge (e.g., "3x")
    - Confidence progress bar
    - Confidence percentage text
  
- **Quote Tab**:
  - QuoteBuilder component (only when showQuoteEngine=true)
  - Empty state with icon + message when no quote generated
  
- **Settings Tab**: Placeholder (TBD)

## Event Handlers

### `handleUpload(uploadedFile: File)`
- **Purpose**: Process uploaded file through analysis pipeline
- **Steps**:
  1. Set file state
  2. Set isProcessing = true
  3. Create FormData with file, projectId, category='blueprint'
  4. POST to /api/analysis endpoint
  5. Parse response.analysis.detections or response.detections
  6. Convert to OverlayItem[] format
  7. setAnalysisData with detections
  8. setActiveTab('viewport') - show results immediately
  9. Handle errors gracefully
  10. Finally: setIsProcessing = false

### `handleGenerateQuote()`
- **Purpose**: Create cost estimate from detected components
- **Steps**:
  1. Guard: return if no detections
  2. Create QuoteData object from analysisData.detections
  3. Generate quote_id from projectId or timestamp
  4. Create line_items: one per detection with unit/total cost
  5. Calculate summary: subtotal materials, labor, tax, final
  6. Update analysisData with quote object
  7. setShowQuoteEngine = true
  8. setActiveTab('quote') - switch to quote tab

### `setActiveTab(tabValue: string)`
- **Purpose**: Switch which sidebar tab is visible
- **Values**: 'viewport' | 'analysis' | 'quote' | 'settings'
- **Effect**: Triggers Tabs component to show/hide TabsContent

## API Integration

### Upload Endpoint: POST /api/analysis
**Request:**
```typescript
FormData {
  file: File,           // Image file to analyze
  projectId?: string,   // Optional project ID
  category: string      // 'blueprint'
}
```

**Response:**
```typescript
{
  analysis: {
    detections: [
      {
        id: string,
        label: string,           // Component type (e.g., 'ductwork', 'condenser')
        confidence: number,      // 0-1 score
        score: number,          // Alternative confidence field
        bbox: [x, y, w, h],     // Bounding box
        text?: string,          // OCR text
        textConfidence?: number // OCR confidence
      }
    ]
  }
}
```

## File Structure
- **Component**: `src/app/(main)/dashboard/DashboardContent.tsx`
- **Dependencies**:
  - Components: FileUploader, InteractiveViewer, ViewportControls, QuoteBuilder
  - UI Library: Card, Badge, Separator, Button, Tabs (shadcn/ui)
  - Icons: Lucide React (ArrowLeft, Eye, BarChart3, Calculator, Upload, Zap)

## Behavioral Changes

### Quote Generation (BREAKING CHANGE)
**Before**: Quotes auto-generated on analysis completion (automatic, instant)
**After**: Quotes only generated when user clicks "Generate Quote" button (on-demand, manual)

**Benefits**:
- Users can analyze without creating quotes
- Faster initial analysis display (no quote generation latency)
- Quote generation is explicit user action
- Better performance (quote calculation only when needed)

### Tab-Based Navigation
**Before**: Single scrollable page with all content visible
**After**: Sidebar tabs allow focused viewing of specific information

**Benefits**:
- Cleaner, less cluttered interface
- Better information hierarchy
- Professional appearance (CAD viewer style)
- Sidebar can expand/collapse in future iterations

## Future Enhancements

### Phase 2 (Planned)
- [ ] Settings tab with viewport zoom/pan controls
- [ ] Collapse/expand sidebar with animation
- [ ] Keyboard shortcuts for tab navigation (1-4 keys)
- [ ] Export quote to PDF
- [ ] Component filtering in Analysis tab
- [ ] Detection confidence threshold slider

### Phase 3 (Planned)
- [ ] Real-time analysis progress indicator
- [ ] Batch analysis for multiple files
- [ ] Component customization in quote (edit costs, add/remove items)
- [ ] Quote templates/presets
- [ ] Multi-blueprint comparison view

## Testing Checklist

### Functionality
- [ ] Upload file → detections appear in viewport
- [ ] Viewport tab shows image with overlays
- [ ] Analysis tab shows detection list with counts and confidence
- [ ] Generate Quote button disabled until analysis complete
- [ ] Clicking Generate Quote shows cost estimate in Quote tab
- [ ] Quote data calculates correctly from detections
- [ ] New Upload button clears state and shows upload screen
- [ ] Back button navigates to previous page

### UI/UX
- [ ] No horizontal scrolling in any viewport
- [ ] No vertical page scroll (fixed layout)
- [ ] Sidebar tabs responsive to click/keyboard
- [ ] Tab content scrolls independently if needed
- [ ] Upload screen centered and responsive
- [ ] Dark theme consistent across all tabs
- [ ] Buttons have proper hover states
- [ ] Icons render correctly on all tabs

### Responsive
- [ ] Works on 1920x1080 (standard)
- [ ] Works on 1280x720 (small desktop)
- [ ] Works on tablet (future, may need sidebar collapse)
- [ ] Touch-friendly tap targets

### Performance
- [ ] File upload doesn't freeze UI (isProcessing state)
- [ ] Tab switching is instant (no lag)
- [ ] Quote generation completes in <1s
- [ ] No unnecessary re-renders (proper React.memo usage)

## Code Quality Notes

### Design Patterns
- **Controlled Components**: All state managed in DashboardContent
- **Unidirectional Data Flow**: Props flow down, events bubble up
- **Separation of Concerns**: Analysis logic in handlers, rendering in JSX
- **Error Boundaries**: Try/catch in handleUpload with user feedback

### Performance Considerations
- **Lazy Tab Content**: Only active tab rendered (via Tabs component)
- **URL.createObjectURL**: Efficient blob URL for file preview
- **Memoization**: Consider React.memo for tab contents in high-traffic views
- **Async Handling**: isProcessing flag prevents UI blocking during API calls

### Accessibility
- **Icons + Labels**: All buttons have both icons and text labels
- **Tab Navigation**: Standard Tabs component with keyboard support
- **Semantic HTML**: Proper heading hierarchy
- **Color Contrast**: Dark theme tested for WCAG compliance
- **Focus Indicators**: Built into shadcn/ui components

### Browser Compatibility
- **Next.js 15.5.9**: Latest version with Turbopack
- **React 19**: Latest React version
- **Tailwind CSS 3**: Responsive design utilities
- **Target Browsers**: Chrome, Firefox, Safari, Edge (latest 2 versions)

## References
- [Shadcn/ui Tabs Component](https://ui.shadcn.com/docs/components/tabs)
- [Tailwind CSS Responsive Design](https://tailwindcss.com/docs/responsive-design)
- [React Hooks Best Practices](https://react.dev/reference/react)
