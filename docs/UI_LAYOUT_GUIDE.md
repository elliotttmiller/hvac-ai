# Professional UI Layout Guide

## Full-Page Viewer Architecture

### Viewport Layout Diagram
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                TOP BAR (h-16)                              │
│  ← Back | Project File.pdf                              Generate Quote | ⤢  │
└─────────────────────────────────────────────────────────────────────────────┘
┌────────────────────────────────────────────┬──────────────────────────────┐
│                                            │                              │
│                                            │      RIGHT SIDEBAR (360px)   │
│                                            ├──────────────────────────────┤
│                                            │ [👁] [📊] [💰] [⚙️]         │
│                                            ├──────────────────────────────┤
│         VIEWPORT PANEL (70%)               │                              │
│         InteractiveViewer                  │  Viewport Tab:               │
│         with overlays                      │  • File: blueprint.pdf       │
│         [IMAGE WITH BOXES]                 │  • Size: 2.5 MB              │
│                                            │  • Type: PDF                 │
│                                            │  • Controls: [+] [-] [↔]     │
│                                            │                              │
│                                            │ Analysis Tab:                │
│                                            │ ✓ 12 components detected     │
│                                            │                              │
│                                            │ [Ductwork] 3x [95%]          │
│                                            │ [Compressor] 1x [92%]        │
│                                            │ [Coil] 2x [88%]              │
│                                            │ [Valve] 1x [85%]             │
│                                            │                              │
│                                            │ Quote Tab:                   │
│                                            │ (Not shown until button)      │
│                                            │                              │
│                                            │ Settings Tab:                │
│                                            │ (Future: viewport controls)   │
│                                            │                              │
└────────────────────────────────────────────┴──────────────────────────────┘
```

## Screen States

### State 1: Upload Screen (No File)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HVAC Blueprint Analysis                             │
│                   Blueprint & Component Detection                           │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                                                                             │
│                      ┌─────────────────────────────┐                       │
│                      │  UPLOAD FILE                │                       │
│                      │  Drag & Drop here or click  │                       │
│                      │  to browse files            │                       │
│                      │  Max 500MB                  │                       │
│                      │         📁                  │                       │
│                      └─────────────────────────────┘                       │
│                                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State 2: Analysis Screen - Viewport Tab (Active)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ← Back | Project File.pdf (12 components)      [Zap Generate Quote] [New]   │
├────────────────────────────────────────────┬──────────────────────────────┤
│                                            │ [👁 Viewport] [📊] [💰] [⚙️] │
│                                            ├──────────────────────────────┤
│                                            │                              │
│         🖼️  IMAGE VIEWER                  │  File Information:           │
│         WITH DETECTION BOXES               │  ├─ Name: blueprint.pdf     │
│         🔲 Ductwork (95%)                  │  ├─ Size: 2.50 MB           │
│         🔲 Compressor (92%)                │  └─ Type: application/pdf   │
│         🔲 Coil (88%)                      │                              │
│         🔲 Valve (85%)                     │  Viewport Controls:          │
│                                            │  ├─ Zoom: [−] 100% [+]      │
│                                            │  ├─ Pan: [↑] [↓] [←] [→]    │
│                                            │  └─ Reset: [⟳]              │
│                                            │                              │
│                                            │                              │
│                                            │                              │
│                                            │                              │
│                                            │                              │
└────────────────────────────────────────────┴──────────────────────────────┘
```

### State 3: Analysis Screen - Analysis Tab (Active)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ← Back | Project File.pdf (12 components)      [Zap Generate Quote] [New]   │
├────────────────────────────────────────────┬──────────────────────────────┤
│                                            │ [👁] [📊 Analysis] [💰] [⚙️]│
│                                            ├──────────────────────────────┤
│         🖼️  IMAGE VIEWER                  │                              │
│         (shown in background)              │ ✓ 12 Components Detected    │
│                                            │                              │
│                                            │ Detected Components:         │
│                                            │ ┌──────────────────────────┐ │
│                                            │ │ Ductwork         3x       │ │
│                                            │ │ [████████████████░░░] 95%│ │
│                                            │ ├──────────────────────────┤ │
│                                            │ │ Compressor       1x       │ │
│                                            │ │ [██████████████░░░░░░] 92%│ │
│                                            │ ├──────────────────────────┤ │
│                                            │ │ Coil             2x       │ │
│                                            │ │ [████████████░░░░░░░░░] 88%│ │
│                                            │ ├──────────────────────────┤ │
│                                            │ │ Valve            1x       │ │
│                                            │ │ [██████████░░░░░░░░░░░] 85%│ │
│                                            │ ├──────────────────────────┤ │
│                                            │ │ Thermostat       1x       │ │
│                                            │ │ [█████████░░░░░░░░░░░░░] 82%│ │
│                                            │ └──────────────────────────┘ │
│                                            │                              │
│                                            │ [Scroll for more items]      │
│                                            │                              │
└────────────────────────────────────────────┴──────────────────────────────┘
```

### State 4: Analysis Screen - Quote Tab (Active)
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ← Back | Project File.pdf (12 components)      [Zap Generate Quote] [New]   │
├────────────────────────────────────────────┬──────────────────────────────┤
│                                            │ [👁] [📊] [💰 Quote] [⚙️]    │
│                                            ├──────────────────────────────┤
│         🖼️  IMAGE VIEWER                  │                              │
│         (shown in background)              │ QUOTE-PROJECT-001            │
│                                            │                              │
│                                            │ LINE ITEMS:                  │
│                                            │ ┌──────────────────────────┐ │
│                                            │ │ DUCTWORK 1      $250.00  │ │
│                                            │ │ DUCTWORK 2      $250.00  │ │
│                                            │ │ DUCTWORK 3      $250.00  │ │
│                                            │ │ COMPRESSOR 1    $250.00  │ │
│                                            │ │ COIL 1          $250.00  │ │
│                                            │ │ COIL 2          $250.00  │ │
│                                            │ │ VALVE 1         $250.00  │ │
│                                            │ │ + 5 more items           │ │
│                                            │ ├──────────────────────────┤ │
│                                            │ │ Subtotal Materials: $3000│ │
│                                            │ │ Subtotal Labor:     $120 │ │
│                                            │ │ ───────────────────────  │ │
│                                            │ │ TOTAL ESTIMATE:   $4440  │ │
│                                            │ └──────────────────────────┘ │
│                                            │                              │
│                                            │ [📥 Export Quote] [Print]    │
│                                            │                              │
│                                            │ [Scroll for more]            │
│                                            │                              │
└────────────────────────────────────────────┴──────────────────────────────┘
```

## Component Dimensions

### Fixed/Responsive Sizes
```
Full Page
├── Header
│   └── Height: 64px (py-4)
│   └── Padding: px-6 py-4
│
├── Main Content (flex, h-screen - header)
│   ├── Viewport Panel
│   │   ├── Width: calc(100% - 360px)
│   │   ├── Height: 100%
│   │   └── Padding: p-4
│   │       └── Viewer Container
│   │           ├── Bg: black
│   │           ├── Border Radius: rounded-lg
│   │           ├── Aspect: responsive (full container)
│   │           └── InteractiveViewer (w-full h-full)
│   │
│   └── Sidebar Tabs
│       ├── Width: 360px (fixed)
│       ├── Height: 100%
│       ├── Border Left: 1px slate-800
│       │
│       ├── Tab List
│       │   ├── Height: auto
│       │   ├── Grid: grid-cols-3
│       │   └── Padding: px-4 pt-4
│       │
│       └── Tab Contents
│           ├── Height: flex-1
│           ├── Overflow: auto
│           └── Padding: p-4
```

### Typography Hierarchy
```
Page Title (h1): text-3xl font-bold
  ↓ File Name (h2): text-xl font-bold
    ↓ Tab Titles (h3): text-sm font-semibold
      ↓ Body Text (p): text-sm
        ↓ Small Text (label): text-xs
```

## Color Scheme

### Dark Theme Palette
```
Background:
  └── Page: bg-slate-950
  └── Header: bg-slate-900
  └── Sidebar: bg-slate-900
  └── Viewer Container: bg-black
  └── Cards/Boxes: bg-slate-800

Text:
  └── Primary: text-white
  └── Secondary: text-slate-200
  └── Tertiary: text-slate-400
  └── Disabled: text-slate-600

Borders:
  └── Primary: border-slate-800
  └── Secondary: border-slate-700

Accents:
  └── Success/Primary: emerald-600 (buttons, highlights)
  └── Hover: emerald-700
  └── Badge: emerald-500 (progress bars, badges)

Special:
  └── Success Background: emerald-900/20 (with border-emerald-800)
  └── Success Text: emerald-300/emerald-200
```

## Interactive States

### Button States
```
Default (Generate Quote):
├── bg-emerald-600
├── text-white
├── cursor-pointer
└── Shadow: shadow-md

Hover:
├── bg-emerald-700
└── text-white

Disabled:
├── opacity-50
├── cursor-not-allowed
└── No background change

Focus:
├── ring-2 ring-emerald-500
└── ring-offset-2
```

### Tab States
```
Active Tab:
├── bg-emerald-600 (implicit in TabsList)
├── text-white
└── Underline: emerald border

Inactive Tab:
├── bg-slate-700
├── text-slate-300
└── Hover: text-white

Tab Content:
├── Visible: data-[state=active]:block
└── Hidden: data-[state=inactive]:hidden
```

## Responsive Behavior

### Desktop (1920x1080)
```
Viewport: 1560px | Sidebar: 360px
All content fully visible
No scrolling in main areas
Tabs content may scroll internally if many items
```

### Tablet (1280x720) - Not Yet Optimized
```
Note: Current design doesn't collapse sidebar on tablet
Future enhancement: Sidebar collapse/drawer on <1280px
```

### Mobile - Not Supported Yet
```
Note: Current design requires > 720px height and > 960px width
Future enhancement: Vertical layout, sidebar becomes bottom tabs
```

## Performance Considerations

### Rendering Optimization
```
Tab Content:
└── Only active tab's content rendered (via Tabs component)
└── Inactive tabs unmounted (data-[state=inactive]:hidden)
└── Improves performance with large image files

Image Display:
└── URL.createObjectURL() creates blob URL
└── More efficient than base64 encoding
└── Released automatically when component unmounts

API Calls:
└── Single /api/analysis call per upload
└── No unnecessary re-fetches on tab switch
└── isProcessing flag prevents duplicate uploads
```

## Accessibility Features

### Keyboard Navigation
```
Tab Key: Navigate between tabs and buttons
Enter/Space: Activate focused button or tab
Escape: Close any modals (future)

Tab Order:
1. Back button
2. Generate Quote button
3. New Upload button
4. Sidebar tabs (left to right)
5. Sidebar content (tab-specific)
```

### Screen Reader Support
```
Header:
├── Semantic <h1> for page title
├── Semantic <p> for subtitle
└── aria-label on buttons

Tabs:
├── role="tablist" (implicit in Tabs component)
├── role="tab" on tab buttons
├── role="tabpanel" on content areas
└── aria-selected on active tab

Detections:
├── Proper heading hierarchy
├── <ul>/<li> for component lists
└── Progress bar with aria-valuenow

Buttons:
├── Proper contrast (WCAG AA)
├── Focus indicators (2px ring)
└── Icons with text labels
```

## Future Enhancement Opportunities

### Phase 2 UI Improvements
```
1. Collapsible Sidebar
   └── Hamburger menu on left side
   └── Smooth slide-out animation
   └── Tablet/mobile optimization

2. Settings Tab Implementation
   └── Zoom level selector
   └── Pan controls
   └── Confidence threshold slider
   └── Overlay visibility toggles

3. Advanced Quote Customization
   └── Edit line item costs
   └── Add/remove components
   └── Apply discounts
   └── Save quote templates

4. Multi-File Analysis
   └── Tab view for multiple uploads
   └── Batch comparison
   └── Side-by-side viewer

5. Export Options
   └── Download quote as PDF
   └── Email quote directly
   └── Print blueprint with overlays
   └── Save analysis data as JSON
```

## Testing Viewport Sizes

### Recommended Test Configurations
```
1920x1080 (Full HD)
  └── Primary testing environment
  └── All content visible
  └── Standard desktop monitor

1440x900 (Common laptop)
  └── Sidebar still visible
  └── Some tab content may scroll
  └── Good real-world test size

1280x720 (Tablet width)
  └── Minimum width before sidebar collapse needed
  └── Good test for responsive boundaries
  └── May need adjustment for optimal experience

1024x768 (Legacy)
  └── Sidebar would need to collapse
  └── Not currently optimized
  └── Future mobile version needed
```

## Migration Notes from Previous UI

### Breaking Changes
```
1. Quote Generation
   └── Was: Automatic on analysis completion
   └── Now: Manual button click required
   └── Impact: Users must click "Generate Quote" to see cost estimate

2. Layout System
   └── Was: 2-column grid with scrolling
   └── Now: Fixed viewport with sidebar tabs
   └── Impact: Some users may need to re-learn tab navigation

3. Navigation Flow
   └── Was: Single page, scroll-based
   └── Now: Tab-based sidebar navigation
   └── Impact: Information discovery slightly different
```

### Backward Compatibility
```
✓ Same API endpoints
✓ Same file format support
✓ Same analysis pipeline
✓ Same data structure (detections, quotes)
✗ Different UI layout
✗ Different quote trigger mechanism
```
