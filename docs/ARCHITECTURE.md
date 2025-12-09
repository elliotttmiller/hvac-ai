# HVAC AI Platform - Architecture Overview

## System Architecture

The HVAC AI Platform is a full-stack application combining a modern React frontend with a Python-based AI backend.

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Next.js Frontend (Port 3000)                   │  │
│  │  - React 18 + TypeScript                             │  │
│  │  - Tailwind CSS + Radix UI                           │  │
│  │  - Interactive canvas components                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ HTTP/REST API
                           │
┌─────────────────────────────────────────────────────────────┐
│              Python Backend (Port 8000)                      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           FastAPI Application                         │  │
│  │  - RESTful API endpoints                             │  │
│  │  - Async request handling                            │  │
│  │  - Auto-generated OpenAPI docs                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                           │                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Core Business Logic                      │  │
│  │                                                        │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │  │
│  │  │  AI Module  │  │   Document   │  │ Estimation │  │  │
│  │  │             │  │  Processing  │  │            │  │  │
│  │  │ - SAM       │  │              │  │ - Costs    │  │  │
│  │  │ - YOLO      │  │ - OCR        │  │ - Labor    │  │  │
│  │  │ - Detector  │  │ - CAD Parse  │  │            │  │  │
│  │  └─────────────┘  └──────────────┘  └────────────┘  │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │         Location Intelligence                    │ │  │
│  │  │  - Building codes  - Climate zones              │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ GPU Acceleration
                           │
┌─────────────────────────────────────────────────────────────┐
│                  AI Models & Data                            │
│  - SAM (Segment Anything Model) - Fine-tuned                │
│  - YOLO (Object Detection)                                   │
│  - Tesseract/EasyOCR (Text Recognition)                     │
│  - Model weights cached in memory                            │
└─────────────────────────────────────────────────────────────┘
```

## Frontend Architecture

### Technology Stack
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **UI Library**: React 18
- **Styling**: Tailwind CSS
- **Components**: Radix UI
- **State Management**: React hooks + Context API
- **Authentication**: NextAuth.js + Supabase

### Directory Structure
```
src/
├── app/                    # Next.js App Router
│   ├── api/               # API routes (proxies to backend)
│   ├── auth/              # Authentication pages
│   ├── documents/         # Document management
│   ├── projects/          # Project dashboard
│   └── sam-analysis/      # SAM analysis interface
├── components/            # React components
│   ├── ai/               # AI service components
│   ├── auth/             # Auth components
│   ├── common/           # Shared components
│   ├── documents/        # Document upload/display
│   ├── hvac/             # HVAC-specific components
│   ├── layout/           # Layout components
│   ├── projects/         # Project components
│   ├── sam/              # SAM analysis components
│   └── ui/               # Base UI components (Radix)
├── lib/                  # Utilities and services
│   ├── ai-services.ts    # AI service client
│   ├── auth.ts           # Authentication utilities
│   ├── supabase.ts       # Supabase client
│   └── utils.ts          # General utilities
└── types/                # TypeScript type definitions
```

## Backend Architecture

### Technology Stack
- **Framework**: FastAPI
- **Language**: Python 3.10+
- **AI/ML**: PyTorch, Segment Anything, Ultralytics (YOLO)
- **Computer Vision**: OpenCV, scikit-image
- **OCR**: Tesseract, EasyOCR
- **Document Processing**: PyPDF2, PyMuPDF, ezdxf

### Core Modules

#### AI Module (`core/ai/`)
Handles all AI inference and analysis:
- **SAM Inference** - Segment Anything Model for precise component segmentation
- **Component Detection** - YOLO-based object detection
- **Spatial Analysis** - Relationship analysis between components
- **Classification** - Multi-stage component classification

**Key Features:**
- Embedding cache (LRU, configurable size)
- Adaptive grid processing
- GPU acceleration
- Batch processing

#### Document Module (`core/document/`)
Processes various blueprint and CAD formats:
- PDF extraction and rendering
- DWG/DXF parsing
- Image enhancement
- OCR text extraction
- Drawing classification

#### Estimation Module (`core/estimation/`)
Cost and labor estimation:
- Material cost calculation
- Labor hour estimation
- Regional price adjustments
- Component-level pricing

#### Location Module (`core/location/`)
Location intelligence and compliance:
- Building code lookup
- Climate zone analysis
- Regional requirements
- Compliance checking

## Data Flow

### SAM Analysis Workflow

```
1. User uploads blueprint image
   └─> Frontend validates file
       └─> Sends to backend API

2. Backend receives image
   └─> Generates image embedding (cached)
       └─> User clicks on component
           └─> Frontend sends point coordinates

3. SAM Inference Engine
   └─> Retrieves cached embedding
       └─> Generates segmentation mask
           └─> Classifies component type
               └─> Returns results

4. Frontend displays results
   └─> Renders mask overlay
       └─> Shows component details
           └─> Updates component list
```

### Component Counting Workflow

```
1. User uploads blueprint image
   └─> Frontend validates file
       └─> Sends to backend API

2. Backend processes image
   └─> Generates image embedding
       └─> Creates adaptive grid
           └─> Runs inference on grid points
               └─> Applies NMS de-duplication

3. Classification & Aggregation
   └─> Classifies each detection
       └─> Groups by category
           └─> Calculates statistics
               └─> Returns summary

4. Frontend displays results
   └─> Shows total count
       └─> Category breakdown
           └─> Visual markers on image
```

## API Design

### RESTful Endpoints

**Format**: All endpoints use standard REST conventions
- `GET` - Retrieve resources
- `POST` - Create resources or trigger actions
- `PUT` - Update resources
- `DELETE` - Remove resources

**Authentication**: Bearer token (JWT)
**Content Types**: JSON, multipart/form-data (for file uploads)
**Error Handling**: Standard HTTP status codes + detailed error messages

### API Versioning

Current version: `v1`
- Base path: `/api/v1/`
- Backward compatibility maintained
- New features use optional parameters

## Performance Optimizations

### Frontend
- Code splitting with Next.js dynamic imports
- Image optimization with Next.js Image component
- Client-side caching for API responses
- Lazy loading for components
- Debouncing for user interactions

### Backend
- **Embedding Cache**: LRU cache for image embeddings (reduces inference time)
- **GPU Acceleration**: CUDA-enabled PyTorch models
- **Async Processing**: FastAPI async endpoints
- **Batch Processing**: Multiple inferences per request
- **Connection Pooling**: Efficient resource management

### Infrastructure
- Docker containers for consistent deployment
- GPU-enabled Docker images (CUDA support)
- Health checks for service monitoring
- Horizontal scaling capability

## Security Considerations

- Environment variables for sensitive data
- CORS configuration for production
- API rate limiting (planned)
- Input validation on all endpoints
- Secure file upload handling
- Authentication required for protected routes

## Scalability

### Current Architecture
- Monolithic backend (single service)
- Stateless design (except model cache)
- Horizontal scaling ready

### Future Considerations
- Microservices architecture
- Message queue for async tasks (Celery + Redis)
- Model serving optimization (ONNX, TensorRT)
- CDN for static assets
- Database for persistent storage

## Development Workflow

1. **Local Development**: Frontend + Backend on localhost
2. **Docker Development**: Both services in containers
3. **Production**: Containerized deployment with orchestration

## Monitoring & Logging

- **Logging**: Python logging module + structured logs
- **Metrics**: Performance metrics endpoint (`/api/v1/metrics`)
- **Health Checks**: `/health` endpoint for monitoring
- **Error Tracking**: Comprehensive error messages in API responses

## Technology Choices

### Why Next.js?
- Server-side rendering for better SEO
- API routes for backend proxying
- File-based routing
- Built-in optimization
- Great developer experience

### Why FastAPI?
- High performance (async support)
- Auto-generated API documentation
- Type hints and validation
- Easy integration with Python ML libraries
- Modern Python features

### Why SAM?
- State-of-the-art segmentation
- Zero-shot learning capabilities
- Fine-tunable for specific domains
- Interactive and automatic modes
- Excellent accuracy on technical diagrams

## Future Enhancements

- [ ] Real-time collaboration features
- [ ] 3D BIM model generation
- [ ] Advanced cost optimization
- [ ] Multi-user project management
- [ ] Mobile application
- [ ] Offline mode support
- [ ] Cloud storage integration
- [ ] Advanced analytics dashboard
