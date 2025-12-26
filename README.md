# HVAC AI Platform

**Revolutionary AI-Powered HVAC Blueprint Analysis Platform**

An enterprise-grade platform combining Next.js frontend with Python AI services for intelligent HVAC system analysis using YOLOv11 object detection with bounding boxes.

## Features

### Core Capabilities
- 🤖 **YOLOv11 Object Detection** - Fast and accurate bounding box detection for HVAC components
- 🧠 **Multi-Modal AI Pipeline** - 🆕 Combined vision + OCR with Ray Serve distributed inference
- 📐 **Geometric Intelligence** - 🆕 Automatic perspective correction for rotated text extraction
- 📝 **Hybrid OCR + VLM** - Advanced document processing with 92%+ accuracy (research-driven)
- 🔄 **Semantic Caching** - 70% faster processing for similar blueprints
- 🔍 **HVAC System Analysis** - Relationship graphs and connectivity validation
- 📋 **Code Compliance** - ASHRAE Standard 62.1 and SMACNA validation
- 📐 **Multi-Format Support** - PDF, DWG, DXF, PNG, JPG, TIFF processing
- ⚡ **Adaptive Processing** - Quality assessment and enhancement pipeline
- 🎯 **Domain Expertise** - HVAC-specific prompt engineering templates

### Technical Highlights
- Distributed inference with Ray Serve (fractional GPU allocation)
- Universal service architecture (Domain-Driven Design)
- Fast bounding box detection optimized for HVAC components
- GPU memory efficient (<8GB for large blueprints)
- Real-time inference with streaming progress updates
- Rotation-invariant text detection for engineering drawings
- Component relationship analysis with engineering constraints
- Professional prompt templates based on industry standards
- Comprehensive testing framework (85%+ coverage target)

## Quick Start

### Option 1: Ray Serve Mode (Recommended for Production)

```bash
# 1. Install dependencies
npm install
cd python-services && pip install -r requirements.txt && cd ..

# 2. Configure environment
cp .env.example .env.local
# Edit .env.local and set YOLO_MODEL_PATH to your model

# 3. Run with Ray Serve (distributed inference)
python scripts/start_unified.py --mode ray-serve
```

### Option 2: Legacy Mode (Development)

```bash
# 1. Install dependencies
npm install
cd python-services && pip install -r requirements.txt && cd ..

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in your values (see Getting Started guide)

# 3. Run development servers
npm run dev                      # Frontend (port 3000)
cd python-services && python hvac_analysis_service.py  # Backend (port 8000)
```

### New Architecture: HVAC Cortex

The platform now features a **distributed inference architecture** using Ray Serve:

- 🏗️ **Ray Serve Infrastructure** - Scalable, distributed AI pipeline
- 🧠 **Multi-Modal Analysis** - Combined vision (YOLO) + language (OCR) processing
- 📐 **Geometric Intelligence** - Automatic rotation correction for text extraction
- 🎯 **Selective Inference** - Smart routing based on detection classes
- ⚡ **Fractional GPU** - Efficient resource allocation (40% + 30% split)

**Learn More:**
- [Ray Serve Architecture](RAY_SERVE_ARCHITECTURE.md) - Complete infrastructure guide
- [Proof of Completion](PROOF_OF_COMPLETION.md) - Testing and validation guide

## Documentation

📚 **[Full Documentation](docs/README.md)** - Complete guides and API references

### Quick Links
- [Getting Started](docs/GETTING_STARTED.md) - Setup and installation
- **[Deep-Zoom Viewport](docs/DEEP_ZOOM_VIEWPORT.md)** - 🆕 Google Maps-style blueprint navigation
- **[HVAC Refactoring Guide](docs/HVAC_REFACTORING_GUIDE.md)** - New HVAC-specialized architecture
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Fix common issues with uploads and analysis
- [Inference Deployment (YOLO/Ultralytics)](docs/SAM_DEPLOYMENT.md) - Deploy inference model features
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when backend is running)

### New Features Documentation
- **[Advanced Document Processing](docs/ADVANCED_DOCUMENT_PROCESSING.md)** - 🆕 State-of-the-art OCR + VLM pipeline
- **[Research Summary](docs/RESEARCH_SUMMARY.md)** - 🆕 AI document processing research findings
- **[Future Enhancements Roadmap](docs/FUTURE_ENHANCEMENTS_ROADMAP.md)** - 🆕 18-month development plan
- **[VLM Implementation Guide](docs/VLM_IMPLEMENTATION_GUIDE.md)** - Vision-Language Model for HVAC
- **[VLM Development Roadmap](docs/VLM_ROADMAP.md)** - 12-month VLM plan
- **[VLM Examples](examples/vlm/README.md)** - Training and inference examples
- [SAHI Integration](docs/adr/001-sahi-integration.md) - Slice-based inference architecture
- [Prompt Engineering](docs/adr/002-hvac-prompt-engineering.md) - HVAC-specific prompts
- [System Validation](docs/adr/003-system-relationship-validation.md) - Relationship analysis
- [Advanced Document Processing ADR](docs/adr/004-advanced-document-processing.md) - 🆕 Document processing architecture
- [Services README](services/README.md) - Modular service architecture
- [Examples](examples/README.md) - Practical usage examples

### Setup Validation

Run this command to check your setup:
```bash
./scripts/check-setup.sh

# Or use the new HVAC-specific setup
bash hvac-scripts/setup_hvac_dev_env.sh
```

## Project Structure

```
hvac-ai/
├── src/                    # Frontend (Next.js/React)
│   ├── app/               # Next.js App Router pages
│   ├── components/        # React components
│   └── lib/               # Utility libraries
├── services/              # 🆕 New modular HVAC services
│   ├── hvac-ai/          # SAHI engine & prompt engineering
│   ├── hvac-domain/      # System validation & relationships
│   ├── hvac-document/    # Document processing & enhancement
│   └── gateway/          # API gateway (future)
├── python-services/        # Backend (FastAPI/Python)
│   ├── core/              # Core business logic
│   │   ├── ai/           # AI models and inference
│   │   ├── vlm/          # 🆕 Vision-Language Model system
│   │   ├── document/     # Document processing
│   │   ├── estimation/   # Cost estimation
│   │   └── location/     # Location intelligence
│   └── hvac_analysis_service.py  # Main API service
├── docs/                   # Documentation
│   ├── adr/               # 🆕 Architecture Decision Records
│   └── HVAC_REFACTORING_GUIDE.md  # 🆕 Refactoring documentation
├── examples/              # 🆕 Usage examples
├── hvac-tests/            # 🆕 Test suites (unit & integration)
├── hvac-scripts/          # 🆕 HVAC automation scripts
├── hvac-datasets/         # 🆕 HVAC training data
├── notebooks/             # Jupyter notebooks for ML
└── datasets/              # Training datasets
```

## Technology Stack

**Frontend:**
- Next.js 15 with React 18
- TypeScript
- Tailwind CSS
- Radix UI components

**Backend:**
- Python 3.10+
- FastAPI
- PyTorch & YOLO (Ultralytics)
- OpenCV, Tesseract (OCR)

## Contributing

Contributions are welcome! Please see our documentation for guidelines.

---

**Built with ❤️ for the HVAC industry**
