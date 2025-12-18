# HVAC AI Platform

**Revolutionary AI-Powered HVAC Blueprint Analysis Platform**

An enterprise-grade platform combining Next.js frontend with Python AI services for intelligent HVAC system analysis. Now featuring SAHI (Slice Aided Hyper Inference) for improved accuracy on large blueprints and HVAC-specific validation based on ASHRAE/SMACNA standards.

## Features

### Core Capabilities
- 🤖 **SAHI-Powered Detection** - Slice-based inference for 90%+ accuracy on all blueprint sizes
- 🧠 **Vision-Language Model (VLM)** - Domain-specific AI for pristine HVAC analysis precision
- 🔍 **HVAC System Analysis** - Relationship graphs and connectivity validation
- 📋 **Code Compliance** - ASHRAE Standard 62.1 and SMACNA validation
- 📐 **Multi-Format Support** - PDF, DWG, DXF, PNG, JPG, TIFF processing
- ⚡ **Adaptive Processing** - Quality assessment and enhancement pipeline
- 🎯 **Domain Expertise** - HVAC-specific prompt engineering templates

### Technical Highlights
- Linear scaling with blueprint size (handles 10,000px+ blueprints)
- GPU memory efficient (<8GB for large blueprints)
- Component relationship analysis with engineering constraints
- Professional prompt templates based on industry standards
- Comprehensive testing framework (85%+ coverage target)

## Quick Start

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

## Documentation

📚 **[Full Documentation](docs/README.md)** - Complete guides and API references

### Quick Links
- [Getting Started](docs/GETTING_STARTED.md) - Setup and installation
- **[HVAC Refactoring Guide](docs/HVAC_REFACTORING_GUIDE.md)** - New HVAC-specialized architecture
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Fix common issues with uploads and analysis
- [Inference Deployment (YOLO/Ultralytics)](docs/SAM_DEPLOYMENT.md) - Deploy inference model features
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when backend is running)

### New Features Documentation
- **[VLM Implementation Guide](docs/VLM_IMPLEMENTATION_GUIDE.md)** - 🆕 Vision-Language Model for HVAC
- **[VLM Development Roadmap](docs/VLM_ROADMAP.md)** - 🆕 12-month development plan
- **[VLM Examples](examples/vlm/README.md)** - 🆕 Training and inference examples
- [SAHI Integration](docs/adr/001-sahi-integration.md) - Architecture decision for slice-based inference
- [Prompt Engineering](docs/adr/002-hvac-prompt-engineering.md) - HVAC-specific prompts
- [System Validation](docs/adr/003-system-relationship-validation.md) - Relationship analysis
- [Services README](services/README.md) - New modular service architecture
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
