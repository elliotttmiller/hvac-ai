# HVAC AI Platform

**Revolutionary AI-Powered HVAC Blueprint Analysis Platform**

An enterprise-grade platform combining Next.js frontend with Python AI services for intelligent HVAC system analysis.

## Features

- 🤖 AI-powered component detection with SAM (Segment Anything Model)
- 📐 Multi-format blueprint processing (PDF, DWG, DXF, PNG, JPG)
- 🌍 Location intelligence & building code compliance
- 💰 Automated cost estimation
- 📊 3D visualization and interactive analysis

## Quick Start

```bash
# Install dependencies
npm install
cd python-services && pip install -r requirements.txt

# Run development servers
npm run dev                      # Frontend (port 3000)
python hvac_analysis_service.py  # Backend (port 8000)
```

## Documentation

📚 **[Full Documentation](docs/README.md)** - Complete guides and API references

### Quick Links
- [Getting Started](docs/GETTING_STARTED.md) - Setup and installation
- [SAM Deployment](docs/SAM_DEPLOYMENT.md) - Deploy SAM model features
- [API Documentation](http://localhost:8000/docs) - Interactive API docs (when backend is running)

## Project Structure

```
hvac-ai/
├── src/                    # Frontend (Next.js/React)
│   ├── app/               # Next.js App Router pages
│   ├── components/        # React components
│   └── lib/               # Utility libraries
├── python-services/        # Backend (FastAPI/Python)
│   ├── core/              # Core business logic
│   │   ├── ai/           # AI models and inference
│   │   ├── document/     # Document processing
│   │   ├── estimation/   # Cost estimation
│   │   └── location/     # Location intelligence
│   └── hvac_analysis_service.py  # Main API service
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for ML
└── datasets/               # Training datasets
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
- PyTorch & Segment Anything Model (SAM)
- OpenCV, Tesseract (OCR)

## Contributing

Contributions are welcome! Please see our documentation for guidelines.

---

**Built with ❤️ for the HVAC industry**
