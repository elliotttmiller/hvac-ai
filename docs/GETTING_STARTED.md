# HVAC AI Platform - Getting Started

## Prerequisites

- **Node.js 18+** (with npm)
- **Python 3.9+** (with pip)
- **Git**

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/elliotttmiller/hvac-ai.git
cd hvac-ai
```

### 2. Install Frontend Dependencies

```bash
npm install --legacy-peer-deps
```

### 3. Install Python Dependencies

```bash
cd python-services
pip install -r requirements.txt
cd ..
```

## Running the Platform

### Option 1: Using Startup Scripts

**Terminal 1 - Python Service:**
```bash
cd python-services
./start.sh
```

**Terminal 2 - Next.js Frontend:**
```bash
npm run dev
```

### Option 2: Manual Start

**Terminal 1 - Python Service:**
```bash
cd python-services
python hvac_analysis_service.py
```

**Terminal 2 - Next.js Frontend:**
```bash
npm run dev
```

## Access the Platform

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Root**: http://localhost:8000

## Quick Test

1. Open http://localhost:3000
2. Navigate to "Documents" or click "Upload Blueprint"
3. Upload a test PDF or image file
4. View analysis results

## API Usage Examples

### Using curl

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Upload Blueprint:**
```bash
curl -X POST http://localhost:8000/api/analyze/blueprint \
  -F "file=@/path/to/blueprint.pdf" \
  -F "project_id=test_project" \
  -F "location=Chicago, IL"
```

**Get Analysis:**
```bash
curl http://localhost:8000/api/analyze/analysis_20231205_143022
```

**Generate Estimation:**
```bash
curl -X POST http://localhost:8000/api/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "analysis_20231205_143022",
    "location": "Chicago, IL"
  }'
```

### Using Python

```python
import requests

# Upload blueprint
with open('blueprint.pdf', 'rb') as f:
    files = {'file': f}
    data = {
        'project_id': 'test_project',
        'location': 'Chicago, IL'
    }
    response = requests.post(
        'http://localhost:8000/api/analyze/blueprint',
        files=files,
        data=data
    )
    analysis = response.json()
    print(f"Analysis ID: {analysis['analysis_id']}")

# Generate estimation
estimation_data = {
    'analysis_id': analysis['analysis_id'],
    'location': 'Chicago, IL'
}
response = requests.post(
    'http://localhost:8000/api/estimate',
    json=estimation_data
)
estimation = response.json()
print(f"Total Cost: ${estimation['total_cost']:,.2f}")
```

### Using JavaScript/TypeScript

```typescript
// Upload blueprint
const formData = new FormData();
formData.append('file', file);
formData.append('projectId', 'test_project');
formData.append('location', 'Chicago, IL');

const analysisResponse = await fetch('/api/hvac/analyze', {
  method: 'POST',
  body: formData,
});
const analysis = await analysisResponse.json();

// Generate estimation
const estimationResponse = await fetch('/api/hvac/estimate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    analysisId: analysis.analysis_id,
    location: 'Chicago, IL',
  }),
});
const estimation = await estimationResponse.json();
console.log(`Total Cost: $${estimation.total_cost}`);
```

## Environment Configuration

### Frontend (.env.local)

```bash
# Copy example file
cp .env.example .env.local

# Edit configuration
NEXT_PUBLIC_AI_SERVICE_URL=http://localhost:8000
```

### Python Service (.env)

```bash
# Copy example file
cd python-services
cp .env.example .env

# Edit configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## Development Workflow

### Frontend Development

```bash
# Run development server
npm run dev

# Run linter
npm run lint

# Format code
npm run format

# Build for production
npm run build
```

### Python Service Development

```bash
cd python-services

# Run service with auto-reload
uvicorn hvac_analysis_service:app --reload

# Run tests
pytest tests/

# Format code
black .
```

## Troubleshooting

### Port Already in Use

If port 3000 or 8000 is already in use:

```bash
# Frontend
PORT=3001 npm run dev

# Python service
uvicorn hvac_analysis_service:app --port 8001
```

### Module Import Errors

Ensure virtual environment is activated:

```bash
cd python-services
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### TypeScript Errors

```bash
# Install dependencies
npm install --legacy-peer-deps

# Run type check
npx tsc --noEmit
```

## Next Steps

- Read [ARCHITECTURE.md](./ARCHITECTURE.md) for technical details
- Explore API documentation at http://localhost:8000/docs
- Check out example blueprints in `/examples`
- Review code in `/python-services/core/` modules

## Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` directory
- **API Docs**: http://localhost:8000/docs when service is running
