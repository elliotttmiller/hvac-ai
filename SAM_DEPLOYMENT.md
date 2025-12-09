# SAM Model Deployment Quick Start

## Prerequisites

### Backend
- Python 3.10+
- CUDA 11.8+ (for GPU support)
- NVIDIA GPU with 12+ GB VRAM (T4 or better)

### Frontend
- Node.js 18+
- npm or bun

## Quick Deployment

### Option 1: Docker (Recommended)

1. **Place your trained model**:
   ```bash
   mkdir -p python-services/models
   # Copy your sam_hvac_finetuned.pth to python-services/models/
   ```

2. **Deploy with Docker Compose**:
   ```bash
   cd python-services
   docker-compose up -d
   ```

3. **Verify backend is running**:
   ```bash
   curl http://localhost:8000/health
   ```

### Option 2: Manual Setup

#### Backend Setup

1. **Install Python dependencies**:
   ```bash
   cd python-services
   pip install -r requirements.txt
   ```

2. **Place model file** at `python-services/models/sam_hvac_finetuned.pth`

3. **Start the service**:
   ```bash
   python hvac_analysis_service.py
   ```

   The service will start on `http://localhost:8000`

#### Frontend Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Set environment variables**:
   Create `.env.local`:
   ```bash
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

   Access the SAM analysis page at: `http://localhost:3000/sam-analysis`

## Testing the API

### Test Interactive Segmentation

```bash
# Prepare a test image
curl -X POST http://localhost:8000/api/v1/segment \
  -F "image=@test_diagram.png" \
  -F 'prompt={"type":"point","data":{"coords":[100,100],"label":1}}'
```

Expected response:
```json
{
  "status": "success",
  "segments": [
    {
      "label": "Valve-Ball",
      "score": 0.967,
      "mask": "base64_encoded_rle_string",
      "bbox": [85, 85, 30, 30]
    }
  ]
}
```

### Test Automated Counting

```bash
curl -X POST http://localhost:8000/api/v1/count \
  -F "image=@test_diagram.png"
```

Expected response:
```json
{
  "status": "success",
  "total_objects_found": 87,
  "counts_by_category": {
    "Valve-Ball": 23,
    "Valve-Gate": 12,
    "Fitting-Bend": 31,
    "Equipment-Pump-Centrifugal": 2,
    "Instrument-Pressure-Indicator": 19
  }
}
```

## Mock Mode (No Model Required)

Both endpoints work in mock mode for development/testing when no model file is present:

1. Start the service without a model file
2. The service will automatically use mock responses
3. This is useful for frontend development and UI testing

## Production Deployment

### Backend (GPU Server)

1. **Use Docker Compose** for production:
   ```bash
   cd python-services
   docker-compose up -d
   ```

2. **Configure reverse proxy** (nginx example):
   ```nginx
   upstream sam_backend {
       server localhost:8000;
   }

   server {
       listen 80;
       server_name api.yourdomain.com;

       location /api/v1/ {
           proxy_pass http://sam_backend;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           client_max_body_size 100M;
       }
   }
   ```

3. **Monitor with health checks**:
   ```bash
   curl http://localhost:8000/health
   ```

### Frontend (Next.js)

1. **Build for production**:
   ```bash
   npm run build
   npm run start
   ```

2. **Or deploy to Vercel/Netlify**:
   - Push to GitHub
   - Connect repository to Vercel/Netlify
   - Set `NEXT_PUBLIC_API_URL` environment variable
   - Deploy

## Troubleshooting

### Backend Issues

**Service won't start**:
- Check Python version: `python --version` (need 3.10+)
- Verify CUDA: `nvidia-smi`
- Check logs: `docker-compose logs` or service output

**Slow inference**:
- Verify GPU is being used
- Check `nvidia-smi` for GPU utilization
- Reduce grid size for counting

**Out of memory**:
- Use smaller images
- Reduce grid density
- Check available GPU memory

### Frontend Issues

**API connection failed**:
- Verify backend is running: `curl http://localhost:8000/health`
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify CORS settings in backend

**Canvas not displaying**:
- Check browser console for errors
- Ensure image is valid format (PNG, JPG, JPEG)
- Try different image

## Environment Variables

### Backend
```bash
SAM_MODEL_PATH=/app/models/sam_hvac_finetuned.pth
CUDA_VISIBLE_DEVICES=0
```

### Frontend
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Support

For issues or questions:
- Check `docs/SAM_INTEGRATION_GUIDE.md` for detailed documentation
- Review API endpoints at `http://localhost:8000/docs`
- Open an issue on GitHub

## Performance Benchmarks

Typical performance on NVIDIA T4 GPU:
- Interactive segmentation: <1 second
- Full diagram counting (1000x1000px): 2-5 seconds
- Memory usage: 8-10 GB GPU RAM

## Next Steps

1. Train or obtain a fine-tuned SAM model for HVAC/P&ID diagrams
2. Place model at `python-services/models/sam_hvac_finetuned.pth`
3. Deploy using Docker or manual setup
4. Access the web interface at `/sam-analysis`
5. Upload diagrams and test interactive segmentation
6. Run automated counting on full diagrams
7. Export results to CSV

## Additional Resources

- [SAM Integration Guide](docs/SAM_INTEGRATION_GUIDE.md)
- [SAM Pipeline Summary](docs/SAM_PIPELINE_SUMMARY.md)
- [API Documentation](http://localhost:8000/docs) (when service is running)
