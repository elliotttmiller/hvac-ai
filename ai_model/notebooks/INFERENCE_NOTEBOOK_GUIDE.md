# HVAC AI - YOLO Inference Notebook Guide

## 📚 Overview

This guide documents the YOLO11 inference notebooks available in this directory and provides recommendations for different use cases.

## 📓 Available Notebooks

### 1. `hvac-inference_yolo.ipynb` - Quick Start (Improved)
**Use Case:** Fast deployment, simple setup, minimal code

**Features:**
- ✅ 5-cell simple workflow
- ✅ GPU validation
- ✅ Error handling
- ✅ Configuration validation
- ✅ Path validation
- ✅ Clear documentation

**Best For:**
- Quick testing
- Demo purposes
- Users familiar with Colab
- Minimal setup requirements

**Time to Deploy:** ~5 minutes

### 2. `hvac-inference_yolo_enhanced.ipynb` - Production Ready (New)
**Use Case:** Turn-key backend/inference deployment, comprehensive validation

**Features:**
- ✅ Comprehensive environment validation (GPU, CUDA, dependencies)
- ✅ Advanced configuration management with security best practices
- ✅ Model loading with warm-up and validation
- ✅ Test inference with visualization
- ✅ Performance benchmarking across multiple image sizes
- ✅ Production-ready API deployment
- ✅ Monitoring and troubleshooting guide
- ✅ Error handling throughout
- ✅ Security best practices (token management)

**Best For:**
- Production deployments
- Client demonstrations
- Performance optimization
- Quality assurance
- First-time users

**Time to Deploy:** ~10-15 minutes (includes validation and testing)

## 🚀 Quick Start

### Option 1: Quick Deployment (Minimal Version)

```bash
1. Open: hvac-inference_yolo.ipynb
2. Set Runtime → GPU
3. Update MODEL_PATH and NGROK_AUTHTOKEN (Cell 3)
4. Run all cells
```

### Option 2: Production Deployment (Enhanced Version)

```bash
1. Open: hvac-inference_yolo_enhanced.ipynb
2. Set Runtime → GPU
3. Follow the step-by-step guide:
   - Environment setup & validation
   - Drive mounting
   - Configuration
   - Model validation
   - Test inference
   - Performance benchmarking
   - Server deployment
```

## 📊 Feature Comparison

| Feature | Quick Start | Enhanced |
|---------|-------------|----------|
| **Cells** | 5 | 7 |
| **Setup Time** | 5 min | 10-15 min |
| **GPU Validation** | ✅ Basic | ✅ Comprehensive |
| **Dependency Check** | ✅ | ✅ |
| **Model Validation** | ❌ | ✅ |
| **Test Inference** | ❌ | ✅ |
| **Benchmarking** | ❌ | ✅ |
| **Visualization** | ❌ | ✅ |
| **Path Validation** | ✅ | ✅ |
| **Error Handling** | ✅ Basic | ✅ Comprehensive |
| **Documentation** | ✅ Basic | ✅ Detailed |
| **Monitoring** | ❌ | ✅ |
| **Troubleshooting** | ❌ | ✅ Comprehensive |
| **Security Practices** | ❌ | ✅ |

## 🎯 Use Case Recommendations

### For Quick Testing
```
Use: hvac-inference_yolo.ipynb
Why: Fast setup, minimal steps, gets you running quickly
```

### For Production/Client Demo
```
Use: hvac-inference_yolo_enhanced.ipynb
Why: Comprehensive validation, professional output, full monitoring
```

### For First-Time Users
```
Use: hvac-inference_yolo_enhanced.ipynb
Why: Step-by-step guidance, troubleshooting, validation at each step
```

### For Development
```
Use: hvac-inference_yolo_enhanced.ipynb (for initial setup)
Then: hvac-inference_yolo.ipynb (for quick iterations)
```

## 📋 Prerequisites

Both notebooks require:

1. **Google Colab Account** (free)
2. **GPU Runtime** (T4 or better)
   - Runtime → Change runtime type → GPU
3. **Trained YOLO11 Model** (`.pt` file in Google Drive)
4. **Ngrok Account** (free) for public URL
   - Get token: https://ngrok.com/

## 🔧 Configuration

### Minimal Configuration (Both Notebooks)
```env
MODEL_PATH="/content/drive/MyDrive/path/to/best.pt"
NGROK_AUTHTOKEN="your_token_here"
PORT=8000
```

### Enhanced Configuration (Enhanced Notebook Only)
```python
# Inference settings
DEFAULT_CONF_THRESHOLD = 0.50  # Detection confidence
DEFAULT_IOU_THRESHOLD = 0.45   # NMS IoU threshold
MAX_IMAGE_SIZE = 1024          # Input image size
```

## 🚦 Deployment Checklist

### Quick Start Deployment
- [ ] Open notebook in Colab
- [ ] Set runtime to GPU
- [ ] Update MODEL_PATH in cell 3
- [ ] Update NGROK_AUTHTOKEN in cell 3
- [ ] Run all cells
- [ ] Access API at provided URL

### Enhanced Deployment
- [ ] Open notebook in Colab
- [ ] Set runtime to GPU
- [ ] Run environment setup (validates GPU)
- [ ] Mount Google Drive
- [ ] Configure paths and tokens
- [ ] Validate model loading
- [ ] Run test inference (optional but recommended)
- [ ] Benchmark performance (optional)
- [ ] Deploy server
- [ ] Test endpoints using /docs
- [ ] Monitor GPU usage (optional)

## 📈 Performance Expectations

### Hardware: Google Colab T4 GPU (~15GB)

| Image Size | Inference Time | FPS | Memory | Use Case |
|------------|----------------|-----|--------|----------|
| 640x640 | ~20-30ms | 40-50 | ~2GB | Real-time, speed priority |
| 1024x1024 | ~40-60ms | 20-25 | ~3GB | **Recommended: Balanced** |
| 1280x1280 | ~80-100ms | 10-12 | ~4GB | Quality priority |

### Expected Results
- **Detection Accuracy:** Based on your trained model (typically mAP50 > 0.85)
- **Processing Speed:** 20-40 FPS on T4 GPU with 1024px images
- **Startup Time:** 20-40 seconds for model loading + warm-up
- **Memory Usage:** 2-4 GB GPU memory depending on image size

## 🔍 Testing Your Deployment

### 1. Health Check
```bash
curl https://your-ngrok-url.ngrok.io/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "2.1.0",
  "device": "cuda"
}
```

### 2. Test Detection
```bash
curl -X POST "https://your-ngrok-url.ngrok.io/analyze/detect" \
     -F "image=@test_blueprint.jpg" \
     -F "conf_threshold=0.5"
```

### 3. Interactive API Testing
Visit: `https://your-ngrok-url.ngrok.io/docs`

This provides:
- Interactive API documentation
- Try-it-out functionality
- Request/response schemas
- Example payloads

## 🐛 Common Issues & Solutions

### Issue: "Model not found"
**Solution:**
1. Verify MODEL_PATH is correct
2. Check if Drive is mounted
3. Confirm .pt file exists at that location
```python
# Check in a cell:
import os
print(os.path.exists(MODEL_PATH))
```

### Issue: "No GPU detected"
**Solution:**
1. Set Runtime → Change runtime type → GPU
2. Restart runtime
3. Re-run setup cells

### Issue: "CUDA out of memory"
**Solution:**
1. Reduce MAX_IMAGE_SIZE to 640 or 800
2. Restart runtime (clears GPU memory)
3. Close other tabs/notebooks

### Issue: "Ngrok connection failed"
**Solution:**
1. Verify token is correct
2. Check ngrok account is active
3. Note: Server works locally without ngrok

### Issue: "Slow inference (>100ms)"
**Solution:**
1. Confirm GPU is being used (check Cell 1 output)
2. Reduce image size
3. Ensure warm-up completed
4. Check if other processes using GPU

### Issue: "No objects detected"
**Solution:**
1. Lower confidence threshold (try 0.25)
2. Verify model was trained on similar images
3. Check image quality and format
4. Test with a known good sample

## 📊 Monitoring Best Practices

### While Server is Running:

1. **Watch Server Logs**
   - Check cell output for request logs
   - Monitor for errors or warnings

2. **Monitor GPU Usage** (Enhanced notebook only)
   - Run the GPU monitor cell
   - Watch for memory spikes
   - Ensure memory stays below 80%

3. **Track Performance**
   - Log inference times
   - Monitor detection counts
   - Watch for degradation

4. **Test Regularly**
   - Use /health endpoint
   - Try sample images
   - Verify response times

## 🔒 Security Best Practices

### Token Management
```python
# ✅ GOOD: Use Colab Secrets
from google.colab import userdata
NGROK_AUTHTOKEN = userdata.get('NGROK_AUTHTOKEN')

# ❌ BAD: Hardcode tokens
NGROK_AUTHTOKEN = "2abc..."  # DON'T DO THIS
```

### Access Control
- Ngrok URLs are public - don't share sensitive data
- Consider adding API authentication for production
- Rotate tokens regularly
- Use environment variables

## 📞 Support & Resources

### Documentation
- **HVAC-AI Docs:** [Repository Documentation](https://github.com/elliotttmiller/hvac-ai/tree/main/docs)
- **Training Guide:** `../TRAINING_GUIDE.md`
- **Optimization Guide:** `../OPTIMIZATION_GUIDE.md`

### External Resources
- **YOLO11 Docs:** https://docs.ultralytics.com/
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Ngrok Docs:** https://ngrok.com/docs

### Troubleshooting
- Check `hvac-inference_yolo_enhanced.ipynb` Cell 8 for comprehensive troubleshooting
- Review server logs for error messages
- Test with known good configurations

## 🎓 Best Practices

### Development Workflow
1. **Start with Enhanced:** Use enhanced notebook for initial setup and validation
2. **Test Thoroughly:** Run test inference and benchmarking
3. **Document Config:** Save your working configuration
4. **Switch to Quick:** Use quick start for rapid iterations once validated
5. **Monitor Performance:** Track metrics over time

### Production Deployment
1. **Always validate** environment before deploying
2. **Test inference** with representative samples
3. **Benchmark performance** to set expectations
4. **Monitor continuously** during operation
5. **Keep logs** for troubleshooting
6. **Document** your configuration and model version

### Model Management
```python
# Document your model
MODEL_VERSION = "v1.2.0"
MODEL_DATE = "2024-12-25"
MODEL_METRICS = {
    "mAP50": 0.87,
    "mAP50-95": 0.68,
    "classes": 12
}
```

## 🚀 Next Steps

After successful deployment:

1. **Integrate with Frontend**
   - Connect Next.js app to API endpoint
   - Implement file upload and results display
   - Add progress tracking

2. **Optimize Performance**
   - Tune inference settings
   - Experiment with image sizes
   - Profile bottlenecks

3. **Scale Up**
   - Consider permanent hosting (AWS, GCP, Azure)
   - Implement caching
   - Add load balancing

4. **Improve Model**
   - Collect production data
   - Retrain with edge cases
   - Version control models

5. **Add Features**
   - Batch processing
   - Webhook notifications
   - Result persistence

## 📝 Changelog

### Version 2.0 (2024-12-25)
- ✨ Created enhanced production-ready notebook
- ✨ Added comprehensive validation and testing
- ✨ Implemented performance benchmarking
- ✨ Added monitoring and troubleshooting guide
- ✨ Improved security best practices
- 🔧 Enhanced original notebook with better error handling
- 📚 Created comprehensive documentation

### Version 1.0 (Initial)
- Basic 4-cell inference notebook
- Simple setup and deployment
- Ngrok integration

## 🤝 Contributing

Found an issue or have a suggestion?

1. Test your improvement
2. Document the changes
3. Submit a PR with:
   - Description of enhancement
   - Use case/benefit
   - Testing results
   - Updated documentation

## 📜 License

This project follows the main repository license.

---

**Last Updated:** 2024-12-25  
**Version:** 2.0  
**Maintainers:** HVAC-AI Team

---

## 🎉 Quick Reference

| Task | Notebook | Time |
|------|----------|------|
| Quick test | `hvac-inference_yolo.ipynb` | 5 min |
| Production deploy | `hvac-inference_yolo_enhanced.ipynb` | 15 min |
| Client demo | `hvac-inference_yolo_enhanced.ipynb` | 15 min |
| First time | `hvac-inference_yolo_enhanced.ipynb` | 15 min |
| Development | Both (enhanced → quick) | Varies |

**Remember:** Always start with GPU runtime enabled! 🚀
