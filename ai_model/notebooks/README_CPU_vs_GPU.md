# CPU vs GPU Notebook Comparison

## Overview

This directory contains two versions of the HVAC Auto-Labeling Pipeline:

1. **`autodistill_hvac_grounded_sam2.ipynb`** - GPU-Optimized Version (Original)
2. **`autodistill_hvac_pipeline_CPU.ipynb`** - CPU-Optimized Version (New)

## When to Use Each Version

### Use GPU Version (`autodistill_hvac_grounded_sam2.ipynb`) When:

- ✅ You have access to NVIDIA GPU with CUDA support
- ✅ Processing large datasets (50+ images)
- ✅ Running production training (100+ epochs)
- ✅ Time is critical
- ✅ Multiple training iterations needed
- ✅ Fast experimentation required

### Use CPU Version (`autodistill_hvac_pipeline_CPU.ipynb`) When:

- ✅ No GPU available
- ✅ Small datasets (< 10 images)
- ✅ Development and testing
- ✅ Learning the pipeline
- ✅ Budget constraints
- ✅ CPU-only cloud environments

## Performance Comparison

| Operation | CPU Time | GPU Time | Speed Difference |
|-----------|----------|----------|------------------|
| **Auto-labeling** (5 images) | 1-5 minutes | 15-30 seconds | 5-10x slower |
| **Training** (50 epochs) | 2-4 hours | 5-15 minutes | 10-20x slower |
| **Training** (100 epochs) | 4-8 hours | 10-30 minutes | 10-20x slower |
| **Inference** (per image) | 1-5 seconds | 0.1-0.5 seconds | 5-10x slower |

## Key Differences

### 1. PyTorch Installation

**GPU Version:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU Version:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Environment Setup

**GPU Version:**
- Installs CUDA-enabled PyTorch
- Checks for CUDA availability
- Clears GPU memory cache
- Reports GPU device name

**CPU Version:**
- Installs CPU-only PyTorch (smaller, faster install)
- Configures CPU thread optimization
- Checks system RAM and CPU cores
- Provides CPU performance warnings

### 3. Training Configuration

**GPU Version:**
- Default: 100 training epochs
- Optimized for fast GPU training

**CPU Version:**
- Default: 50 training epochs (reduced for CPU)
- Recommends 10-20 epochs for testing
- Includes time estimates based on CPU performance

### 4. Progress Monitoring

**GPU Version:**
- Standard progress updates
- Assumes fast processing

**CPU Version:**
- Enhanced progress indicators
- Estimated time remaining
- More frequent status updates
- Performance tips and warnings

### 5. Documentation

**GPU Version:**
- Focus on GPU requirements
- Assumes fast execution
- Production-oriented

**CPU Version:**
- Extensive CPU performance notes
- Time estimates for each phase
- Optimization tips for CPU
- Clear expectations about longer processing times

## Technical Details

### GPU-Specific Code Removed in CPU Version

1. **CUDA Installation:**
   - `--index-url https://download.pytorch.org/whl/cu118` → `--index-url https://download.pytorch.org/whl/cpu`

2. **GPU Memory Management:**
   - `torch.cuda.empty_cache()` - Removed
   - CUDA availability checks - Modified to warn if present

3. **GPU Device Detection:**
   - `torch.cuda.get_device_name(0)` - Removed
   - `torch.version.cuda` - Removed

### CPU-Specific Code Added

1. **CPU Optimization:**
   - `torch.set_num_threads(cpu_count)` - Uses all available CPU cores
   - System resource checking with `psutil`
   - Platform information reporting

2. **Enhanced Progress Tracking:**
   - Estimated time remaining calculations
   - Per-image progress with time predictions
   - More verbose status updates

3. **Performance Warnings:**
   - Multiple sections with CPU performance expectations
   - Time estimates for each phase
   - Recommendations for batch sizes and epochs

## File Sizes

- GPU Version: ~62 KB (original)
- CPU Version: ~72 KB (includes additional documentation)

## Compatibility

### GPU Version Requirements:
- NVIDIA GPU with CUDA 11.8+ or 12.x
- 8GB+ VRAM recommended
- CUDA toolkit installed
- Linux/Windows with GPU drivers

### CPU Version Requirements:
- Any CPU (x86_64, ARM)
- 8GB+ RAM recommended (16GB+ optimal)
- Works on Windows, Linux, macOS
- No special hardware requirements

## Maintenance

Both notebooks should be kept synchronized for:
- Core algorithm updates
- Bug fixes
- Security patches
- Feature enhancements

The only differences should be in:
- Hardware-specific setup
- Performance configurations
- Documentation and warnings

## Testing Recommendations

### GPU Version Testing:
1. Test with 5-10 images initially
2. Run 10 epochs for pipeline validation
3. Full 100+ epoch training for production

### CPU Version Testing:
1. Start with 1-3 images to gauge performance
2. Run 5-10 epochs for pipeline validation
3. 20-50 epochs maximum for CPU training
4. Consider using GPU version for production training

## Migration Between Versions

### From GPU to CPU:
- No data migration needed
- Re-run setup cells with CPU version
- Reduce training epochs
- Increase processing time expectations

### From CPU to GPU:
- No data migration needed
- Re-run setup cells with GPU version
- Increase training epochs for better results
- Expect much faster execution

## Support

For issues or questions:
- GPU Version: Check CUDA installation and GPU drivers
- CPU Version: Check system RAM and CPU utilization
- Both: Review logs in `pipeline_logs/` directory

## Version History

- **v1.0** (December 2024)
  - Initial GPU version created
  - Comprehensive pipeline implementation
  
- **v1.1** (December 2024)
  - CPU-optimized version created
  - Enhanced documentation for both versions
  - Performance comparison added

## License

Both versions follow the same license as the main repository.
