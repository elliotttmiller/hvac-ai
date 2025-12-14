# Navigate to project root

# Install NVIDIA PyPI index for optimized packages
pip install nvidia-pyindex

# Install CUDA runtime libraries (cu12 for CUDA 12.x)
pip install nvidia-cuda-runtime-cu12

# Install GPU-enabled PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install AutoDistill ecosystem with GPU support
pip install autodistill
pip install autodistill-grounded-sam
pip install autodistill-yolov8
pip install supervision

# Install supporting GPU-accelerated libraries
pip install opencv-python-headless
pip install transformers
pip install sentence-transformers
pip install ultralytics
pip install numpy
pip install pandas
pip install matplotlib