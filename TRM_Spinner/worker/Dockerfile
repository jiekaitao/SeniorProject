FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12, pip, and curl
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python

# Set workdir
WORKDIR /app

# Install Python dependencies (web server)
COPY requirements.txt .
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 12.8 (required for RTX 5090 / sm_120)
RUN pip install --break-system-packages --no-cache-dir \
    torch==2.9.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install TRM-specific dependencies (from the original project)
# These are installed separately since they may need special build flags
RUN pip install --break-system-packages --no-cache-dir \
    adam-atan2==0.0.3 \
    einops==0.8.1 \
    tqdm==4.67.1 \
    coolname==2.2.0 \
    pydantic==2.11.7 \
    argdantic==1.3.3 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2 \
    huggingface_hub==0.34.4 \
    packaging==24.1 \
    numba==0.61.2

# Install muon optimizer
RUN pip install --break-system-packages --no-cache-dir \
    "muon-optimizer @ git+https://github.com/KellerJordan/Muon"

# Copy application code
COPY . /app

# Expose port
EXPOSE 8000

# Health check uses curl (installed above)
HEALTHCHECK --interval=15s --timeout=10s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
