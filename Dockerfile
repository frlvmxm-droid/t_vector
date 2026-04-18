# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps for tkinter (headless), sklearn, and openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-tk \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn numpy scipy joblib openpyxl

# Copy source
COPY . .

# Default: run tests (CI mode); override with "python app.py" for GUI
ENV PYTHONPATH=/app
CMD ["python", "-m", "pytest", "-q", "--tb=short", "tests/"]
