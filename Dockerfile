FROM python:3.10-slim-buster AS clip

# Install system-level dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender-dev \
    libxext6 \
    build-essential python3-dev

# Install libraries
COPY requirements_docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Copy scripts, model, and test data
ENV PYTHONPATH="/src:${PYTHONPATH}"
COPY src/ /src

# Add volume to allow updating test data while running container
VOLUME ["/src/test_images"]
VOLUME ["/src/models"]

WORKDIR /
EXPOSE 8000
CMD ["python", "/src/main.py", "--host", "0.0.0.0", "--port", "8000"]
