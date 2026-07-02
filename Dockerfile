FROM python:3.12-slim

# Install system utilities needed for audio/video parsing or OpenCV if used
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Create an empty checkpoints directory where the cloud bucket will map onto
RUN mkdir -p /code/checkpoints

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the API server and source code ONLY (No weights!)
COPY ./api /code/api
COPY ./src /code/src

EXPOSE 8001

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8001}
