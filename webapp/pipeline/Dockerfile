FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pipeline code
COPY . /app/webapp/pipeline/
# Copy existing analysis modules (needed for incremental analysis)
COPY ../../venezuela-us-reddit-discourse/analysis/ /app/analysis/

ENV PYTHONPATH=/app

CMD ["python", "-m", "webapp.pipeline.main"]
