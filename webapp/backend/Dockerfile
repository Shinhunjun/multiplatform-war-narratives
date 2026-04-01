FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code into proper package structure
COPY . /app/webapp/backend/

# Create parent package __init__.py
RUN touch /app/webapp/__init__.py

ENV PYTHONPATH=/app
ENV DATA_DIR=/tmp/analysis_data
ENV PORT=8080

CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8080"]
