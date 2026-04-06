FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Scaling parameters (override via -e flag or HF Spaces Variables)
ENV WORKERS=2
ENV MAX_CONCURRENT_ENVS=100
ENV PORT=7860
ENV HOST=0.0.0.0

# Use uvicorn directly with multi-worker support
CMD ["sh", "-c", "uvicorn server.app:app --host $HOST --port $PORT --workers $WORKERS"]
