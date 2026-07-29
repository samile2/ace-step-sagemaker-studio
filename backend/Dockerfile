FROM python:3.11-slim

RUN pip install --no-cache-dir \
    boto3 \
    fastapi \
    uvicorn

WORKDIR /app

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8090"]
