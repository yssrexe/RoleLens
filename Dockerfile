FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app
COPY requirements.txt requirements-docker.txt ./
# CPU wheels avoid installing CUDA libraries in the app image.
RUN pip install --no-cache-dir torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY web ./web
COPY web_app.py ./

EXPOSE 8000
CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "8000"]
