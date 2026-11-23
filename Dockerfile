FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050 \
    HOST=0.0.0.0

WORKDIR /app
COPY requirements-cloud.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py main.py ./
COPY packetflowai ./packetflowai

EXPOSE 8050
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:server"]
