FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

# The project uses only Python's standard library. BotHost mounts persistent
# storage at /app/data, where DATABASE_PATH should point in production.
RUN mkdir -p /app/data

CMD ["python3", "-m", "src.main"]
