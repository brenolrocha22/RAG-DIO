FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY chainlit.md ./
COPY .chainlit/config.toml ./.chainlit/config.toml
COPY data ./data

EXPOSE 8000

CMD ["chainlit", "run", "src/main.py", "--host", "0.0.0.0", "--port", "8000"]
