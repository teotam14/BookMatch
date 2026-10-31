FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version ./

RUN uv sync --frozen --no-dev

COPY app/ app/
COPY assets/ assets/
COPY data/ data/
COPY models/ models/

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]