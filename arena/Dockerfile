FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The Space repo root holds the arena/ package contents (app.py, etc.). app.py
# uses package-relative imports (`from . import db`), so copy everything into an
# `arena/` package dir and serve `arena.app:app` — flattening would break imports.
COPY . ./arena
ENV ARENA_DB=/tmp/arena.db
EXPOSE 7860
CMD ["uvicorn", "arena.app:app", "--host", "0.0.0.0", "--port", "7860"]
