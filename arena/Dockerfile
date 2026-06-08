FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV ARENA_DB=/tmp/arena.db
EXPOSE 7860
CMD ["uvicorn", "arena.app:app", "--host", "0.0.0.0", "--port", "7860"]
