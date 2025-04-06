FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001
ENV MODEL_ID='sentence-transformers/all-MiniLM-L6-v2'

CMD ["uvicorn", "embedding_model_host_server_script.py:app", "--host", "0.0.0.0", "--port", "8001"]