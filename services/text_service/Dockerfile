FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=10000 \
    OMP_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the 102MB Q4 ONNX model and tokenizer at build time
RUN python3 -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download(repo_id='tooape/embeddinggemma-300m-qat-q8-ONNX', filename='onnx/model_q4.onnx', local_dir='/app/model'); \
    hf_hub_download(repo_id='tooape/embeddinggemma-300m-qat-q8-ONNX', filename='tokenizer.json', local_dir='/app/model')"

COPY app.py .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
