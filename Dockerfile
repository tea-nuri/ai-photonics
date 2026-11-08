FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
WORKDIR /app
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY photonics ./photonics
COPY predict.py .
COPY weights/model.pt ./weights/model.pt
ENTRYPOINT ["python", "predict.py"]
CMD ["--help"]
