# API + Dashboard juntos (para Docker local o Railway/Render)
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 8000

# Dashboard (Streamlit) + API (FastAPI) en paralelo
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0"]
