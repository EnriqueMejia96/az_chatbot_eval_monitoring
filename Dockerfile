FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Streamlit startup
CMD ["streamlit", "run", "app.py", "--server.fileWatcherType","none"]
