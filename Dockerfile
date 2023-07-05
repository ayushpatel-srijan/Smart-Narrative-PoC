FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8502

ENTRYPOINT ["streamlit", "run", "app_2.py", "--server.port=8502", "--server.address=0.0.0.0"]
