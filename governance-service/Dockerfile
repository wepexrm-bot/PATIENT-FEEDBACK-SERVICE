FROM python:3.11-slim
WORKDIR /app
COPY shared ./shared
COPY governance-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY governance-service/app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8010"]