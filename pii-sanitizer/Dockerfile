FROM python:3.11-slim
WORKDIR /app
COPY shared ./shared
COPY pii-sanitizer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_lg
COPY pii-sanitizer/app ./app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]