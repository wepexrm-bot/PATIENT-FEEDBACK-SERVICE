FROM python:3.11-slim
WORKDIR /app
COPY shared ./shared
COPY sentiment-model/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m nltk.downloader punkt punkt_tab stopwords wordnet \
       averaged_perceptron_tagger averaged_perceptron_tagger_eng
# Bake model weights + tokenizer-compat metadata so the image runs air-gapped.
# Produce them first with:  python scripts/convert_sacr_model.py --model-dir <sacr_out>
COPY sentiment-model/app ./app
COPY sentiment-model/weights ./weights
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]