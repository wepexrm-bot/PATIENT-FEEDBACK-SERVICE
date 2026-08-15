"""SACR-compatible text-cleaning + tokenizer so converted SACR pipelines unpickle
and predict identically to their original training environment.

These functions mirror `sacr_cli.py` from the SACR project exactly:
  - `contraction_expansion` (uses `contractions` when installed, regex fallback)
  - `data_cleaning` (clean + stopword + 'not_' negation handling)
  - `LemmaTokenizer` (word_tokenize -> pos_tag -> lemmatize)

The vectorizer inside a converted pipeline references `LemmaTokenizer` from this
module, which is why this module must ship inside the sentiment-model container.
"""
import re

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    import contractions  # type: ignore
    HAS_CONTRACTIONS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_CONTRACTIONS = False

_NLTK_RESOURCES = [
    "punkt_tab", "punkt", "stopwords", "wordnet",
    "averaged_perceptron_tagger_eng", "averaged_perceptron_tagger",
]

_MODEL_STOP_WORDS: set[str] | None = None


def _ensure_nltk() -> None:
    """Download required NLTK resources exactly once (no-op if present)."""
    global _MODEL_STOP_WORDS
    if _MODEL_STOP_WORDS is not None:
        return
    for res_name in _NLTK_RESOURCES:
        try:
            if res_name in ("punkt_tab", "punkt"):
                nltk.data.find(f"tokenizers/{res_name}")
            elif res_name in ("stopwords", "wordnet"):
                nltk.data.find(f"corpora/{res_name}")
            else:
                nltk.data.find(f"taggers/{res_name}")
        except LookupError:
            nltk.download(res_name, quiet=True)
    _MODEL_STOP_WORDS = set(stopwords.words("english"))
    _MODEL_STOP_WORDS.discard("not")
    _MODEL_STOP_WORDS.update(["would", "shall", "could", "might"])


def contraction_expansion(content: str) -> str:
    if HAS_CONTRACTIONS:
        return contractions.fix(content)
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"n\'t", " not", content)
    return content


def data_cleaning(content) -> str:
    _ensure_nltk()
    if not isinstance(content, str):
        return ""
    content = contraction_expansion(content)
    content = re.sub(r"http\S+", "", content)
    content = re.sub(r"\W+", " ", content)

    tokens = []
    negate = False
    for w in content.split():
        wl = w.strip().lower()
        if wl == "not":
            negate = True
            continue
        if wl.isalpha() and wl not in _MODEL_STOP_WORDS:
            if negate:
                tokens.append(f"not_{wl}")
                negate = False
            else:
                tokens.append(wl)
    return " ".join(tokens)


class LemmaTokenizer:
    def __init__(self):
        _ensure_nltk()
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews: str):
        _ensure_nltk()
        tokens = word_tokenize(reviews)
        try:
            ptags = pos_tag(tokens)
        except LookupError:  # pragma: no cover - defensive
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            ptags = pos_tag(tokens)
        lemmas = []
        for word, tag in ptags:
            pos = "n"
            if tag.startswith("V"):
                pos = "v"
            elif tag.startswith("J"):
                pos = "a"
            elif tag.startswith("R"):
                pos = "r"
            lemmas.append(self.wordnetlemma.lemmatize(word, pos))
        return lemmas