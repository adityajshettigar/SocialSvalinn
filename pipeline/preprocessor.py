# pipeline/preprocessor.py
#
# Handles all text cleaning before it is passed to the NLP model.
# Clean input text = more reliable personality inference.
# This module keeps all preprocessing logic in one place.

import re
import string
import spacy 

def _simple_sent_tokenize(text: str) -> list:
    """Simple regex-based sentence splitter that requires no downloads."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """
    Applies basic cleaning to raw text:
    - Lowercase
    - Remove URLs, email addresses, special characters
    - Collapse extra whitespace
    - Strip leading/trailing whitespace

    Note: We intentionally keep stopwords and punctuation here
    because the personality model was trained on natural text.
    Aggressive stopword removal would degrade its accuracy.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove mentions and hashtags (for social media input)
    text = re.sub(r"[@#]\w+", "", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def chunk_text(text: str, max_tokens: int = 400) -> list:
    """
    Splits long text into sentence-aware chunks so the transformer
    model can process it without truncation errors.

    Each chunk stays under max_tokens (estimated as words * 1.3
    to approximate subword tokenization overhead).

    Args:
        text: Cleaned input text.
        max_tokens: Approximate maximum token count per chunk.

    Returns:
        List of text chunks.
    """
    sentences = _simple_sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0
    # Approximate token count from word count
    token_limit = int(max_tokens / 1.3)

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > token_limit and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def extract_linguistic_features(text: str) -> dict:
    """
    Extracts simple linguistic signals that correlate with
    personality traits. Used as supplementary features or
    as a fallback when the transformer model is unavailable.

    Features extracted:
    - avg_sentence_length: longer sentences → higher Conscientiousness
    - exclamation_ratio: high ratio → higher Extraversion / Neuroticism
    - first_person_ratio: high I-usage → higher Neuroticism
    - we_ratio: high we-usage → higher Agreeableness / Extraversion
    - certainty_word_ratio: words like "always", "never" → Conscientiousness
    - tentative_word_ratio: words like "maybe", "perhaps" → Openness / Neuroticism

    Returns:
        Dict of feature name → float value (0.0 to 1.0 normalized)
    """
    if not text:
        return {}

    words = text.lower().split()
    sentences = _simple_sent_tokenize(text)
    total_words = len(words) if words else 1

    certainty_words = {"always", "never", "definitely", "certainly", "absolutely",
                       "must", "required", "mandatory", "every", "all"}
    tentative_words = {"maybe", "perhaps", "possibly", "sometimes", "might",
                       "could", "uncertain", "unclear", "depends", "generally"}
    first_person = {"i", "me", "my", "myself", "mine"}
    we_words = {"we", "our", "us", "together", "team", "collaborate", "colleagues"}

    features = {
        "avg_sentence_length": (
            sum(len(s.split()) for s in sentences) / len(sentences)
            if sentences else 0
        ) / 30.0,  # normalize against ~30-word baseline

        "exclamation_ratio": min(
            text.count("!") / total_words, 1.0
        ),

        "first_person_ratio": min(
            sum(1 for w in words if w in first_person) / total_words, 1.0
        ) * 5,  # scale up since I-words are naturally rare

        "we_ratio": min(
            sum(1 for w in words if w in we_words) / total_words, 1.0
        ) * 10,

        "certainty_word_ratio": min(
            sum(1 for w in words if w in certainty_words) / total_words, 1.0
        ) * 15,

        "tentative_word_ratio": min(
            sum(1 for w in words if w in tentative_words) / total_words, 1.0
        ) * 15,
    }

    # Clip all values to [0, 1]
    return {k: min(max(v, 0.0), 1.0) for k, v in features.items()}


def preprocess(text: str) -> dict:
    """
    Full preprocessing pipeline for a single profile's text.

    Returns:
        {
            "cleaned": cleaned full text string,
            "chunks": list of text chunks for model input,
            "features": dict of linguistic features,
            "word_count": int
        }
    """
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned)
    features = extract_linguistic_features(cleaned)

    return {
        "cleaned": cleaned,
        "chunks": chunks,
        "features": features,
        "word_count": len(cleaned.split()),
    }
