# pipeline/personality_inference.py
#
# Infers Big Five (OCEAN) personality scores from input text.
#
# Primary method:   Minej/bert-base-personality transformer model
# Fallback method:  Linguistic feature heuristics (rule-based)
#
# The fallback activates automatically if the HuggingFace model
# fails to load (e.g., no internet, first-time download failure).
# This ensures the pipeline always produces output.

import numpy as np
from config import OCEAN_TRAITS, PERSONALITY_MODEL_NAME, MAX_TOKEN_LENGTH
from pipeline.preprocessor import preprocess
import sys

# ------------------------------------------------------------------
# Attempt to load the transformer model at import time.
# If it fails, we set MODEL_AVAILABLE = False and use fallback.
# ------------------------------------------------------------------
MODEL_AVAILABLE = False
_personality_pipeline = None

def _load_model():
    global MODEL_AVAILABLE, _personality_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print(f"[PersonalityModel] Loading model: {PERSONALITY_MODEL_NAME}")
        print("[PersonalityModel] This may take a minute on first run (downloading weights)...")
        _personality_pipeline = hf_pipeline(
            "text-classification",
            model=PERSONALITY_MODEL_NAME,
            top_k=None,  # <--- THE FIX: Forces transformers to return ALL scores, not just top 1
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        MODEL_AVAILABLE = True
        print("[PersonalityModel] Model loaded successfully.")
    except Exception as e:
        print(f"[PersonalityModel] WARNING: Could not load transformer model: {e}")
        print("[PersonalityModel] Falling back to linguistic feature-based inference.")
        MODEL_AVAILABLE = False


# ------------------------------------------------------------------
# Linguistic feature → OCEAN mapping (fallback heuristics)
#
# Based on correlations reported in:
# - Mairesse et al. (2007): Using linguistic cues for personality prediction
# - Golbeck et al. (2011): Predicting personality from Twitter
# ------------------------------------------------------------------

def _fallback_ocean_from_features(features: dict) -> dict:
    """
    Estimates OCEAN scores from linguistic features when the
    transformer model is unavailable.
    """
    certainty   = features.get("certainty_word_ratio", 0.3)
    tentative   = features.get("tentative_word_ratio", 0.3)
    exclaim     = features.get("exclamation_ratio", 0.1)
    first_pers  = features.get("first_person_ratio", 0.3)
    we_ratio    = features.get("we_ratio", 0.2)
    sent_len    = features.get("avg_sentence_length", 0.5)

    scores = {
        "Openness":          0.4 + tentative * 0.4 + we_ratio * 0.2,
        "Conscientiousness": 0.3 + certainty * 0.5 + sent_len * 0.2,
        "Extraversion":      0.3 + exclaim * 0.4 + we_ratio * 0.4,
        "Agreeableness":     0.35 + we_ratio * 0.4 + (1 - first_pers) * 0.25,
        "Neuroticism":       0.25 + first_pers * 0.4 + exclaim * 0.2,
    }

    # Normalize to [0.1, 0.9] — avoid extreme edge values
    normalized = {}
    for trait, score in scores.items():
        normalized[trait] = round(min(max(score, 0.1), 0.9), 4)

    return normalized


 # Add this to the top of your file if it's not there!

def _infer_with_model(text_chunks: list) -> dict:
    """
    Runs text chunks through the transformer model and averages
    the OCEAN scores across all chunks.
    """
    chunk_scores = []

    for chunk in text_chunks:
        if not chunk.strip():
            continue
        try:
            result = _personality_pipeline(chunk)
            if result and isinstance(result[0], list):
                label_scores = result[0]
            else:
                label_scores = result

            label_dict = {item["label"].lower(): item["score"] for item in label_scores}

            chunk_ocean = {}
            
            trait_mapping = {
                "Openness": ["openness", "opn", "o", "label_0"],
                "Conscientiousness": ["conscientiousness", "con", "c", "label_1"],
                "Extraversion": ["extraversion", "extroversion", "ext", "e", "label_2"],
                "Agreeableness": ["agreeableness", "agr", "a", "label_3"],
                "Neuroticism": ["neuroticism", "neu", "n", "label_4"]
            }

            for trait in OCEAN_TRAITS:
                score_found = False
                for possible_label in trait_mapping.get(trait, []):
                    for key in label_dict.keys():
                        if key.startswith(possible_label) or key == possible_label:
                            chunk_ocean[trait] = label_dict[key]
                            score_found = True
                            break 
                    if score_found:
                         break 
                
                if not score_found:
                    chunk_ocean[trait] = 0.5 

            chunk_scores.append(chunk_ocean)

        except Exception as e:
            # We only want to see this if something actually crashes
            print(f"\n[Error] NLP Chunk inference failed: {e}")
            continue

    if not chunk_scores:
        return {trait: 0.5 for trait in OCEAN_TRAITS}

    averaged = {}
    for trait in OCEAN_TRAITS:
        values = [c[trait] for c in chunk_scores if trait in c]
        averaged[trait] = round(np.mean(values), 4) if values else 0.5

    return averaged


def infer_batch(profiles: list) -> list:
    """
    Runs personality inference on a list of employee profiles with a clean progress indicator.
    """
    total = len(profiles)
    print(f"  [PersonalityModel] Analyzing {total} profiles through the NLP engine...")
    
    for i, profile in enumerate(profiles):
        ocean = infer_personality(profile.get("text", ""))
        profile["ocean_scores"] = {k: v for k, v in ocean.items() if k != "method"}
        profile["inference_method"] = ocean.get("method", "unknown")
        
        # Clean, single-line progress update instead of spamming the console
        progress = int(((i + 1) / total) * 100)
        sys.stdout.write(f"\r  [PersonalityModel] Progress: {progress}% complete ")
        sys.stdout.flush()
        
    print() # Print a newline when done so the next step formats correctly
    return profiles
    """
    Runs text chunks through the transformer model and averages
    the OCEAN scores across all chunks.
    """
    chunk_scores = []
    
    # Flag to ensure we only print the debug line once per run
    debug_printed = False 

    for chunk in text_chunks:
        if not chunk.strip():
            continue
        try:
            result = _personality_pipeline(chunk)
            if result and isinstance(result[0], list):
                label_scores = result[0]
            else:
                label_scores = result

            # Force all incoming labels to lowercase for matching
            label_dict = {item["label"].lower(): item["score"] for item in label_scores}

            if not debug_printed:
                print(f"  [DEBUG] Raw HuggingFace Model Labels: {list(label_dict.keys())}")
                debug_printed = True

            chunk_ocean = {}
            
            # Map the unformatted LABEL_X outputs to OCEAN traits
            trait_mapping = {
                "Openness": ["openness", "opn", "o", "label_0"],
                "Conscientiousness": ["conscientiousness", "con", "c", "label_1"],
                "Extraversion": ["extraversion", "extroversion", "ext", "e", "label_2"],
                "Agreeableness": ["agreeableness", "agr", "a", "label_3"],
                "Neuroticism": ["neuroticism", "neu", "n", "label_4"]
            }

            for trait in OCEAN_TRAITS:
                score_found = False
                for possible_label in trait_mapping.get(trait, []):
                    for key in label_dict.keys():
                        if key.startswith(possible_label) or key == possible_label:
                            chunk_ocean[trait] = label_dict[key]
                            score_found = True
                            break 
                    if score_found:
                         break 
                
                if not score_found:
                    chunk_ocean[trait] = 0.5 # Failsafe

            chunk_scores.append(chunk_ocean)

        except Exception as e:
            print(f"[PersonalityModel] Chunk inference error: {e}")
            continue

    if not chunk_scores:
        return {trait: 0.5 for trait in OCEAN_TRAITS}

    # Average scores across all chunks
    averaged = {}
    for trait in OCEAN_TRAITS:
        values = [c[trait] for c in chunk_scores if trait in c]
        averaged[trait] = round(np.mean(values), 4) if values else 0.5

    return averaged


def infer_personality(text: str) -> dict:
    """
    Main entry point. Takes raw profile text and returns
    OCEAN trait scores between 0.0 and 1.0.

    High score = trait is strongly present.
    Low score  = trait is weakly present.

    Args:
        text: Raw text from an employee's public profile.

    Returns:
        Dict mapping each OCEAN trait to a float score [0.0, 1.0].
        Also includes "method" key: "transformer" or "heuristic".
    """
    processed = preprocess(text)

    if MODEL_AVAILABLE and _personality_pipeline is not None:
        scores = _infer_with_model(processed["chunks"])
        scores["method"] = "transformer"
    else:
        scores = _fallback_ocean_from_features(processed["features"])
        scores["method"] = "heuristic"

    return scores


def infer_batch(profiles: list) -> list:
    """
    Runs personality inference on a list of employee profiles.

    Args:
        profiles: List of profile dicts (must have "text" key).

    Returns:
        Same list with "ocean_scores" added to each profile dict.
    """
    total = len(profiles)
    for i, profile in enumerate(profiles):
        print(f"[PersonalityModel] Inferring {i+1}/{total}: {profile.get('id', '?')} — {profile.get('name', '?')}")
        ocean = infer_personality(profile.get("text", ""))
        profile["ocean_scores"] = {k: v for k, v in ocean.items() if k != "method"}
        profile["inference_method"] = ocean.get("method", "unknown")

    return profiles