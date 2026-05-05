# pipeline/susceptibility_scorer.py
#
# Maps Big Five (OCEAN) personality scores to susceptibility
# scores for each of Cialdini's six persuasion principles.
#
# The core math:
#   susceptibility(principle) = weighted sum of OCEAN scores
#   where weights come from the SUSCEPTIBILITY_MATRIX in config.py
#
# Then we assign a risk tier (HIGH / MEDIUM / LOW) per principle.

import numpy as np
from config import (
    OCEAN_TRAITS,
    CIALDINI_PRINCIPLES,
    SUSCEPTIBILITY_MATRIX,
    RISK_THRESHOLDS,
)


def compute_susceptibility(ocean_scores: dict) -> dict:
    """
    Computes a susceptibility score for each Cialdini principle
    given a person's OCEAN trait scores.

    Math:
        For each principle P:
            score(P) = sum over all traits T of (ocean_score(T) * weight(T, P))
            normalized by the maximum possible score for P.

    Args:
        ocean_scores: Dict of OCEAN trait → float score [0.0, 1.0]

    Returns:
        Dict of principle → raw susceptibility score [0.0, 1.0]
    """
    principle_scores = {}

    for idx, principle in enumerate(CIALDINI_PRINCIPLES):
        raw_score = 0.0
        max_possible = 0.0

        for trait in OCEAN_TRAITS:
            trait_score = ocean_scores.get(trait, 0.5)
            weight = SUSCEPTIBILITY_MATRIX[trait][idx]
            raw_score += trait_score * weight
            max_possible += weight  # max if trait_score = 1.0

        # Normalize to [0, 1] by dividing by max possible
        normalized = raw_score / max_possible if max_possible > 0 else 0.0
        principle_scores[principle] = round(normalized, 4)

    return principle_scores


def classify_risk(score: float) -> str:
    """
    Converts a numeric susceptibility score into a risk tier label.

    Args:
        score: Float in [0.0, 1.0]

    Returns:
        "HIGH", "MEDIUM", or "LOW"
    """
    if score >= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif score >= RISK_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    else:
        return "LOW"


def get_top_vulnerabilities(principle_scores: dict, top_n: int = 3) -> list:
    """
    Returns the top N most susceptible principles for a profile,
    sorted by score descending.

    Returns:
        List of (principle, score, risk_tier) tuples
    """
    sorted_principles = sorted(
        principle_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return [
        (principle, score, classify_risk(score))
        for principle, score in sorted_principles
    ]


def score_profile(profile: dict) -> dict:
    """
    Scores a single employee profile for social engineering
    susceptibility.

    Args:
        profile: Dict with "ocean_scores" key populated by
                 the personality inference step.

    Returns:
        The same profile dict with these keys added:
        - susceptibility_scores: dict of principle → float
        - risk_tiers: dict of principle → risk label
        - top_vulnerabilities: list of (principle, score, tier)
        - overall_risk_score: mean susceptibility across all principles
    """
    ocean = profile.get("ocean_scores", {})

    if not ocean:
        # Neutral default if OCEAN scores are missing
        ocean = {trait: 0.5 for trait in OCEAN_TRAITS}

    sus_scores = compute_susceptibility(ocean)
    risk_tiers = {p: classify_risk(s) for p, s in sus_scores.items()}
    top_vulns = get_top_vulnerabilities(sus_scores)

    # Overall risk = mean susceptibility score
    overall_risk = round(np.mean(list(sus_scores.values())), 4)

    profile["susceptibility_scores"] = sus_scores
    profile["risk_tiers"] = risk_tiers
    profile["top_vulnerabilities"] = top_vulns
    profile["overall_risk_score"] = overall_risk

    return profile


def score_organization(profiles: list) -> list:
    """
    Scores all profiles, applying cohort-based normalization to stretch variance
    and highlight true organizational outliers.
    """
    # Step 1: Calculate raw scores for everyone
    for profile in profiles:
        ocean = profile.get("ocean_scores", {t: 0.5 for t in OCEAN_TRAITS})
        profile["raw_susceptibility"] = compute_susceptibility(ocean)

    # Step 2: Find the min and max raw scores across the cohort for each principle
    mins = {p: 1.0 for p in CIALDINI_PRINCIPLES}
    maxs = {p: 0.0 for p in CIALDINI_PRINCIPLES}
    
    for p in profiles:
        for principle, raw_score in p["raw_susceptibility"].items():
            if raw_score < mins[principle]: mins[principle] = raw_score
            if raw_score > maxs[principle]: maxs[principle] = raw_score

    # Step 3: Apply Min-Max scaling to stretch scores from 0.0 to 1.0
    for profile in profiles:
        scaled_scores = {}
        for principle, raw in profile["raw_susceptibility"].items():
            denominator = maxs[principle] - mins[principle]
            # Avoid division by zero if everyone scores exactly the same
            scaled = (raw - mins[principle]) / denominator if denominator > 0 else 0.5
            scaled_scores[principle] = round(scaled, 4)
            
        profile["susceptibility_scores"] = scaled_scores
        profile["risk_tiers"] = {p: classify_risk(s) for p, s in scaled_scores.items()}
        profile["top_vulnerabilities"] = get_top_vulnerabilities(scaled_scores)
        profile["overall_risk_score"] = round(max(scaled_scores.values()), 4)

    return profiles


def aggregate_department_risk(profiles: list) -> dict:
    """
    Computes average susceptibility scores per department and
    per Cialdini principle. This is what gets visualized in the
    department-level risk heatmap.

    Returns:
        Dict of department → {principle → avg_score}
    """
    from collections import defaultdict

    dept_data = defaultdict(list)
    for profile in profiles:
        dept = profile.get("department", "Unknown")
        dept_data[dept].append(profile.get("susceptibility_scores", {}))

    dept_averages = {}
    for dept, score_list in dept_data.items():
        dept_averages[dept] = {}
        for principle in CIALDINI_PRINCIPLES:
            values = [s.get(principle, 0.5) for s in score_list]
            dept_averages[dept][principle] = round(np.mean(values), 4)

    return dept_averages
