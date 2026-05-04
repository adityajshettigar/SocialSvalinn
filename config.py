# config.py
# Central configuration for the entire project.
# All thresholds, mappings, and model choices live here
# so nothing is hardcoded inside pipeline files.

# ------------------------------------------------------------------
# Model Settings
# ------------------------------------------------------------------

# HuggingFace model for Big Five personality inference.
# Minej/bert-base-personality is a BERT model fine-tuned on the
# Essays dataset (Pennebaker & King, 1999) for OCEAN prediction.
PERSONALITY_MODEL_NAME = "Minej/bert-base-personality"

# Max tokens the model can handle in one pass.
# Longer text is chunked and averaged.
MAX_TOKEN_LENGTH = 512

# ------------------------------------------------------------------
# Big Five (OCEAN) Trait Labels
# ------------------------------------------------------------------
# These are the exact label names the model returns.
OCEAN_TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]

# ------------------------------------------------------------------
# Cialdini's Six Principles of Persuasion
# ------------------------------------------------------------------
CIALDINI_PRINCIPLES = [
    "Authority",
    "Urgency",
    "Scarcity",
    "Social Proof",
    "Liking",
    "Reciprocity"
]

# ------------------------------------------------------------------
# Susceptibility Mapping Matrix
# Maps each Big Five trait to how strongly it predicts
# susceptibility to each Cialdini principle.
#
# Values are weights (0.0 to 1.0) based on:
# - Workman (2008): Phishing susceptibility and personality
# - Vishwanath et al. (2011): Social engineering and Big Five
# - Cialdini (1984): Original principles framework
#
# Layout: rows = OCEAN traits, cols = Cialdini principles
# Order:  Authority, Urgency, Scarcity, Social Proof, Liking, Reciprocity
# ------------------------------------------------------------------
SUSCEPTIBILITY_MATRIX = {
    #                   Auth  Urg   Scar  SocPr  Like  Recip
    "Openness":        [0.2,  0.1,  0.2,  0.3,   0.5,  0.7],
    "Conscientiousness":[0.8, 0.3,  0.6,  0.2,   0.2,  0.4],
    "Extraversion":    [0.2,  0.3,  0.2,  0.8,   0.7,  0.5],
    "Agreeableness":   [0.3,  0.2,  0.2,  0.6,   0.8,  0.7],
    "Neuroticism":     [0.5,  0.9,  0.8,  0.4,   0.3,  0.3],
}

# ------------------------------------------------------------------
# Risk Thresholds
# Used to classify a susceptibility score into a risk tier.
# ------------------------------------------------------------------
RISK_THRESHOLDS = {
    "HIGH":   0.65,   # score >= 0.65 → High Risk
    "MEDIUM": 0.40,   # score >= 0.40 → Medium Risk
    # below 0.40     → Low Risk
}

# ------------------------------------------------------------------
# Departments in the Simulated Organization
# ------------------------------------------------------------------
DEPARTMENTS = ["Finance", "HR", "Engineering", "Legal", "Sales", "Operations"]

# ------------------------------------------------------------------
# Output Paths
# ------------------------------------------------------------------
OUTPUT_DIR = "output"
GRAPH_OUTPUT_FILE = "output/vulnerability_graph.png"
HEATMAP_OUTPUT_FILE = "output/susceptibility_heatmap.png"
REPORT_OUTPUT_FILE = "output/risk_report.csv"
