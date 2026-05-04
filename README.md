# Behavioral NLP Framework for Predicting Social Engineering Susceptibility

A research prototype that analyzes publicly available text from organizational members,
infers their personality traits using the Big Five (OCEAN) model, and maps those traits
to susceptibility scores for each of Cialdini's six principles of persuasion.

---

## Project Structure

```
psyber_trace/
├── main.py                         ← Entry point — run this
├── config.py                       ← All constants, thresholds, mappings
├── requirements.txt                ← Python dependencies
│
├── data/
│   ├── __init__.py
│   └── synthetic_generator.py      ← Generates synthetic employee profiles
│
├── pipeline/
│   ├── __init__.py
│   ├── preprocessor.py             ← Text cleaning and linguistic feature extraction
│   ├── personality_inference.py    ← OCEAN trait inference (transformer + fallback)
│   ├── susceptibility_scorer.py    ← Maps OCEAN to Cialdini susceptibility scores
│   └── graph_builder.py            ← Builds the NetworkX vulnerability graph
│
├── visualization/
│   ├── __init__.py
│   └── visualizer.py               ← All charts, heatmaps, network graphs, CSV export
│
└── output/                         ← All results saved here (created at runtime)
    ├── susceptibility_heatmap.png
    ├── vulnerability_graph.png
    ├── top_risk_individuals.png
    ├── risk_report.csv
    ├── graph.json
    └── synthetic_profiles.json
```

---

## Setup Instructions

### Step 1 — Clone or copy the project folder

```bash
cd psyber_trace
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A — Fast mode (no internet, no model download)
Uses rule-based linguistic heuristics for personality inference.
Runs in under 5 seconds. Good for testing the pipeline.

```bash
python main.py --no-transformer
```

### Option B — Full transformer mode (requires internet, first run ~2-3 min)
Downloads and uses the `Minej/bert-base-personality` BERT model from HuggingFace.
Produces more accurate OCEAN scores. Model is cached after first download.

```bash
python main.py
```

### Controlling the number of employees

```bash
# Run with 50 employees
python main.py --employees 50

# Run with 100 employees in heuristic mode
python main.py --employees 100 --no-transformer
```

---

## What Gets Generated

After the run completes, the `output/` folder contains:

| File | Description |
|---|---|
| `susceptibility_heatmap.png` | Department × Principle heatmap (red = high risk) |
| `vulnerability_graph.png` | Network graph: employees → departments → principles |
| `top_risk_individuals.png` | Bar chart of highest-risk individuals |
| `risk_report.csv` | Full structured data — OCEAN scores, susceptibility scores, risk tiers |
| `graph.json` | NetworkX graph in JSON format (load into D3.js or Gephi) |
| `synthetic_profiles.json` | Raw generated employee profiles |

---

## How the Pipeline Works

```
Raw Text
    ↓
[Preprocessor]         → Cleans text, extracts linguistic features
    ↓
[Personality Inference] → Infers Big Five (OCEAN) scores per person
    ↓
[Susceptibility Scorer] → Maps OCEAN traits to Cialdini principle scores
    ↓
[Graph Builder]         → Builds a NetworkX organizational vulnerability graph
    ↓
[Visualizer]            → Generates heatmaps, network graphs, CSV report
```

---

## The Susceptibility Mapping

The core of the framework is a **mapping matrix** that translates personality
trait scores into persuasion susceptibility scores.

Example:

| Trait | High Score Means | Most Susceptible To |
|---|---|---|
| Neuroticism | Stress-reactive, anxious | Urgency, Scarcity |
| Conscientiousness | Rule-following, hierarchical | Authority, Scarcity |
| Agreeableness | Trusting, cooperative | Liking, Reciprocity, Social Proof |
| Extraversion | Social, outward-facing | Social Proof, Liking |
| Openness | Curious, creative | Reciprocity, Liking |

---

## Risk Tiers

| Score Range | Tier | Meaning |
|---|---|---|
| ≥ 0.65 | HIGH | Strong training priority — likely to respond to this attack vector |
| 0.40 – 0.64 | MEDIUM | Moderate risk — needs awareness training |
| < 0.40 | LOW | Relatively resistant to this specific principle |

---

## Extending the Project

**To use real text input instead of synthetic profiles:**
Edit `main.py` and replace the `generate_organization()` call with your own
data loader. Each profile dict needs at minimum:
```python
{
    "id": "EMP-001",
    "name": "Name Here",
    "department": "Finance",
    "title": "Analyst",
    "text": "The public text you want to analyze..."
}
```

**To add a new Cialdini principle:**
1. Add its name to `CIALDINI_PRINCIPLES` in `config.py`
2. Add its weight column to `SUSCEPTIBILITY_MATRIX` in `config.py`

**To swap the personality model:**
Change `PERSONALITY_MODEL_NAME` in `config.py` to any HuggingFace model
that returns Big Five (OCEAN) labels.

---

## Research References

- Cialdini, R. B. (1984). *Influence: The Psychology of Persuasion.*
- Mairesse et al. (2007). Using linguistic cues for the automatic recognition of personality.
- Workman, M. (2008). Wisecrackers: A theory-grounded investigation of phishing and pretext social engineering.
- Kosinski, M. et al. (2013). Private traits and attributes are predictable from digital records.
- Vishwanath, A. et al. (2011). Why do people get phished?

---

## Notes on Ethics

This framework is designed exclusively for **defensive research**.
The synthetic data generator is the default input source specifically
to avoid processing any real individual's data without consent.
Any deployment against real organizational data requires:
- Explicit informed consent from all employees
- Institutional ethics approval (IRB or equivalent)
- Data handling compliance with DPDP Act 2023 (India) / GDPR

---

*Research Prototype — v1.0*
