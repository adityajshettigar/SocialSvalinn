# SocialSvalinn
**Behavioral NLP Framework for Predicting Social Engineering Susceptibility**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace Models](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-orange)](https://huggingface.co/Minej/bert-base-personality)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/adityajshettigar/SocialSvalinn)

> **SocialSvalinn** bridges the gap between psycholinguistics and defensive cybersecurity. Instead of reacting to phishing clicks, this framework proactively maps human attack surfaces by analyzing public text, inferring Big Five (OCEAN) personality traits, and calculating target susceptibility to Cialdini's Six Principles of Persuasion.

---

##  Table of Contents
- [Executive Summary](#-executive-summary)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
- [Output & Telemetry](#-output--telemetry)
- [Theoretical Foundation](#-theoretical-foundation)
- [Extending the Framework (OSINT Integration)](#-extending-the-framework-osint-integration)
- [Ethical & Compliance Guardrails](#-ethical--compliance-guardrails)

---

## Executive Summary

Traditional Security Awareness Training (SAT) treats all employees as identical targets. **SocialSvalinn** recognizes that human vulnerability is highly individualized. By analyzing behavioral language inputs (via OSINT or internal communications), the pipeline generates a dynamic organizational risk topology. 

**Key Capabilities:**
* **Transformer-Based NLP:** Utilizes HuggingFace `Minej/bert-base-personality` for deep semantic trait extraction.
* **Heuristic Fallback Engine:** SpaCy-powered linguistic feature extraction ensures the pipeline runs even in air-gapped or offline environments.
* **Cohort Min-Max Scaling:** Automatically isolates the "weakest link" in an organization, preventing extreme risk vectors from being hidden behind average behavioral scores.
* **Interactive Threat Telemetry:** Outputs structured JSON and NetworkX graphs ready for SIEM ingestion or Next.js UI dashboards.

---

## System Architecture

```
Raw Text Input (OSINT / Synthetic)
      │
      ▼
[  Preprocessor Module ]  ─────▶ Cleans text, extracts linguistic features via SpaCy
      │
      ▼
[  Inference Engine ]     ─────▶ HuggingFace BERT Transformer (OCEAN trait scoring)
      │
      ▼
[  Susceptibility Scorer] ─────▶ Maps OCEAN traits to Cialdini Principles + Cohort Scaling
      │
      ▼
[  Graph Builder ]        ─────▶ Generates NetworkX Vulnerability Topology
      │
      ▼
[  Threat Telemetry ]     ─────▶ Exports CSV, JSON, Heatmaps, and Threat Network Visuals
```

### Directory Structure
```text
SocialSvalinn/
├── main.py                     # Primary pipeline execution
├── config.py                   # Constants, matrix mappings, risk thresholds
├── requirements.txt            # Python dependencies
├── .env.example                # API key template (for synthetic generation)
├── data/
│   └── synthetic_generator.py  # LLM-powered synthetic employee generation
├── pipeline/
│   ├── preprocessor.py         # Text cleaning and POS tagging
│   ├── personality_inference.py# HuggingFace & Fallback inference engines
│   ├── susceptibility_scorer.py# Min-Max scaling and risk categorization
│   └── graph_builder.py        # Network graph construction
└── visualization/
    └── visualizer.py           # Matplotlib & Seaborn chart generation
```

---

##  Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/adityajshettigar/SocialSvalinn.git
cd SocialSvalinn
```

**2. Initialize the Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Environment Variables (Required for Synthetic Mode)**
Copy the environment template and add your Groq API key (used for generating hyper-realistic synthetic employee text).
```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY="gsk_your_api_key_here"
```

---

##  Usage Guide

SocialSvalinn operates as a CLI tool. All generated reports and graphs are output to the `/output` directory.

### Full NLP Mode (Recommended)
Downloads and utilizes the HuggingFace Transformer model for high-fidelity psychological inference. *(Note: First run will download a ~438MB model to cache).*
```bash
python main.py --employees 30
```

### Heuristic / Air-Gapped Mode
Bypasses the Transformer model and uses rule-based SpaCy linguistic feature extraction. Executes in < 5 seconds. Ideal for rapid pipeline testing or offline environments.
```bash
python main.py --employees 50 --no-transformer
```

---

##  Output & Telemetry

After execution, the `output/` directory is populated with actionable SOC intelligence:

| File | Description | SIEM / Dashboard Use |
|---|---|---|
| `risk_report.csv` | Full structured data containing raw OCEAN scores, calculated susceptibility, and risk tiers. | Ideal for Splunk / Elastic ingestion. |
| `graph.json` | Complete NetworkX topological graph in JSON format. | Load into D3.js, Gephi, or React-Force-Graph. |
| `vulnerability_graph.png`| Visual network mapping employees to their specific Cialdini vulnerabilities (Red = HIGH Risk). | Exec summaries / War Room dashboards. |
| `susceptibility_heatmap.png`| Department × Principle risk density matrix. | Identifying departmental training needs. |
| `top_risk_individuals.png` | Bar chart identifying the highest-risk targets within the analyzed cohort. | Spear-phishing defense prioritization. |

---

##  Theoretical Foundation

The framework relies on a heavily researched **Susceptibility Matrix** that maps standard Big Five (OCEAN) traits to Robert Cialdini's Six Principles of Persuasion.

| Personality Trait | High Score Indicators | Primary Attack Vectors |
|---|---|---|
| **Neuroticism** | Stress-reactive, anxious, hyper-vigilant | Urgency, Scarcity |
| **Conscientiousness** | Rule-following, structured, hierarchical | Authority, Scarcity |
| **Agreeableness** | Trusting, highly cooperative, empathetic | Liking, Reciprocity, Social Proof |
| **Extraversion** | Outward-facing, highly social, status-aware| Social Proof, Liking |
| **Openness** | Curious, creative, risk-tolerant | Reciprocity, Liking |

### Risk Tiers
* **HIGH (≥ 0.65):** Critical vulnerability. The individual is highly likely to respond to this specific psychological lure. Immediate targeted SAT required.
* **MEDIUM (0.40 – 0.64):** Moderate vulnerability. General awareness training recommended.
* **LOW (< 0.40):** High resistance. The individual is naturally skeptical of this psychological lever.

---

##  Extending the Framework (OSINT Integration)

SocialSvalinn defaults to analyzing synthetic profiles to ensure privacy. To deploy this tool against real targets (e.g., Red Team engagements), replace the `generate_organization()` call in `main.py` with an OSINT scraper module.


---

##  Ethical & Compliance Guardrails

**This framework is designed exclusively for defensive cybersecurity research and organizational protection.** 

Because psycholinguistic profiling handles highly sensitive behavioral data, any deployment of SocialSvalinn on real individuals must adhere to the following strict guidelines:
1. **Consent:** Explicit, informed consent must be obtained from all analyzed personnel.
2. **Compliance:** Data handling must comply fully with local privacy laws (e.g., GDPR, CCPA, DPDP Act 2023).
3. **Anonymization:** For internal corporate use, Employee Names should be hashed or omitted prior to running the pipeline, retaining only Department and Job Title metadata.
4. **No Weaponization:** This tool must not be used to conduct malicious spear-phishing campaigns or psychological manipulation.

---
