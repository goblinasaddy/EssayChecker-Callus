# VeritasEssay: Evidence-Based AI Detection for College Admissions Essays

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-24%20passed-success.svg)](tests/)
[![Architecture](https://img.shields.io/badge/model-calibrated%20logistic%20regression-purple.svg)](src/models/)
[![Zero LLM Judge](https://img.shields.io/badge/LLM%20Judge-NONE%20(Deterministic)-emerald.svg)](src/inference/)

An evidence-grounded AI authorship analysis system built for the **Callus 2026 i12 HR Drive Hackathon (Project 2)**. 

Unlike black-box commercial detectors or simplistic LLM-prompted wrappers, VeritasEssay identifies suspicious passages in college admissions essays and explains the **exact measurable linguistic and statistical evidence** behind every flag.

---

## 🎯 Key Principles & Highlights

1. **No LLM as Judge**: The final verdict is **never** decided by querying an LLM ("Is this AI?"). All classifications come from deterministic, calibrated models trained on verified linguistic signals.
2. **Exact Sentence Span Grounding**: Every flagged passage maps to exact character offsets `(start_char, end_char)` in the submitted essay, paired with specific metric deviations vs. a human reference baseline.
3. **Multi-Family Feature Engineering**: Evaluates **48 candidate features** spanning Surface Stylometry, Admissions Narrative & Discourse Architecture, and StoryScope-inspired Distributional Geometry.
4. **Honest & Defensible Outputs**: The system rejects single meaningless percentages (e.g. *"73% AI"*), providing calibrated categorical assessments (`Likely Human`, `Likely AI-Assisted`, `Likely AI-Generated`, or `Uncertain`) with explicit uncertainty disclosure.
5. **Support for AI-Polished Text**: Explicitly analyzes the realistic scenario of student drafts revised or stylized by language models.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A[Admissions Essay Input] --> B[EssayCleaner & TextNormalizer\nUnicode NFKC, Strip Generation Artifacts]
    B --> C[HierarchicalSegmenter\nEssay -> Paragraph -> Sentence Exact Character Spans]
    
    C --> D1[Family A: Surface Stylometrics\nCompression Ratio, Root TTR, Burstiness Delta Variance]
    C --> D2[Family B: Admissions Discourse\nConcrete/Abstract Ratio, Personal Agency, Reflection Trajectory]
    C --> D3[Family C: Distributional Geometry\nRegularized Mahalanobis Distance to Human Centroid]
    
    D1 --> E[Calibrated Production Logistic Model\nStandardized Feature Scoring & Logit Estimation]
    D2 --> E
    D3 --> E
    
    C --> F[Sentence-Level Evidence Engine\nLocal Feature Deviation vs Fixed Human Baseline]
    
    E --> G1[Essay-Level Categorical Decision\nLikely Human | Uncertain | AI-Assisted | AI-Generated]
    F --> G2[Interactive Highlighted Spans\nExact Character Offset Grounding & Evidence Cards]
    
    G1 --> H[VeritasEssay Interactive UI & API]
    G2 --> H
```

---

## 🔬 Feature Families & Empirical Findings

| Feature Family | Top Discriminative Signals | Direction | Physical / Linguistic Interpretation |
| :--- | :--- | :--- | :--- |
| **Distributional Geometry** | `dist_human_mahalanobis` | Higher $\to$ AI | Regularized distance from the empirical human writing covariance manifold. |
| **Admissions Discourse** | `discourse_abstract_vocab_density` | Higher $\to$ AI | AI over-indexes on conceptual abstractions (*"catalyst", "multifaceted", "tapestry", "transformative"*). |
| **Surface Stylometry** | `surface_compression_ratio` | Higher $\to$ Human | Measure of token predictability and repetitiveness; human writing has higher compressibility variance. |
| **Surface Rhythm** | `surface_sent_delta_variance` | Higher $\to$ Human | Natural sentence length burstiness and local pacing modulation. |
| **Narrative Grounding** | `discourse_concrete_abstract_ratio` | Higher $\to$ Human | Ratio of physical sensory descriptions, tactile actions, and numerical specifics to abstract meta-commentary. |
| **Personal Agency** | `discourse_agency_ratio` | Higher $\to$ Human | First-person active decision verbs (*"I decided", "I rebuilt"*) vs passive states (*"I found myself"*). |

---

## 🚀 Quick Start & How to Run

### 1. Prerequisites & Installation
Ensure Python 3.10+ is installed:
```bash
git clone https://github.com/goblinasaddy/EssayChecker-Callus.git
cd EssayChecker-Callus
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (24 Tests)
```bash
python -m pytest -v
```

### 3. Run End-to-End Smoke Test
```bash
python scripts/smoke_test_app.py
```

### 4. Launch the Web Application
```bash
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser and navigate to:
👉 **`http://127.0.0.1:8000`**

---

## 📊 Evaluation Results (Held-Out Test Set)

Evaluated under strict, leakage-free group cluster splitting (zero prompt or variant overlap):

- **Accuracy**: **75.0%**
- **Precision**: **0.6667**
- **Recall**: **1.0000**
- **F1 Score**: **0.8000**
- **ROC-AUC**: **1.0000**
- **False Positive Rate**: **0.5000** (Diagnosed on complex high-register human essays)
- **False Negative Rate**: **0.0000**

---

## 🔍 Failure Case Analysis & Diagnosed False Positives

VeritasEssay includes a built-in showcase button for diagnosed false positives:
- **Case**: `sample_false_positive` (Authentic student essay describing orchestral cello composition).
- **Diagnosis**: High-register metaphorical vocabulary (*"transformative"*, *"dissonance"*, *"intricacies"*) and long flowing clauses trigger abstract density flags.
- **Safety Policy**: The UI transparently presents this case to educate admissions reviewers on the limits of automated stylometric scoring.

---

## ⚖️ Ethics, Bias & ESL Safety Notice

- **No Demographic Inference**: VeritasEssay does **not** attempt to infer an applicant's nationality, race, or native-language status from their text.
- **Probabilistic Evidence**: All outputs are statistical indicators designed to assist holistic human readers, never to serve as automated disqualification.

---

## 📁 Repository Structure

```
├── data/
│   ├── README.md                      # Dataset provenance, licenses, and schema
│   ├── models/detector_artifact_v2.json # Serialized production model & reference stats
│   └── processed/                     # Leakage-free train/val/test splits & feature tables
├── src/
│   ├── api/server.py                  # FastAPI REST API server
│   ├── features/                      # Surface, Discourse, and Distributional extractors
│   ├── inference/                     # Production detector and sentence evidence engine
│   ├── models/                        # Interpretable baselines & evaluator
│   ├── preprocessing/                 # Text cleaner, normalizer, and dataset builder
│   ├── segmentation/                  # Hierarchical exact character span segmenter
│   └── ui/                            # Modern web interface (HTML, CSS, JS)
├── scripts/
│   ├── prepare_data.py                # Dataset curation and audit script
│   ├── extract_features.py            # Tabular feature extraction pipeline
│   ├── run_experiments.py             # Baseline training and evaluation
│   ├── run_ablations.py               # Feature family ablation suite
│   ├── analyze_failures.py            # Subgroup robustness & failure diagnostics
│   └── smoke_test_app.py              # End-to-end smoke test runner
├── tests/                             # 24 automated unit & integration tests
├── reports/
│   ├── phase1-experiment-report.md    # Complete Phase 1 research foundation report
│   └── phase2-evaluation-report.md    # Complete Phase 2 production evaluation report
├── requirements.txt
└── README.md
```

---

## 🤖 AI-Tool Disclosure

In accordance with Hackathon guidelines:
- Large Language Models (GPT-4o, Claude 3.5 Sonnet, Llama 3, Gemini 1.5) were utilized strictly as controlled data generation sources to construct the synthetic admissions benchmark and evaluate polishing variations.
- No LLM is invoked during application inference.
