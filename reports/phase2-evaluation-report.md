# Phase 2 Evaluation Report
**Project 2: Evidence-Based AI Detection for College Admissions Essays**  
*Phase 2: Production Detector, Sentence Evidence Attribution Engine, and Interactive Web Application*

---

## 1. Final Architecture

The Phase 2 system (`VeritasEssay`) implements a multi-tier, interpretable, deterministic AI authorship analysis architecture designed specifically for college admissions essays. It strictly enforces the core constraint: **no LLM is ever used as a judge or arbiter of authorship.**

```mermaid
flowchart TD
    A[Admissions Essay Input] --> B[EssayCleaner & TextNormalizer\nUnicode NFKC, Strip Generation Headers/Notes]
    B --> C[HierarchicalSegmenter\nEssay -> Paragraph -> Sentence Exact Character Spans]
    
    C --> D1[Surface Stylometry & Predictability\nCompression Ratio, Root TTR, Delta Variance, Entropy]
    C --> D2[Admissions Discourse & Narrative\nConcrete/Abstract Ratio, Personal Agency, Reflection Trajectory]
    C --> D3[Distributional Geometry Engine\nRegularized Mahalanobis Distance to Fixed Human Centroid]
    
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

## 2. Model & Version

- **Model Version**: `v2.0.0`
- **Model Family**: Standardized L2-Regularized Calibrated Logistic Regression.
- **Inference Mode**: Purely deterministic (zero external network calls, zero stochastic sampling).
- **Artifact Serialization**: `data/models/detector_artifact_v2.json` contains:
  - Exact 48-feature schema and standardized weights
  - Scaler training means and variance scales
  - Fixed empirical human reference statistics ($\mu, \sigma, \text{median}, P_{10}, P_{90}$)
  - Regularized human covariance matrix inverse $(\Sigma_{human} + \lambda I)^{-1}$
  - Calibration decision thresholds

---

## 3. Feature Subset & Top Signals

A total of 48 features are evaluated across 3 families. The top 10 discriminative features in production:

| Feature Name | Family | Model Coefficient | Direction | Physical / Linguistic Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| `dist_human_mahalanobis` | Distributional | **+0.4752** | Higher $\to$ AI | Regularized statistical distance from human admissions covariance manifold. |
| `discourse_abstract_vocab_density` | Discourse | **+0.4242** | Higher $\to$ AI | Concentration of conceptual abstractions (*"catalyst", "multifaceted", "tapestry", "transformative"*). |
| `surface_compression_ratio` | Surface | **-0.3426** | Higher $\to$ Human | Measure of token predictability and repetitiveness; human writing has higher compressibility variance. |
| `dist_human_euclidean` | Distributional | **+0.3228** | Higher $\to$ AI | Euclidean distance from human feature centroid. |
| `surface_root_ttr` | Surface | **-0.2695** | Higher $\to$ Human | Length-stabilized vocabulary richness. |
| `surface_emdash_rate` | Surface | **-0.2626** | Higher $\to$ Human | Idiosyncratic parenthetical punctuation rhythm. |
| `surface_bigram_entropy` | Surface | **-0.2516** | Higher $\to$ Human | Transition entropy between adjacent word pairs. |
| `surface_word_entropy` | Surface | **-0.2503** | Higher $\to$ Human | Lexical information diversity. |
| `discourse_concrete_abstract_ratio` | Discourse | **-0.2499** | Higher $\to$ Human | Situational grounding (sensory actions, physical objects, numbers) relative to abstract commentary. |
| `surface_para_len_mean` | Surface | **-0.2436** | Higher $\to$ Human | Average paragraph chunk length. |

---

## 4. Threshold & Calibration Method

Thresholds are established using training/validation distributions without tuning against the held-out test set:

- **$P(\text{AI}) < 0.35$** $\longrightarrow$ **Likely Human** (`LIKELY_HUMAN`): Authentic student pacing, concrete grounding, and active first-person agency.
- **$0.35 \le P(\text{AI}) < 0.65$** $\longrightarrow$ **Uncertain / Mixed Evidence** (`UNCERTAIN`): Balanced or conflicting signals; definite authorship cannot be asserted.
- **$0.65 \le P(\text{AI}) < 0.85$** $\longrightarrow$ **Likely AI-Assisted / Polished** (`LIKELY_AI_ASSISTED`): Human narrative ideas combined with smoothed burstiness and elevated abstract vocabulary.
- **$P(\text{AI}) \ge 0.85$** $\longrightarrow$ **Likely AI-Generated** (`LIKELY_AI_GENERATED`): Strong divergence across multiple feature families (high Mahalanobis distance, abstract buzzwords, formulaic conclusions).

---

## 5. Held-Out Test Set Results

Evaluated on the strictly held-out Test Set (2 Human, 2 AI, zero prompt/group leakage):

| Metric | Logistic Regression (Production) | Random Forest Baseline |
| :--- | :--- | :--- |
| **Accuracy** | **75.0%** (3/4) | **75.0%** (3/4) |
| **Precision** | **0.6667** | **0.6667** |
| **Recall** | **1.0000** | **1.0000** |
| **F1 Score** | **0.8000** | **0.8000** |
| **ROC-AUC** | **1.0000** | **1.0000** |
| **False Positive Rate (FPR)** | **0.5000** (1/2) | **0.5000** (1/2) |
| **False Negative Rate (FNR)** | **0.0000** (0/2) | **0.0000** (0/2) |

---

## 6. AI-Polished Evaluation

- **Benchmark**: Controlled synthetic polishing benchmark (`CORPUS_AI_POLISHED_V1`) where authentic student drafts were refined for grammar, style, and flow by GPT-4o and Claude 3.5 Sonnet.
- **Observed Behavior**:
  - Unedited Human Draft: $P(\text{AI}) = 0.033$ (0 flagged sentences).
  - Polished Human Draft: $P(\text{AI}) = 0.854$ (2 flagged sentences).
- **Linguistic Cause**: Polishing removes authentic informal phrasing and replaces concrete action descriptions with latinate abstractions (*"vibrant dumpling-making haven"*, *"master the delicate art"*, *"stabilizing foundation"*), elevating both `discourse_abstract_vocab_density` and `dist_human_mahalanobis`.

---

## 7. Failure Case Analysis

### Diagnosed False Positive: `sample_false_positive` (`human_2e48bd75`)
- **Ground Truth**: Human (Authentic student narrative on orchestral cello composition)
- **Model Output**: `Likely AI-Assisted` ($P(\text{AI}) = 0.790$)
- **Sentence Flagged**:
  > *"When our youth orchestra premiered the third movement, listening to sixty musicians breathe life into melodies that had previously existed only inside my head was transformative."*
- **Diagnostic Root Cause**:
  1. Contains high-register metaphorical vocabulary (*"transformative"*, *"dissonance"*, *"intricacies"*).
  2. Sentences are syntactically complex with long, flowing clauses (averaging 32.4 words/sentence).
  3. Lacks punctuation interruptions (zero em-dashes).
- **Safety Handling**: The UI explicitly displays this case under the **"Diagnosed False Positive"** showcase button, explaining to admissions officers that articulate artistic writing can trigger abstract vocabulary flags.

---

## 8. ESL / Bias Safety Audit

- **Evaluation Limitation**: The held-out test split did not contain ethically verified non-native demographic records.
- **Strict Compliance**: The system refuses to infer English proficiency or nationality from writing style.
- **UI Guardrail**: Every analysis report displays an explicit notice stating: *"This system does not infer or evaluate applicant demographic background or English proficiency. Highly articulate or non-native writing styles may exhibit idiosyncratic feature profiles."*

---

## 9. Evidence Generation Methodology

The sentence evidence engine (`src/inference/evidence_engine.py`) operates as follows:
1. Segments essay into exact character spans without altering whitespace.
2. Extracts local sentence metrics:
   - Abstract buzzword density (with exact keyword matching)
   - Formulaic moral/takeaway phrases (*"in conclusion"*, *"this taught me that"*)
   - Personal agency (active action verbs vs passive observation/feeling states)
   - Sentence length and pacing outliers
3. Computes $Z$-score against the training human reference distribution:
   $$Z = \frac{x_{\text{observed}} - \mu_{\text{human}}}{\sigma_{\text{human}}}$$
4. Attaches structured evidence records with `start_char`, `end_char`, `observed_value`, `reference_mean`, `deviation_z`, and plain-language explanation.

---

## 10. Known Limitations

1. **Short Text Sensitivity**: Essays under 100 words have reduced statistical reliability for rhythm variance.
2. **Elevated High-Register Diction**: Highly polished, metaphor-dense human essays may trigger elevated abstract vocabulary scores.
3. **Reference Corpus Size**: Reference distributions are estimated from 21 verified admissions records; larger reference pools will further refine covariance estimates.

---

## 11. Reproducibility Instructions

### 1. Run Complete Test Suite
```bash
python -m pytest -v
```
*(All 24 unit and integration tests must pass)*

### 2. Run End-to-End Smoke Test
```bash
python scripts/smoke_test_app.py
```

### 3. Launch Local Web Application
```bash
python -m uvicorn src.api.server:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000` to interact with the full web interface.

---

## 12. Summary of Changes from Phase 1 to Phase 2

1. **Production Serialization**: Implemented `ProductionModelArtifact` bundling model weights, scaler, reference statistics, and thresholds into a standalone artifact.
2. **Deterministic Inference Pipeline**: Implemented `AdmissionsAIDetector` with input validation, edge case handling, and multi-tier categorical outputs.
3. **Sentence Evidence Engine**: Implemented `EvidenceEngine` computing local $Z$-score deviations and exact character span grounding.
4. **FastAPI Backend & Interactive UI**: Built complete REST API and modern glassmorphic web application with interactive sentence inspector and benchmark sample loaders.
5. **Comprehensive Test Coverage**: Expanded test suite from 15 to 24 tests covering end-to-end inference, API integration, and span fidelity.

---

## 13. Future Improvements (Phase 3 Roadmap)

1. **Dynamic Span Hover Tooltips**: Add inline hover tooltips on highlighted spans in addition to the side drawer.
2. **Custom Reference Profile Selection**: Allow admissions officers to toggle reference profiles (e.g. STEM Personal Statement vs Humanities/Creative Statement).
3. **Expanded International Student Calibration**: Collect ethically documented, verified international applicant drafts to further mitigate false positive rates.

*End of Phase 2 Evaluation Report.*
