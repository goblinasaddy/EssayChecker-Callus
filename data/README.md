# Dataset Registry & Provenance Documentation

This document records the exact provenance, licensing, collection methodology, schema, and known limitations for all datasets used in Phase 1 of Project 2 (Evidence-Based AI Detection for College Admissions Essays).

## 1. Provenance & Source Registry

| Dataset ID | Source / Origin | Type | Domain | Generation Method / Model | Verified ESL Metadata | License / Terms | Known Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `CORPUS_HUMAN_ADM_V1` | Curated Open College Admissions Essays & Personal Statements (Common App, Ivy League & University Admissions Archive) | Human | College Admissions Essays | Authentic student authorship (Pre-LLM era / verified student collections) | Documented per record where available; otherwise `unspecified` | Open Educational / Research Use | Variance in student writing maturity; predominantly US higher ed context |
| `CORPUS_AI_GEN_V1` | Controlled generation across 7 standard Common App prompts | AI Generated | College Admissions Essays | Multi-model prompt suite: `gpt-4o`, `gpt-3.5-turbo`, `claude-3-5-sonnet`, `llama-3-70b`, `gemini-1.5-pro` (temp 0.6–0.9) | N/A (Machine-generated) | Research benchmark | Prompts emulate real student prompts, but synthetic generation may miss real-world student idiosyncratic formatting |
| `CORPUS_AI_POLISHED_V1` | Controlled synthetic polishing benchmark | Synthetic AI Polished | College Admissions Essays | Human draft essays transformed via light-to-medium stylistic and grammatical polishing prompts (`gpt-4o`, `claude-3-5-sonnet`) | Preserved from source human draft | Research benchmark | Clearly designated as synthetic/controlled; does not represent all possible human editing workflows |

## 2. Dataset Schema

Every sample in `data/processed/admissions_corpus.jsonl` follows this strict JSON schema:

```json
{
  "essay_id": "string (unique)",
  "group_id": "string (prompt/family cluster key for leakage-free splitting)",
  "label": "human | ai | synthetic_polished",
  "binary_label": 0 (human) or 1 (ai / synthetic_polished),
  "text": "string (raw essay text)",
  "topic_category": "personal_growth | overcoming_adversity | intellectual_curiosity | community_leadership | challenging_beliefs | creative_expression",
  "model_family": "null | gpt4o | gpt35 | claude35 | llama3 | gemini15",
  "generation_prompt": "null | string",
  "word_count": 0,
  "esl_metadata": "verified_native | verified_esl | unspecified",
  "provenance": {
    "source_dataset": "string",
    "license": "string",
    "is_synthetic": true | false
  }
}
```

## 3. Leakage Prevention Protocol

1. **Grouped Splitting**: Essays sharing the same underlying prompt cluster, source essay family, or author group are assigned identical `group_id` values.
2. **Zero Split Crossing**: Group splitting guarantees that no prompt variant, human draft, or its AI-polished synthetic counterpart crosses the Train, Validation, and Test partitions.
3. **Exact & Near-Duplicate Auditing**: MinHash / SHA-256 deduplication rejects identical or near-identical text submissions before dataset finalization.
