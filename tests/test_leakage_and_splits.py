"""Unit tests for split leakage prevention and dataset deduplication."""
import pytest
from src.preprocessing.dataset_builder import AdmissionsDatasetBuilder


def test_audit_deduplicates_exact_and_near_duplicates():
    builder = AdmissionsDatasetBuilder(min_word_count=5, min_char_count=10)

    records = [
        {"text": "This is a unique admissions essay text about discovering coding and building an app.", "label": "human", "group_id": "g1"},
        {"text": "This is a unique admissions essay text about discovering coding and building an app.", "label": "human", "group_id": "g2"}, # Exact dup
        {"text": "This is a unique admissions essay text about discovering coding and building an application.", "label": "human", "group_id": "g3"}, # Near dup prefix
        {"text": "Completely different essay text describing cellist performance and orchestral compositions in youth.", "label": "human", "group_id": "g4"},
    ]

    cleaned, stats = builder.audit_and_deduplicate(records)
    assert len(cleaned) == 2
    assert stats["exact_duplicates_dropped"] == 1
    assert stats["near_duplicates_dropped"] == 1


def test_create_grouped_splits_guarantees_zero_leakage():
    builder = AdmissionsDatasetBuilder()

    records = []
    for g_idx in range(12):
        grp_name = f"prompt_grp_{g_idx}"
        for s_idx in range(3):
            records.append({
                "text": f"Essay content for group {g_idx} sample {s_idx}. Long enough text to meet threshold.",
                "label": "human" if s_idx % 2 == 0 else "ai",
                "binary_label": 0 if s_idx % 2 == 0 else 1,
                "group_id": grp_name,
                "topic_category": "personal_growth",
                "model_family": "gpt4o" if s_idx % 2 != 0 else None,
                "word_count": 50,
            })

    train_df, val_df, test_df, split_stats = builder.create_grouped_splits(records, seed=42)

    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    # Strict assertion: zero intersection across split sets
    assert len(train_groups.intersection(val_groups)) == 0
    assert len(train_groups.intersection(test_groups)) == 0
    assert len(val_groups.intersection(test_groups)) == 0

    assert len(train_df) + len(val_df) + len(test_df) == len(records)
