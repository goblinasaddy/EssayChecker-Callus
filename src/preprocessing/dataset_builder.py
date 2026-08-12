"""Dataset builder and integrity auditor for admissions essay corpora."""
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.preprocessing.cleaner import EssayCleaner
from src.preprocessing.normalizer import TextNormalizer


class AdmissionsDatasetBuilder:
    """
    Constructs, audits, deduplicates, and partitions the admissions essay benchmark.
    Guarantees strict group-level isolation across Train, Validation, and Test splits.
    """

    def __init__(self, cleaner: EssayCleaner = None, min_word_count: int = 30, min_char_count: int = 50):
        self.cleaner = cleaner or EssayCleaner()
        self.normalizer = TextNormalizer()
        self.min_word_count = min_word_count
        self.min_char_count = min_char_count

    def audit_and_deduplicate(self, records: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Audits records, removes exact and near-duplicates, cleans text, and collects audit stats.
        """
        seen_hashes = set()
        seen_normalized_prefixes = set()
        cleaned_records = []
        
        stats = {
            "total_raw": len(records),
            "exact_duplicates_dropped": 0,
            "near_duplicates_dropped": 0,
            "empty_or_malformed_dropped": 0,
            "cleaned_total": 0,
            "by_label": {},
            "by_topic": {},
            "by_model": {},
        }

        for rec in records:
            raw_text = rec.get("text", "")
            if not raw_text or len(raw_text.strip()) < self.min_char_count:
                stats["empty_or_malformed_dropped"] += 1
                continue

            # Clean and normalize
            cleaned_text, audit_meta = self.cleaner.clean(raw_text)
            if audit_meta["is_empty"] or len(cleaned_text.split()) < self.min_word_count:
                stats["empty_or_malformed_dropped"] += 1
                continue

            # Exact hash check
            text_hash = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
            if text_hash in seen_hashes:
                stats["exact_duplicates_dropped"] += 1
                continue
            seen_hashes.add(text_hash)

            # Near duplicate check (first 60 chars normalized)
            norm_prefix = re.sub(r"\W+", "", cleaned_text[:60].lower())
            if len(norm_prefix) >= 30 and norm_prefix in seen_normalized_prefixes:
                stats["near_duplicates_dropped"] += 1
                continue
            if len(norm_prefix) >= 30:
                seen_normalized_prefixes.add(norm_prefix)

            words = cleaned_text.split()
            word_count = len(words)

            rec_copy = dict(rec)
            rec_copy["text"] = cleaned_text
            rec_copy["word_count"] = word_count
            rec_copy["text_hash"] = text_hash

            # Ensure binary label: 0 for human, 1 for AI / AI-polished
            lbl = rec_copy.get("label", "human")
            rec_copy["binary_label"] = 0 if lbl == "human" else 1

            # Tally stats
            stats["by_label"][lbl] = stats["by_label"].get(lbl, 0) + 1
            topic = rec_copy.get("topic_category", "unspecified")
            stats["by_topic"][topic] = stats["by_topic"].get(topic, 0) + 1
            model = rec_copy.get("model_family", "human")
            stats["by_model"][str(model)] = stats["by_model"].get(str(model), 0) + 1

            cleaned_records.append(rec_copy)

        stats["cleaned_total"] = len(cleaned_records)
        return cleaned_records, stats

    def create_grouped_splits(
        self, records: List[Dict], train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Partitions records into Train, Validation, and Test sets based on `group_id`.
        Guarantees that ALL essays sharing a group_id remain strictly within one split.
        """
        df = pd.DataFrame(records)
        unique_groups = df["group_id"].unique()
        
        np.random.seed(seed)
        shuffled_groups = np.random.permutation(unique_groups)

        # Allocate groups to target ratios
        total_samples = len(df)
        target_train = total_samples * train_ratio
        target_val = total_samples * val_ratio

        train_groups = []
        val_groups = []
        test_groups = []

        current_train_count = 0
        current_val_count = 0

        for grp in shuffled_groups:
            grp_count = len(df[df["group_id"] == grp])
            if current_train_count + grp_count <= target_train or (len(train_groups) == 0):
                train_groups.append(grp)
                current_train_count += grp_count
            elif current_val_count + grp_count <= target_val or (len(val_groups) == 0):
                val_groups.append(grp)
                current_val_count += grp_count
            else:
                test_groups.append(grp)

        train_df = df[df["group_id"].isin(train_groups)].copy().reset_index(drop=True)
        val_df = df[df["group_id"].isin(val_groups)].copy().reset_index(drop=True)
        test_df = df[df["group_id"].isin(test_groups)].copy().reset_index(drop=True)

        split_stats = {
            "total_groups": len(unique_groups),
            "train_groups": len(train_groups),
            "val_groups": len(val_groups),
            "test_groups": len(test_groups),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "train_ratio_actual": len(train_df) / total_samples,
            "val_ratio_actual": len(val_df) / total_samples,
            "test_ratio_actual": len(test_df) / total_samples,
            "train_class_balance": train_df["binary_label"].value_counts().to_dict(),
            "val_class_balance": val_df["binary_label"].value_counts().to_dict(),
            "test_class_balance": test_df["binary_label"].value_counts().to_dict(),
        }

        return train_df, val_df, test_df, split_stats
