"""Executes robustness slicing, ESL sensitivity audit, and candidate failure analysis."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np

from src.models.baselines import InterpretableBaselineModels
from src.evaluation.robustness import RobustnessAnalyzer
from src.evaluation.failure_analyzer import FailureAnalyzer


def main():
    print("=" * 60)
    print("PHASE 1 ROBUSTNESS, ESL SENSITIVITY & FAILURE ANALYSIS")
    print("=" * 60)

    data_dir = os.path.join("data", "processed")
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))

    feature_cols = [c for c in train_df.columns if c.startswith(("surface_", "discourse_", "dist_"))]
    y_train = train_df["binary_label"].to_numpy(dtype=int)

    model = InterpretableBaselineModels(random_state=42)
    model.fit(train_df, y_train, feature_cols)

    robust_analyzer = RobustnessAnalyzer()
    
    # 1. Length Robustness
    length_df = robust_analyzer.evaluate_length_robustness(model, test_df)
    print("\n[1/4] Subgroup Analysis: Essay Length Slices")
    print(length_df[["length_slice", "sample_count", "lr_accuracy", "lr_f1", "lr_fpr"]].to_string(index=False))

    # 2. Topic Robustness
    topic_df = robust_analyzer.evaluate_topic_robustness(model, test_df)
    print("\n[2/4] Subgroup Analysis: Admissions Topic Categories")
    print(topic_df[["topic", "sample_count", "lr_accuracy", "lr_f1", "lr_fpr"]].to_string(index=False))

    # 3. Model & AI Polishing Slices
    polished_df = robust_analyzer.evaluate_polished_vs_pure_ai(model, test_df)
    print("\n[3/4] Subgroup Analysis: Human vs Pure AI vs Synthetic AI-Polished")
    print(polished_df.to_string(index=False))

    # 4. ESL Sensitivity Audit
    esl_audit = robust_analyzer.evaluate_esl_sensitivity(model, test_df)
    print(f"\n[ESL Sensitivity Audit]: {esl_audit}")

    # 5. Candidate Failure Case Analysis
    failure_analyzer = FailureAnalyzer(model, train_df)
    failures = failure_analyzer.find_failure_cases(test_df, top_k=3)
    
    print("\n[4/4] Candidate Failure Analysis (False Positives & Negatives):")
    print(f"      False Positives Found: {len(failures['false_positives'])}")
    for fp in failures["false_positives"]:
        print(f"      - [FP] ID: {fp['essay_id']} | AI Prob: {fp['ai_probability']:.3f} | Topic: {fp['topic']}")
        print(f"        Snippet: {fp['text_snippet']}")
        print(f"        Top AI-Skewed Features: {[a['feature'] for a in fp['top_ai_skewed_features']]}")

    print(f"\n      False Negatives Found: {len(failures['false_negatives'])}")
    for fn in failures["false_negatives"]:
        print(f"      - [FN] ID: {fn['essay_id']} | AI Prob: {fn['ai_probability']:.3f} | Topic: {fn['topic']}")
        print(f"        Snippet: {fn['text_snippet']}")
        print(f"        Top Human-Skewed Features: {[a['feature'] for a in fn['top_human_skewed_features']]}")

    # Export full payload
    payload = {
        "length_robustness": length_df.to_dict(orient="records"),
        "topic_robustness": topic_df.to_dict(orient="records"),
        "label_category_robustness": polished_df.to_dict(orient="records"),
        "esl_sensitivity_audit": esl_audit,
        "candidate_failures": failures,
    }

    out_json = os.path.join(data_dir, "failure_and_robustness_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nFailure & robustness report exported to {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
