"""Debug script to inspect local and global evidence on India Gate and other essays."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.detector import AdmissionsAIDetector
from src.features.discourse import DiscourseFeatureExtractor

def main():
    detector = AdmissionsAIDetector()
    india_gate_essay = (
        "India Gate, located in the heart of New Delhi, stands as an imposing monumental tribute "
        "to the brave soldiers who sacrificed their lives during the First World War. "
        "This majestic archway serves as a poignant reminder of valor, resilience, and historical significance. "
        "The architectural grandeur of the monument, coupled with the eternal flame of the Amar Jawan Jyoti, "
        "evokes a profound sense of patriotism and collective memory among visitors from around the world. "
        "Throughout the decades, this iconic landmark has continued to inspire generations, "
        "symbolizing the enduring spirit of national unity and cultural heritage in an ever-evolving global society."
    )

    print("Word count:", len(india_gate_essay.split()))
    res = detector.analyze(india_gate_essay)
    print("Assessment:", res["assessment"])
    print("Flagged sentence count:", res["metadata"]["flagged_sentence_count"])
    
    print("\n--- Global Feature Contributions ---")
    for feat in res["evidence"]["top_ai_evidence"]:
        print(f"  AI Driver: {feat['feature']:<35} observed={feat['observed_value']} impact={feat['impact_score']}")
    for feat in res["evidence"]["top_human_evidence"]:
        print(f"  Human Driver: {feat['feature']:<35} observed={feat['observed_value']} impact={feat['impact_score']}")

    print("\n--- Sentences & Evidence ---")
    for span in res["highlighted_spans"]:
        print(f"\nSentence #{span['sentence_idx'] + 1} ({span['overall_severity'].upper()}): \"{span['text']}\"")
        for it in span["evidence_items"]:
            print(f"   -> [{it['direction']}] {it['feature_display_name']}: observed={it['observed_value']} (ref={it['reference_mean']}) z={it['deviation_z']} strength={it['evidence_strength']}")
            print(f"      {it['explanation']}")

if __name__ == "__main__":
    main()
