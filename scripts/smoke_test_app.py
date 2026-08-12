"""End-to-end smoke test verifying application endpoints and showcase essays."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.api.server import app, SAMPLE_ESSAYS


def main():
    print("=" * 60)
    print("VERITASEASY PHASE 2 SMOKE TEST & SHOWCASE EVALUATION")
    print("=" * 60)

    client = TestClient(app)

    # 1. Test Static Index
    res_index = client.get("/")
    assert res_index.status_code == 200
    print("[1/4] Frontend Web UI served successfully at /")

    # 2. Test Reference Stats
    res_stats = client.get("/api/reference-stats")
    assert res_stats.status_code == 200
    stats_data = res_stats.json()
    print(f"[2/4] Human Reference Stats API returned {stats_data['feature_count']} features.")

    # 3. Test All 4 Showcase Essays
    print("\n[3/4] Evaluating Showcase Admissions Essays:")
    for sample in SAMPLE_ESSAYS:
        res = client.post("/api/analyze", json={"text": sample["text"]})
        assert res.status_code == 200
        data = res.json()
        
        assessment = data["assessment"]
        meta = data["metadata"]
        print(f"\n      --- {sample['title']} ---")
        print(f"      Label Badge:       {sample['badge']}")
        print(f"      Decision:          {assessment['category']} (Code: {assessment['verdict_code']})")
        print(f"      Calibrated AI P:   {assessment['calibrated_ai_probability'] * 100:.1f}%")
        print(f"      Confidence:        {assessment['confidence_level']}")
        print(f"      Flagged Sentences: {meta['flagged_sentence_count']} / {meta['sentence_count']}")
        
        # Verify span grounding integrity
        for span in data["highlighted_spans"]:
            reconstructed = sample["text"][span["start_char"]:span["end_char"]]
            assert reconstructed == span["text"], f"Span mismatch: '{reconstructed}' vs '{span['text']}'"

    print("\n[4/4] Verified 100% exact character span fidelity across all sample essays.")
    print("=" * 60)


if __name__ == "__main__":
    main()
