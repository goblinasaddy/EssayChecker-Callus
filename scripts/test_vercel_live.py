"""Live HTTP verification script testing Vercel dev server endpoints."""
import urllib.request
import json
import sys

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    base_url = f"http://localhost:{port}"
    print(f"Testing live Vercel dev server at {base_url}...")

    # 1. Test Index
    req = urllib.request.urlopen(f"{base_url}/")
    assert req.status == 200
    html_content = req.read().decode("utf-8")
    assert "EssayChecker" in html_content
    print("[1/6] GET / -> HTTP 200 OK (HTML UI served)")

    # 2. Test Static CSS
    req = urllib.request.urlopen(f"{base_url}/static/style.css")
    assert req.status == 200
    print("[2/6] GET /static/style.css -> HTTP 200 OK (CSS static asset)")

    # 3. Test Static JS
    req = urllib.request.urlopen(f"{base_url}/static/app.js")
    assert req.status == 200
    print("[3/6] GET /static/app.js -> HTTP 200 OK (JS static asset)")

    # 4. Test Samples API
    req = urllib.request.urlopen(f"{base_url}/api/samples")
    assert req.status == 200
    samples_data = json.loads(req.read().decode("utf-8"))
    assert "samples" in samples_data
    count = len(samples_data["samples"])
    print(f"[4/6] GET /api/samples -> HTTP 200 OK ({count} samples returned)")

    # 5. Test Reference Stats API
    req = urllib.request.urlopen(f"{base_url}/api/reference-stats")
    assert req.status == 200
    stats_data = json.loads(req.read().decode("utf-8"))
    assert "thresholds" in stats_data
    version = stats_data["model_version"]
    print(f"[5/6] GET /api/reference-stats -> HTTP 200 OK (Model version {version})")

    # 6. Test Analyze API
    sample_text = samples_data["samples"][0]["text"]
    req_data = json.dumps({"text": sample_text}).encode("utf-8")
    post_req = urllib.request.Request(f"{base_url}/api/analyze", data=req_data, headers={"Content-Type": "application/json"})
    res = urllib.request.urlopen(post_req)
    assert res.status == 200
    analysis = json.loads(res.read().decode("utf-8"))
    assert analysis["status"] == "SUCCESS"
    category = analysis["assessment"]["category"]
    prob = analysis["assessment"]["calibrated_ai_probability"] * 100
    print(f"[6/6] POST /api/analyze -> HTTP 200 OK (Assessment: {category}, P={prob:.1f}%)")

    print("\nSUCCESS: ALL VERCEL DEV LIVE ENDPOINTS VERIFIED AND PASSING!")

if __name__ == "__main__":
    main()
