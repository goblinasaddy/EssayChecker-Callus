"""Integration tests for FastAPI application server endpoints and Vercel entrypoint."""
import pytest
from fastapi.testclient import TestClient
from src.api.server import app
from api.index import app as vercel_app


client = TestClient(app)
vercel_client = TestClient(vercel_app)


def test_api_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 4
    
    # Check that showcase includes human, ai, polished, and false positive
    sample_ids = [s["id"] for s in data["samples"]]
    assert "sample_human_dumplings" in sample_ids
    assert "sample_ai_gpt4o" in sample_ids
    assert "sample_ai_polished" in sample_ids
    assert "sample_false_positive" in sample_ids


def test_api_get_reference_stats():
    response = client.get("/api/reference-stats")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "2.0.0"
    assert "thresholds" in data
    assert "reference_stats" in data


def test_api_post_analyze_endpoint():
    sample_payload = {
        "text": """Every Saturday morning, our kitchen transformed into a bustling dim sum factory.
Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day.
My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings.
At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop.
Through those dumplings, I learned the quiet language of my heritage."""
    }

    response = client.post("/api/analyze", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "SUCCESS"
    assert "assessment" in data
    assert "highlighted_spans" in data
    assert "evidence" in data
    assert "disclaimers" in data


def test_vercel_entrypoint_and_static_serving():
    # Test root index
    res_index = vercel_client.get("/")
    assert res_index.status_code == 200
    assert "EssayChecker" in res_index.text

    # Test static assets
    res_css = vercel_client.get("/static/style.css")
    assert res_css.status_code == 200

    res_js = vercel_client.get("/static/app.js")
    assert res_js.status_code == 200

    # Test API through Vercel entrypoint
    res_api = vercel_client.get("/api/reference-stats")
    assert res_api.status_code == 200
    assert "thresholds" in res_api.json()
