"""FastAPI application server for Evidence-Based Admissions AI Detection."""
import os
import sys
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.inference.detector import AdmissionsAIDetector


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw text of the college admissions essay")


app = FastAPI(
    title="EssayChecker",
    description="Evidence-Based AI Authorship Analysis for College Admissions Essays without LLM judges.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize production detector
detector = AdmissionsAIDetector()


SAMPLE_ESSAYS = [
    {
        "id": "sample_human_dumplings",
        "title": "Authentic Student Essay (Cultural Heritage)",
        "label": "Human Authorship",
        "badge": "Ground Truth: Human",
        "badge_color": "emerald",
        "description": "Authentic personal narrative featuring concrete sensory details, tactile apprentice memories, and active first-person agency.",
        "text": """Every Saturday morning, our kitchen transformed into a bustling dim sum factory. Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day. My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings. At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop. Nai Nai never scolded me; she merely pressed another circle of dough into my palm, her rough calloused thumbs guiding mine with silent patience.

Through those dumplings, I learned the quiet language of my heritage. In a household where 'I love you' was rarely spoken aloud, affection was measured in steaming bamboo baskets and bowls of slow-simmered winter melon soup. When my family moved to suburban Ohio in fifth grade, that culinary dialect became my anchor. While classmates brought Lunchables, I unpacked containers of fragrant scallion pancakes, learning to embrace the curious glances rather than shrink from them.

As I grew older, this kitchen apprenticeship evolved into a broader curiosity about food anthropology and cultural preservation. In high school, I founded the Cultural Heritage Exchange, organizing community dinners where students from immigrant backgrounds shared their families' traditional dishes along with the stories behind them. Standing before twenty peers, teaching them Nai Nai’s precise three-fold pleating technique, I realized that food is more than sustenance—it is living history. At university, I hope to continue bridging cultural divides, combining sociology and culinary traditions to ensure immigrant narratives are preserved and celebrated."""
    },
    {
        "id": "sample_ai_gpt4o",
        "title": "AI-Generated Essay (GPT-4o)",
        "label": "Machine Authorship",
        "badge": "Ground Truth: AI (GPT-4o)",
        "badge_color": "coral",
        "description": "Synthetic generation prompted on Common App Prompt 1. Exhibits elevated abstract buzzwords, smoothed sentence variance, and formulaic moral wrapping.",
        "text": """Growing up at the intersection of two distinct cultures, my identity was forged through the vibrant tapestry of traditions that adorned our family home. Every Sunday afternoon, our living room resonated with the harmonious melodies of traditional folk songs juxtaposed against the rhythmic cadence of contemporary American music. This multifaceted environment served as a powerful catalyst for my personal evolution, fostering a deep appreciation for the profound interconnectedness of diverse human experiences.

Throughout my formative years, I often found myself navigating the complex dichotomy between preserving my cultural heritage and assimilating into my suburban community. Rather than viewing this duality as an insurmountable obstacle, I embraced it as a quintessential opportunity for intellectual and emotional enrichment. In high school, I sought to bridge these disparate worlds by organizing multicultural symposiums that celebrated diversity and fostered dialogue among students from myriad backgrounds.

Ultimately, this journey has illuminated the pivotal importance of empathy and cultural diplomacy in an increasingly globalized world. As I embark on my collegiate journey, I am eager to delve deeper into international relations and sociology, utilizing my unique perspective to champion inclusivity, cultivate meaningful cross-cultural connections, and contribute positively to the academic community."""
    },
    {
        "id": "sample_ai_polished",
        "title": "Synthetic AI-Polished Student Draft",
        "label": "AI-Polished / Assisted",
        "badge": "Ground Truth: AI-Polished",
        "badge_color": "purple",
        "description": "Human student draft subsequently polished by an LLM for grammar and stylistic flow. Shows hybrid characteristics with elevated abstract syntax.",
        "text": """Every Saturday morning, our kitchen transformed into a vibrant dumpling-making haven. Flour dusted the floor tiles like a soft blanket of snow, while the rhythmic tempo of my grandmother's cleaver established the day's cadence. My responsibility was clear: master the delicate art of folding siu mai dumplings. At seven years old, my inexperienced hands frequently tore the thin dough wrappers, scattering seasoned pork across the surface. Yet Nai Nai never reprimanded me; instead, she gently guided my fingers with enduring patience.

Through this cherished tradition, I absorbed the profound cultural essence of my heritage. In our household, love was not expressed through overt verbal declarations, but through steaming bamboo baskets and comforting bowls of winter melon soup. When our family relocated to Ohio, this culinary connection became my stabilizing foundation, allowing me to embrace my cultural identity with confidence.

As time progressed, this kitchen experience inspired me to establish the Cultural Heritage Exchange in high school, where students gathered to share traditional family recipes and cultural histories. Guiding my peers through Nai Nai's dumpling-folding technique illuminated the transformative power of culinary storytelling in bridging social divides. In college, I look forward to integrating sociology and cultural studies to preserve and celebrate diverse immigrant traditions."""
    },
    {
        "id": "sample_false_positive",
        "title": "Known Failure Case: Complex Human Essay",
        "label": "False Positive Candidate",
        "badge": "Diagnosed False Positive",
        "badge_color": "amber",
        "description": "Authentic human essay on cello composition that triggers AI flags due to elevated musical vocabulary, complex phrasing, and low punctuation variance.",
        "text": """The cello endpin vibrating against the polished spruce stage floor is the only physical constant in my life. I have moved across four states and attended three high schools, but whenever I unpack my instrument, I am instantly home. In my sophomore year, I began composing an orchestral suite inspired by the migration patterns of monarch butterflies—a metaphor for the dislocation and resilience of itinerant families like mine.

Composing forced me to confront the vulnerability of original creation. Unlike performing Bach or Elgar where the interpretive boundaries are established, a blank score demands complete ownership of every harmonic dissonance and resolution. When our youth orchestra premiered the third movement, listening to sixty musicians breathe life into melodies that had previously existed only inside my head was transformative.

Music composition taught me how to structure complex narratives, balance individual voices within a collective ensemble, and communicate across emotional barriers. At the university, I seek an environment where I can double major in music composition and cognitive science, investigating how acoustic harmony influences neurological empathy and emotional processing."""
    }
]


@app.post("/api/analyze")
async def analyze_essay(req: AnalyzeRequest) -> Dict[str, Any]:
    """Runs complete evidence-based authorship analysis."""
    result = detector.analyze(req.text)
    return result


@app.get("/api/samples")
async def get_sample_essays() -> Dict[str, Any]:
    """Returns curated showcase essays for interactive demonstration."""
    return {"samples": SAMPLE_ESSAYS}


@app.get("/api/reference-stats")
async def get_reference_stats() -> Dict[str, Any]:
    """Returns empirical human reference statistics and model thresholds."""
    return {
        "model_version": detector.artifact.model_version,
        "feature_count": len(detector.artifact.feature_names),
        "thresholds": detector.artifact.thresholds,
        "reference_stats": detector.artifact.human_reference_stats,
    }


# Mount static UI directory
ui_dir = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(ui_dir):
    app.mount("/static", StaticFiles(directory=ui_dir), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(ui_dir, "index.html"))
