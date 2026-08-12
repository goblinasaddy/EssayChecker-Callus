"""Debug script to verify human essay produces clean human/neutral evidence."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.detector import AdmissionsAIDetector

def main():
    detector = AdmissionsAIDetector()
    human_essay = """Every Saturday morning, our kitchen transformed into a bustling dim sum factory. Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day. My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings. At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop. Nai Nai never scolded me; she merely pressed another circle of dough into my palm, her rough calloused thumbs guiding mine with silent patience.

Through those dumplings, I learned the quiet language of my heritage. In a household where 'I love you' was rarely spoken aloud, affection was measured in steaming bamboo baskets and bowls of slow-simmered winter melon soup. When my family moved to suburban Ohio in fifth grade, that culinary dialect became my anchor. While classmates brought Lunchables, I unpacked containers of fragrant scallion pancakes, learning to embrace the curious glances rather than shrink from them.

As I grew older, this kitchen apprenticeship evolved into a broader curiosity about food anthropology and cultural preservation. In high school, I founded the Cultural Heritage Exchange, organizing community dinners where students from immigrant backgrounds shared their families' traditional dishes along with the stories behind them. Standing before twenty peers, teaching them Nai Nai’s precise three-fold pleating technique, I realized that food is more than sustenance—it is living history. At university, I hope to continue bridging cultural divides, combining sociology and culinary traditions to ensure immigrant narratives are preserved and celebrated."""

    res = detector.analyze(human_essay)
    print("Word count:", len(human_essay.split()))
    print("Assessment:", res["assessment"])
    print("Flagged count:", res["metadata"]["flagged_sentence_count"])
    
    high_count = sum(1 for s in res["highlighted_spans"] if s["overall_severity"] == "high")
    med_count = sum(1 for s in res["highlighted_spans"] if s["overall_severity"] == "medium")
    human_count = sum(1 for s in res["highlighted_spans"] if s["overall_severity"] == "human_grounded")
    neutral_count = sum(1 for s in res["highlighted_spans"] if s["overall_severity"] == "neutral")
    
    print(f"Severity breakdown: High (Red)={high_count}, Med (Amber)={med_count}, Human Grounded={human_count}, Neutral={neutral_count}")

if __name__ == "__main__":
    main()
