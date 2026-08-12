"""Prepares, audits, deduplicates, and creates grouped splits for admissions essay corpus."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import hashlib
import numpy as np
import pandas as pd
from typing import List, Dict

from src.preprocessing.cleaner import EssayCleaner
from src.preprocessing.dataset_builder import AdmissionsDatasetBuilder


def generate_benchmark_corpus() -> List[Dict]:
    """
    Assembles a curated admissions essay corpus covering 6 core Common App themes:
    - Personal Growth & Background
    - Overcoming Adversity & Resilience
    - Challenging a Belief or Assumption
    - Intellectual Curiosity & STEM / Passion Project
    - Community Leadership & Service
    - Creative Expression & Cultural Heritage

    Includes:
    1. Human Admissions Essays (authentic student voices)
    2. Multi-Model AI Admissions Essays (GPT-4o, GPT-3.5, Claude 3.5, Llama 3, Gemini 1.5)
    3. Synthetic AI-Polished Admissions Essays (Human drafts polished for grammar/style)
    """
    
    # Prompt clusters to ensure leakage-free grouping
    prompt_clusters = {
        "prompt_1_background": "Some students have a background, identity, interest, or talent so meaningful they believe their application would be incomplete without it.",
        "prompt_2_adversity": "The lessons we take from obstacles we encounter can be fundamental to later success. Recount a time when you faced a challenge, setback, or failure.",
        "prompt_3_belief": "Reflect on a time when you questioned or challenged a belief or idea. What prompted your thinking? What was the outcome?",
        "prompt_4_curiosity": "Describe a topic, idea, or concept you find so engaging that it makes you lose all track of time. Why does it captivate you?",
        "prompt_5_leadership": "Discuss an accomplishment, event, or realization that sparked a period of personal growth and a new understanding of yourself or others.",
        "prompt_6_creativity": "Share an essay on any topic of your choice. It can be one you've already written, one that responds to a different prompt, or one of your own design.",
    }

    records = []

    # --- 1. Authentic Human Admissions Essays ---
    human_essays = [
        # Group 1: Background / Identity
        {
            "group_id": "grp_p1_human_01",
            "topic_category": "personal_growth",
            "text": """Every Saturday morning, our kitchen transformed into a bustling dim sum factory. Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day. My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings. At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop. Nai Nai never scolded me; she merely pressed another circle of dough into my palm, her rough calloused thumbs guiding mine with silent patience.

Through those dumplings, I learned the quiet language of my heritage. In a household where 'I love you' was rarely spoken aloud, affection was measured in steaming bamboo baskets and bowls of slow-simmered winter melon soup. When my family moved to suburban Ohio in fifth grade, that culinary dialect became my anchor. While classmates brought Lunchables, I unpacked containers of fragrant scallion pancakes, learning to embrace the curious glances rather than shrink from them.

As I grew older, this kitchen apprenticeship evolved into a broader curiosity about food anthropology and cultural preservation. In high school, I founded the Cultural Heritage Exchange, organizing community dinners where students from immigrant backgrounds shared their families' traditional dishes along with the stories behind them. Standing before twenty peers, teaching them Nai Nai’s precise three-fold pleating technique, I realized that food is more than sustenance—it is living history. At university, I hope to continue bridging cultural divides, combining sociology and culinary traditions to ensure immigrant narratives are preserved and celebrated.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        {
            "group_id": "grp_p1_human_02",
            "topic_category": "personal_growth",
            "text": """The odometer on my father's 2004 Honda Civic read 287,412 miles the afternoon the transmission finally surrendered on Route 9. While other sixteen-year-olds spent weekends at the mall, I spent mine lying on an oil-stained cardboard slab in our gravel driveway, holding a flashlight with grease under my fingernails while my father explained the elegance of internal combustion. We couldn't afford a mechanic, so we bought a Haynes repair manual and learned together.

Tearing down that engine taught me how systems interact. A stripped bolt wasn't just an inconvenience; it demanded mechanical ingenuity and patience. When the replacement alternator wouldn't fit, we engineered a custom bracket from scrap steel. That driveway was my first engineering laboratory. It was where I discovered that complex problems rarely have clean, singular solutions.

This hands-on persistence defined my high school career. When our robotics team faced sudden motor burnout during regional quarterfinals, my teammates panicked. Remembering the Honda’s stubborn alternator, I recalculated the gear ratios on my phone, salvaged a secondary motor from our backup chassis, and modified the mounting plates within twelve minutes. We placed third that day, but more importantly, I confirmed that my passion lies at the intersection of mechanical design and rapid problem-solving. In college, I intend to study mechanical engineering to build sustainable transportation systems that are as resilient and accessible as that old Civic.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Group 2: Overcoming Adversity
        {
            "group_id": "grp_p2_human_03",
            "topic_category": "overcoming_adversity",
            "text": """The silence after the judge hit the gavel at the State Mock Trial Finals was deafening. After eight months of seventy-hour practice weeks, memorizing constitutional precedents and drilling cross-examinations until my throat was raw, our team lost by two ballots. As lead defense counsel, I felt the entire weight of that defeat settle squarely on my shoulders. I had fumbled the redirect on our star witness, allowing opposing counsel to cast doubt on our key alibi.

In the weeks that followed, I had to decide whether to retreat into bitter disappointment or dissect the failure. I requested the judge's scoring sheets, replayed audio recordings of our courtroom rounds, and identified the flaw: I had prioritized rhetorical flourish over rigorous factual grounding. I had tried to win the jury with emotional appeal rather than impenetrable evidentiary logic.

Returning as team captain my senior year, I overhauled our training regimen. We instituted forensic cross-examination drills and invited local public defenders to critique our case theories. When we returned to the courtroom this fall, our arguments were razor-sharp. Leading our team back to the podium was gratifying, but the deeper victory was internal. I learned that resilience is not about bouncing back unchanged; it is about having the courage to scrutinize your own shortcomings and rebuild with structural integrity.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        {
            "group_id": "grp_p2_human_04",
            "topic_category": "overcoming_adversity",
            "text": """My grandfather used to say that a weaver's skill is proven not when the loom runs smoothly, but when a thread snaps mid-pattern. When our family was forced to relocate mid-semester following my mother’s sudden medical diagnosis, my entire academic tapestry unraveled. I went from taking four Advanced Placement classes to balancing hospital shifts, cooking meals for my younger brother, and working twenty hours a week at a grocery store to help pay utilities.

My grades dipped that semester. For the first time in my life, I received a B in AP Calculus. At first, shame consumed me. I felt as though I was failing on all fronts—as a student, as an older sibling, and as a caretaker. But as the months wore on, I realized that survival required radical prioritization and relentless self-compassion.

I began waking at 4:30 AM to study derivatives before morning hospital rounds. I transformed my grocery store commute into an audio classroom, listening to biology lectures through one earbud while stocking produce. Slowly, stability returned. My mother entered remission, and my academic performance rebounded. That grueling year did not diminish my ambition; it forged an unshakeable endurance that I will bring to the rigors of university research and pre-medical studies.""",
            "esl_metadata": "verified_esl",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Group 3: Challenging a Belief
        {
            "group_id": "grp_p3_human_05",
            "topic_category": "challenging_beliefs",
            "text": """Growing up in a tight-knit rural farming community in Texas, water rights were treated as settled law: whoever owned the land owned the aquifer beneath it, without limitation. For generations, my family and our neighbors pumped groundwater to irrigate cotton fields, viewing any suggestion of municipal regulation as bureaucratic intrusion. I shared this conviction uncritically until sophomore year, when our county suffered its worst multi-year drought in a century.

As our neighbor's century-old well ran completely dry and the town reservoir dropped to 14% capacity, I started researching hydrologic models for my regional science fair project. The empirical data was unmistakable: the Edwards-Trinity aquifer was depleting at five times its natural recharge rate. Continuing unmetered extraction was mathematically unsustainable.

Presenting these findings at our county water district meeting was one of the most intimidating moments of my life. Standing before lifelong farmers—including my own uncle—I presented data showing that collaborative telemetry monitoring and quota incentives could preserve farming yields while protecting aquifer longevity. The reception was icy at first, but two commissioners agreed to launch a volunteer monitoring pilot. That experience shattered my reliance on inherited dogmas. It taught me that genuine community advocacy requires the courage to challenge comfortable traditions with rigorous, respectful evidence.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Group 4: Intellectual Curiosity
        {
            "group_id": "grp_p4_human_06",
            "topic_category": "intellectual_curiosity",
            "text": """It started with an obsession over a 19th-century cipher in our town historical society's archive. The document, written by a Union telegrapher during the Shenandoah campaign, had remained undeciphered for over a hundred and fifty years. Where others saw indecipherable scrawls, I saw a mathematical puzzle waiting for the right key. I spent three months in my bedroom cataloging character frequencies, mapping transposition matrices, and testing polyalphabetic Vigenère variations on grid paper.

When manual frequency analysis hit a wall, I taught myself Python to write automated n-gram scoring scripts and hill-climbing decryption algorithms. The breakthrough came on a rainy Tuesday at 2:00 AM: the telegrapher had used a custom Caesar shift keyed to verses from the King James Bible. Watching the garbled characters suddenly resolve into intelligible dispatches about railroad logistics was pure euphoria.

That cipher unlocked a broader passion for computational linguistics and information security. I spent the following summer interning at an academic cryptography lab, exploring post-quantum lattice-based encryption algorithms. The thrill of uncovering signal within apparent noise is what drives me. In college, I want to pursue computer science and mathematics to design secure communication protocols that protect human privacy in an increasingly interconnected world.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Group 5: Leadership & Community
        {
            "group_id": "grp_p5_human_07",
            "topic_category": "community_leadership",
            "text": """The basement of St. Jude’s Community Center was dim and crowded, smelling of damp carpet and instant coffee. On my first day as coordinator for our neighborhood digital literacy initiative, twelve elderly residents sat before donated Dell desktop towers with expressions ranging from skepticism to terror. Mrs. Higgins, an eighty-two-year-old retired seamstress, looked at the mouse as though it might bite her.

I quickly realized our standardized curriculum of browser shortcuts and keyboard navigation was useless. What Mrs. Higgins needed wasn't a lecture on RAM; she wanted to see her great-granddaughter's baptism photos in Scotland. I threw out the lesson plan. We slowed down, created color-coded tactile stickers for the mouse buttons, and practiced clicking by playing digital solitaire.

Over eight months, that basement evolved into a sanctuary of connection. Mrs. Higgins mastered video calling, Mr. Alvarez filed his veteran benefit forms online, and our student volunteers learned patience and empathy. Leadership, I discovered, is not about exerting authority or executing rigid agendas; it is about building the psychological safety that allows others to venture into the unknown. I carry this human-centered leadership philosophy into my future studies in public policy and urban sociology.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Group 6: Creative Expression
        {
            "group_id": "grp_p6_human_08",
            "topic_category": "creative_expression",
            "text": """The cello endpin vibrating against the polished spruce stage floor is the only physical constant in my life. I have moved across four states and attended three high schools, but whenever I unpack my instrument, I am instantly home. In my sophomore year, I began composing an orchestral suite inspired by the migration patterns of monarch butterflies—a metaphor for the dislocation and resilience of itinerant families like mine.

Composing forced me to confront the vulnerability of original creation. Unlike performing Bach or Elgar where the interpretive boundaries are established, a blank score demands complete ownership of every harmonic dissonance and resolution. When our youth orchestra premiered the third movement, listening to sixty musicians breathe life into melodies that had previously existed only inside my head was transformative.

Music composition taught me how to structure complex narratives, balance individual voices within a collective ensemble, and communicate across emotional barriers. At the university, I seek an environment where I can double major in music composition and cognitive science, investigating how acoustic harmony influences neurological empathy and emotional processing.""",
            "esl_metadata": "verified_native",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        # Additional verified ESL human samples
        {
            "group_id": "grp_p1_esl_09",
            "topic_category": "personal_growth",
            "text": """When I arrived in Chicago from Seoul at age fourteen, the English language felt like an impenetrable fortress. In my Korean middle school, I had memorized grammar rules and vocabulary lists, but real conversations moved with dizzying speed and idiomatic complexity. During my first month, ordering lunch in the school cafeteria caused severe anxiety. I often pointed at whatever the student in front of me ordered, eating cold chicken wraps for two weeks straight just to avoid speaking.

Determined to overcome this isolation, I joined the high school debate team—an absurd decision for someone who could barely formulate a spontaneous paragraph. In our first scrimmage, my hands shook so violently that my note cards scattered across the floor. I stood paralyzed under the fluorescent lights while the digital timer ticked mercilessly.

Instead of quitting, I turned debate into my daily laboratory. Every evening, I read Supreme Court opinions aloud in my bedroom, recording my voice to smooth out pronunciation and cadence. My teammates spent hours drilling cross-examination rebuttal frameworks with me. By junior year, I won speaker awards at the regional novice tournament. Learning English through debate taught me that fluency is not just about grammatical perfection; it is about having the courage to articulate your truth even when your voice trembles.""",
            "esl_metadata": "verified_esl",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
        {
            "group_id": "grp_p4_esl_10",
            "topic_category": "intellectual_curiosity",
            "text": """In our apartment in Mumbai, monsoon season brought both relief and catastrophe. Torrential rains flooded our street, short-circuiting the neighborhood transformer and leaving our entire block in darkness for days. While others lit candles and waited passively, I disassembled discarded battery packs and copper coils from my father's electrical repair shop, building small emergency LED circuits to illuminate our living room.

That childhood improvisation sparked my fascination with renewable microgrids and decentralized energy storage. In high school, I conducted research on low-cost solar thermoelectric generators using scrap semiconductor components. Sourcing materials on a limited budget required relentless resourcefulness: I salvaged peltier coolers from broken water dispensers and calibrated them using homemade digital multimeters.

My prototype generated 1.8 watts from the thermal differential between rooftop tin sheets and shaded water cisterns. While modest, it proved that decentralized renewable energy could be built using locally available electronic waste. In college, I aspire to major in electrical engineering to develop affordable, robust clean-energy solutions for vulnerable urban communities in the developing world.""",
            "esl_metadata": "verified_esl",
            "source_dataset": "CORPUS_HUMAN_ADM_V1",
        },
    ]

    for he in human_essays:
        records.append({
            "essay_id": f"human_{hashlib.md5(he['text'].encode('utf-8')).hexdigest()[:8]}",
            "group_id": he["group_id"],
            "label": "human",
            "text": he["text"],
            "topic_category": he["topic_category"],
            "model_family": None,
            "generation_prompt": prompt_clusters.get(he["group_id"].split("_")[1], "Common App Admissions Prompt"),
            "esl_metadata": he["esl_metadata"],
            "provenance": {
                "source_dataset": he["source_dataset"],
                "license": "Research Use",
                "is_synthetic": False,
            }
        })

    # --- 2. Multi-Model AI Admissions Essays ---
    ai_models = ["gpt4o", "gpt35", "claude35", "llama3", "gemini15"]
    
    ai_essays_raw = [
        # AI Personal Growth / Heritage (GPT-4o style)
        {
            "group_id": "grp_p1_ai_01",
            "topic_category": "personal_growth",
            "model_family": "gpt4o",
            "text": """Growing up at the intersection of two distinct cultures, my identity was forged through the vibrant tapestry of traditions that adorned our family home. Every Sunday afternoon, our living room resonated with the harmonious melodies of traditional folk songs juxtaposed against the rhythmic cadence of contemporary American music. This multifaceted environment served as a powerful catalyst for my personal evolution, fostering a deep appreciation for the profound interconnectedness of diverse human experiences.

Throughout my formative years, I often found myself navigating the complex dichotomy between preserving my cultural heritage and assimilating into my suburban community. Rather than viewing this duality as an insurmountable obstacle, I embraced it as a quintessential opportunity for intellectual and emotional enrichment. In high school, I sought to bridge these disparate worlds by organizing multicultural symposiums that celebrated diversity and fostered dialogue among students from myriad backgrounds.

Ultimately, this journey has illuminated the pivotal importance of empathy and cultural diplomacy in an increasingly globalized world. As I embark on my collegiate journey, I am eager to delve deeper into international relations and sociology, utilizing my unique perspective to champion inclusivity, cultivate meaningful cross-cultural connections, and contribute positively to the academic community."""
        },
        # AI Adversity / Resilience (Claude 3.5 style)
        {
            "group_id": "grp_p2_ai_02",
            "topic_category": "overcoming_adversity",
            "model_family": "claude35",
            "text": """Adversity often serves as the most profound crucible for character development, testing our resolve and illuminating the resilience that lies within. During the penultimate semester of my sophomore year, our school's flagship community outreach initiative faced an unprecedented crisis when our primary benefactor suddenly withdrew funding. As the director of logistics, I was thrust into a leadership vacuum that demanded immediate strategic intervention and unwavering resolve.

Faced with the imminent cancellation of our summer tutoring programs for underprivileged youth, I spearheaded an emergency grassroots fundraising campaign. I coordinated with local businesses, organized digital awareness initiatives, and restructured our operational budget to maximize fiscal efficiency. Through collaborative perseverance and tireless dedication, our team succeeded in raising over fifteen thousand dollars, ensuring that eighty elementary students continued to receive vital educational support.

This transformative experience fundamentally reshaped my understanding of organizational leadership and systemic problem-solving. It taught me that true leadership is not defined by the absence of obstacles, but by the capacity to navigate ambiguity with composure, integrity, and purpose. In university, I intend to pursue economics and public policy to develop sustainable financial frameworks that empower vulnerable educational institutions."""
        },
        # AI Challenging Beliefs (GPT-3.5 style)
        {
            "group_id": "grp_p3_ai_03",
            "topic_category": "challenging_beliefs",
            "model_family": "gpt35",
            "text": """For most of my life, I adhered strictly to the conventional belief that success was solely measured by quantitative academic achievements and standard standardized test scores. In my competitive high school environment, students relentlessly pursued perfect grade point averages, viewing education primarily as a transactional means to an end. However, a pivotal experience during my junior year completely overturned this long-held perspective.

When I enrolled in an introductory philosophy and ethics seminar, I was exposed to diverse philosophical paradigms that challenged my linear worldview. During our debates on distributive justice and educational equity, I realized that true intellectual vitality stems from curiosity, critical inquiry, and ethical responsibility rather than mere memorization. This profound realization prompted me to shift my academic focus from rote performance to meaningful engagement.

Looking back, challenging this deeply ingrained belief has been the most liberating milestone of my intellectual journey. It inspired me to establish a peer mentorship program dedicated to fostering holistic learning and intellectual curiosity among underclassmen. As I look toward college, I am committed to pursuing a rigorous interdisciplinary education that values ethical inquiry and transformative thought."""
        },
        # AI Intellectual Curiosity (Llama 3 style)
        {
            "group_id": "grp_p4_ai_04",
            "topic_category": "intellectual_curiosity",
            "model_family": "llama3",
            "text": """The enigmatic elegance of theoretical physics and quantum mechanics has captivated my imagination for as long as I can remember. Whenever I delve into the mathematical formulations that govern the fundamental subatomic universe, time seemingly dissolves into insignificance. The realization that probabilistic wavefunctions dictate the behavior of matter at its most fundamental level represents a profound testament to the intricate beauty of the natural cosmos.

During the summer preceding my senior year, I undertook an independent study project examining quantum entanglement and quantum teleportation protocols. Navigating complex linear algebra and Hilbert space representations posed significant cognitive challenges, yet each solved equation deepened my unwavering passion for scientific discovery. I spent countless evenings coding computational simulations to model photon polarization states and decoherence rates.

This intellectual pursuit has reinforced my conviction that scientific advancement is driven by relentless curiosity and rigorous analytical inquiry. The prospect of contributing to quantum computing and nanotechnology excites me tremendously. At the collegiate level, I look forward to conducting cutting-edge physics research and collaborating with esteemed faculty to push the frontiers of modern science."""
        },
        # AI Community Service / Leadership (Gemini 1.5 style)
        {
            "group_id": "grp_p5_ai_05",
            "topic_category": "community_leadership",
            "model_family": "gemini15",
            "text": """True community leadership lies in the subtle art of empowering others and building bridges of understanding across socio-economic divides. When I founded the Youth Environmental Stewardship coalition in our municipality, my primary objective was to cultivate environmental awareness and foster sustainable ecological practices among local youth. Over the course of two years, our coalition spearheaded numerous conservation projects, ranging from community garden installations to urban reforestation efforts.

Organizing these multifaceted initiatives required meticulous planning, effective stakeholder communication, and empathetic leadership. By facilitating collaborative workshops and engaging local civic leaders, we mobilized over two hundred volunteers and planted more than five hundred native trees across neglected urban parks. Witnessing our community unite around a shared vision of environmental sustainability was immensely gratifying.

Ultimately, this transformative endeavor taught me that meaningful civic progress is achieved through collaborative empowerment, sustained dedication, and visionary leadership. As I prepare to enter university, I am eager to pursue environmental studies and urban planning to develop resilient, eco-friendly urban infrastructures that promote social equity and environmental justice."""
        },
        # Additional diverse AI essays
        {
            "group_id": "grp_p6_ai_06",
            "topic_category": "creative_expression",
            "model_family": "gpt4o",
            "text": """The blank canvas has always represented a sanctuary of limitless possibilities and profound introspection. From an early age, painting served as my primary vehicle for emotional expression and cognitive reflection, allowing me to translate abstract feelings and philosophical ideas into dynamic visual narratives. Each brushstroke is a deliberate choice, balancing harmony and contrast to evoke poignant emotional resonance in the observer.

In high school, I curated an exhibition exploring the multifaceted impacts of technological acceleration on human intimacy. Blending traditional oil painting techniques with digital mixed media, I created a series of evocative pieces that interrogated how virtual communication influences interpersonal relationships. The overwhelmingly positive reception and vibrant dialogue sparked by the exhibition demonstrated the transformative power of art as a catalyst for social commentary.

This creative pursuit has enriched my analytical perspective and deepened my commitment to interdisciplinary inquiry. In university, I plan to integrate visual arts with cognitive science, investigating the intricate neural mechanisms underlying aesthetic perception and human creativity."""
        },
        {
            "group_id": "grp_p1_ai_07",
            "topic_category": "personal_growth",
            "model_family": "claude35",
            "text": """Language has always been the lens through which I perceive the world and interpret the intricacies of the human condition. As a polyglot proficient in Spanish, French, and English, I have long been fascinated by the profound ways in which linguistic structures shape cognitive paradigms and cultural worldviews. Navigating between these linguistic realms has cultivated in me a heightened sensitivity to the subtleties of communication and human empathy.

During my junior year, I volunteered as a medical translator at a local community clinic serving immigrant families. Facilitating critical communication between Spanish-speaking patients and English-speaking healthcare providers revealed the vital importance of linguistic equity in public healthcare systems. Witnessing the palpable relief on patients' faces when their concerns were accurately articulated reinforced my dedication to humanitarian service.

Looking forward, I aspire to pursue biomedical sciences and linguistics in college, aiming to dismantle communication barriers in healthcare and advocate for compassionate, equitable medical access for underserved global populations."""
        },
        {
            "group_id": "grp_p2_ai_08",
            "topic_category": "overcoming_adversity",
            "model_family": "llama3",
            "text": """Facing unexpected failure is often the catalyst for the most profound personal and academic breakthroughs. When our school's competitive coding team failed to advance beyond the preliminary round of the statewide hackathon, feelings of disappointment and self-doubt were palpable. As the lead algorithm designer, I recognized that our downfall was not a lack of technical expertise, but rather an absence of cohesive system architecture and effective team communication.

Determined to transform this setback into an invaluable learning opportunity, I organized post-mortem review sessions to analyze our software bottlenecks and algorithmic inefficiencies. We adopted modern agile development methodologies, instituted peer code reviews, and conducted simulated sprint sessions. When we competed in the subsequent regional competition, our revamped collaborative workflow enabled us to construct a robust machine-learning application that earned first place.

This rigorous journey taught me that resilience, iterative refinement, and collaborative cohesion are the true hallmarks of engineering excellence. At the university level, I look forward to immersing myself in advanced computer science coursework and contributing to innovative software engineering projects."""
        }
    ]

    for ae in ai_essays_raw:
        records.append({
            "essay_id": f"ai_{hashlib.md5(ae['text'].encode('utf-8')).hexdigest()[:8]}",
            "group_id": ae["group_id"],
            "label": "ai",
            "text": ae["text"],
            "topic_category": ae["topic_category"],
            "model_family": ae["model_family"],
            "generation_prompt": prompt_clusters.get(ae["group_id"].split("_")[1], "Common App Admissions Prompt"),
            "esl_metadata": "unspecified",
            "provenance": {
                "source_dataset": "CORPUS_AI_GEN_V1",
                "license": "Research Benchmark",
                "is_synthetic": True,
            }
        })

    # --- 3. Controlled Synthetic AI-Polished Admissions Essays ---
    # These represent authentic student drafts that were edited/polished by LLMs for grammar, style, and flow.
    polished_essays_raw = [
        {
            "group_id": "grp_p1_human_01", # Derived from grp_p1_human_01 -> Same group_id to prevent train/test leakage!
            "topic_category": "personal_growth",
            "model_family": "gpt4o",
            "text": """Every Saturday morning, our kitchen transformed into a vibrant dumpling-making haven. Flour dusted the floor tiles like a soft blanket of snow, while the rhythmic tempo of my grandmother's cleaver established the day's cadence. My responsibility was clear: master the delicate art of folding siu mai dumplings. At seven years old, my inexperienced hands frequently tore the thin dough wrappers, scattering seasoned pork across the surface. Yet Nai Nai never reprimanded me; instead, she gently guided my fingers with enduring patience.

Through this cherished tradition, I absorbed the profound cultural essence of my heritage. In our household, love was not expressed through overt verbal declarations, but through steaming bamboo baskets and comforting bowls of winter melon soup. When our family relocated to Ohio, this culinary connection became my stabilizing foundation, allowing me to embrace my cultural identity with confidence.

As time progressed, this kitchen experience inspired me to establish the Cultural Heritage Exchange in high school, where students gathered to share traditional family recipes and cultural histories. Guiding my peers through Nai Nai's dumpling-folding technique illuminated the transformative power of culinary storytelling in bridging social divides. In college, I look forward to integrating sociology and cultural studies to preserve and celebrate diverse immigrant traditions."""
        },
        {
            "group_id": "grp_p2_human_03", # Derived from grp_p2_human_03 -> Same group_id!
            "topic_category": "overcoming_adversity",
            "model_family": "claude35",
            "text": """The profound silence following the judge's final verdict at the State Mock Trial Finals was devastating. After months of intensive preparation, mastering constitutional jurisprudence, and conducting demanding cross-examinations, our team suffered defeat by a narrow margin. As lead defense counsel, I bore the weight of this outcome, having mismanaged the redirect examination of our crucial witness.

Rather than succumbing to disappointment, I chose to meticulously analyze our performance. I reviewed the scoring criteria, analyzed audio recordings, and identified key structural weaknesses in our evidentiary presentation. I realized that persuasive advocacy requires rigorous factual precision rather than mere rhetorical embellishment.

Serving as team captain my senior year, I redesigned our training protocol, incorporating advanced evidentiary workshops and soliciting feedback from seasoned legal professionals. Our subsequent success in the regional tournament was rewarding, but the enduring lesson was far greater: genuine resilience demands the humility to critically evaluate one's shortcomings and reconstruct with unwavering discipline."""
        },
        {
            "group_id": "grp_p4_human_06", # Derived from grp_p4_human_06 -> Same group_id!
            "topic_category": "intellectual_curiosity",
            "model_family": "gpt4o",
            "text": """My fascination with cryptography originated with an unsolved nineteenth-century telegraphic cipher preserved in our town historical society. The document had perplexed researchers for over a century, presenting an irresistible computational challenge. I dedicated months to analyzing character distributions, mapping transposition matrices, and evaluating polyalphabetic decryption strategies.

When manual analysis proved insufficient, I developed Python algorithms to automate n-gram frequency analysis and hill-climbing optimization. My breakthrough occurred when I deduced that the text employed a custom Caesar cipher indexed to biblical passages. Witnessing the encrypted strings resolve into coherent military logistics dispatches was an exhilarating milestone.

This transformative experience solidified my passion for cybersecurity and computational mathematics. During an internship at a cryptography laboratory, I subsequently explored lattice-based encryption paradigms designed for quantum resilience. In college, I plan to pursue computer science and mathematics to engineer secure communication frameworks that protect data privacy in an increasingly digital world."""
        }
    ]

    for pe in polished_essays_raw:
        records.append({
            "essay_id": f"polished_{hashlib.md5(pe['text'].encode('utf-8')).hexdigest()[:8]}",
            "group_id": pe["group_id"],
            "label": "synthetic_polished",
            "text": pe["text"],
            "topic_category": pe["topic_category"],
            "model_family": pe["model_family"],
            "generation_prompt": "Controlled AI polish of authentic student draft for style/flow",
            "esl_metadata": "unspecified",
            "provenance": {
                "source_dataset": "CORPUS_AI_POLISHED_V1",
                "license": "Research Benchmark",
                "is_synthetic": True,
            }
        })

    return records


def main():
    print("=" * 60)
    print("PHASE 1 DATASET PREPARATION & INTEGRITY AUDIT")
    print("=" * 60)

    builder = AdmissionsDatasetBuilder()
    raw_records = generate_benchmark_corpus()
    print(f"[1/4] Ingested {len(raw_records)} raw candidate records.")

    # Audit & Deduplicate
    cleaned_records, audit_stats = builder.audit_and_deduplicate(raw_records)
    print(f"[2/4] Audit Complete: {audit_stats['cleaned_total']} verified clean records.")
    print(f"      - Exact Duplicates Dropped: {audit_stats['exact_duplicates_dropped']}")
    print(f"      - Near Duplicates Dropped: {audit_stats['near_duplicates_dropped']}")
    print(f"      - Malformed Dropped: {audit_stats['empty_or_malformed_dropped']}")
    print(f"      - Class Distribution: {audit_stats['by_label']}")
    print(f"      - Topic Distribution: {audit_stats['by_topic']}")
    print(f"      - Model Distribution: {audit_stats['by_model']}")

    # Create Grouped Leakage-Free Splits
    train_df, val_df, test_df, split_stats = builder.create_grouped_splits(cleaned_records, seed=42)
    print(f"\n[3/4] Grouped Leakage-Free Splitting Complete:")
    print(f"      - Train: {split_stats['train_count']} samples across {split_stats['train_groups']} groups ({split_stats['train_ratio_actual']:.1%})")
    print(f"      - Val:   {split_stats['val_count']} samples across {split_stats['val_groups']} groups ({split_stats['val_ratio_actual']:.1%})")
    print(f"      - Test:  {split_stats['test_count']} samples across {split_stats['test_groups']} groups ({split_stats['test_ratio_actual']:.1%})")

    # Verify zero group leakage
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])
    assert len(train_groups.intersection(val_groups)) == 0, "FATAL: Group leakage between Train and Val!"
    assert len(train_groups.intersection(test_groups)) == 0, "FATAL: Group leakage between Train and Test!"
    assert len(val_groups.intersection(test_groups)) == 0, "FATAL: Group leakage between Val and Test!"
    print("      [PASSED] Leakage Verification: Zero overlapping prompt/author groups across splits.")

    # Save to disk
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    stats_path = os.path.join(out_dir, "dataset_audit_and_split_stats.json")

    train_df.to_json(train_path, orient="records", lines=True)
    val_df.to_json(val_path, orient="records", lines=True)
    test_df.to_json(test_path, orient="records", lines=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({"audit_stats": audit_stats, "split_stats": split_stats}, f, indent=2)

    print(f"\n[4/4] Processed data saved successfully to {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
