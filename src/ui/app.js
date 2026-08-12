// ==========================================================================
// VeritasEssay Frontend Interactive Logic
// ==========================================================================

let sampleEssays = {};
let currentAnalysis = null;
let selectedSpanId = null;

// DOM Elements
const essayInput = document.getElementById("essayInput");
const wordCountEl = document.getElementById("wordCount");
const charCountEl = document.getElementById("charCount");
const btnClear = document.getElementById("btnClear");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnAnalyzeText = document.getElementById("btnAnalyzeText");
const analyzeSpinner = document.getElementById("analyzeSpinner");

const sampleInfoCard = document.getElementById("sampleInfoCard");
const sampleBadge = document.getElementById("sampleBadge");
const sampleTitle = document.getElementById("sampleTitle");
const sampleDesc = document.getElementById("sampleDesc");

const emptyPlaceholder = document.getElementById("emptyPlaceholder");
const analysisView = document.getElementById("analysisView");

const verdictText = document.getElementById("verdictText");
const verdictDot = document.getElementById("verdictDot");
const verdictSummary = document.getElementById("verdictSummary");
const confidenceText = document.getElementById("confidenceText");
const gaugeValue = document.getElementById("gaugeValue");
const gaugeFill = document.getElementById("gaugeFill");

const metricWords = document.getElementById("metricWords");
const metricSentences = document.getElementById("metricSentences");
const metricFlagged = document.getElementById("metricFlagged");
const metricMahalanobis = document.getElementById("metricMahalanobis");

const highlightedEssayBody = document.getElementById("highlightedEssayBody");
const selectedSpanBadge = document.getElementById("selectedSpanBadge");
const selectedSentenceText = document.getElementById("selectedSentenceText");
const evidenceItemsContainer = document.getElementById("evidenceItemsContainer");
const globalFactorsList = document.getElementById("globalFactorsList");
const disclaimerText = document.getElementById("disclaimerText");

// Initialize on Load
document.addEventListener("DOMContentLoaded", async () => {
  setupInputListeners();
  await loadSampleEssays();
  
  // Pre-load default human sample
  loadSample("sample_human_dumplings");
});

function setupInputListeners() {
  essayInput.addEventListener("input", updateCounts);
  
  btnClear.addEventListener("click", () => {
    essayInput.value = "";
    updateCounts();
    document.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
    sampleInfoCard.style.display = "none";
    showEmptyState();
  });

  btnAnalyze.addEventListener("click", runAnalysis);

  document.querySelectorAll(".pill-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".pill-btn").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
      const sampleId = btn.getAttribute("data-sample");
      loadSample(sampleId);
    });
  });
}

function updateCounts() {
  const text = essayInput.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  wordCountEl.textContent = words;
  charCountEl.textContent = text.length;
}

async function loadSampleEssays() {
  try {
    const res = await fetch("/api/samples");
    const data = await res.json();
    if (data.samples) {
      data.samples.forEach(s => {
        sampleEssays[s.id] = s;
      });
    }
  } catch (err) {
    console.warn("Could not fetch samples dynamically, using fallbacks:", err);
  }
}

function loadSample(sampleId) {
  const sample = sampleEssays[sampleId];
  if (!sample) return;

  sampleInfoCard.style.display = "block";
  sampleBadge.textContent = sample.badge;
  sampleBadge.className = `sample-badge ${sample.badge_color || 'emerald'}`;
  sampleTitle.textContent = sample.title;
  sampleDesc.textContent = sample.description;

  essayInput.value = sample.text;
  updateCounts();

  // Auto trigger analysis on sample load for seamless exploration
  runAnalysis();
}

async function runAnalysis() {
  const text = essayInput.value.trim();
  if (!text || text.split(/\s+/).length < 20) {
    alert("Please provide an essay with at least 20 words for meaningful admissions feature analysis.");
    return;
  }

  setLoadingState(true);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    if (data.status === "SUCCESS") {
      currentAnalysis = data;
      renderAnalysisResults(data);
    } else {
      alert(`Analysis Notice: ${data.message || 'Error occurred'}`);
    }
  } catch (err) {
    console.error("Analysis failed:", err);
    alert("Failed to connect to the analysis engine. Please verify the backend service.");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(loading) {
  if (loading) {
    btnAnalyze.disabled = true;
    btnAnalyzeText.textContent = "Extracting Signals...";
    analyzeSpinner.style.display = "inline-block";
  } else {
    btnAnalyze.disabled = false;
    btnAnalyzeText.textContent = "Analyze Measurable Evidence";
    analyzeSpinner.style.display = "none";
  }
}

function showEmptyState() {
  emptyPlaceholder.style.display = "block";
  analysisView.style.display = "none";
}

function renderAnalysisResults(data) {
  emptyPlaceholder.style.display = "none";
  analysisView.style.display = "block";

  const assessment = data.assessment;
  const metadata = data.metadata;
  const evidence = data.evidence;

  // 1. Verdict & Gauge
  verdictText.textContent = assessment.category;
  verdictText.style.color = assessment.verdict_color;
  verdictDot.style.background = assessment.verdict_color;
  verdictDot.style.boxShadow = `0 0 10px ${assessment.verdict_color}`;
  confidenceText.textContent = assessment.confidence_level;
  verdictSummary.textContent = assessment.summary;

  const probPct = (assessment.calibrated_ai_probability * 100).toFixed(1);
  gaugeValue.textContent = `Calibrated AI Score: ${probPct}%`;
  gaugeFill.style.width = `${Math.min(100, Math.max(5, probPct))}%`;

  // 2. Metrics Row
  metricWords.textContent = metadata.word_count;
  metricSentences.textContent = metadata.sentence_count;
  metricFlagged.textContent = metadata.flagged_sentence_count;
  const mahalDist = evidence.all_feature_measurements["dist_human_mahalanobis"] || 0;
  metricMahalanobis.textContent = mahalDist.toFixed(2);

  // 3. Render Interactive Highlighted Spans
  renderHighlightedText(data.highlighted_spans);

  // 4. Render Global Decision Drivers
  renderGlobalFactors(evidence);

  // 5. Render Disclaimers
  disclaimerText.innerHTML = `
    <strong>Statistical Notice:</strong> ${data.disclaimers.probabilistic_warning}<br><br>
    <strong>Fairness & Integrity:</strong> ${data.disclaimers.esl_fairness_notice} ${data.disclaimers.no_llm_judge_guarantee}
  `;

  // Select first flagged span (or span 0)
  const firstFlagged = data.highlighted_spans.find(s => s.overall_severity === "high" || s.overall_severity === "medium") || data.highlighted_spans[0];
  if (firstFlagged) {
    selectSpan(firstFlagged.span_id);
  }
}

function renderHighlightedText(spans) {
  highlightedEssayBody.innerHTML = "";
  
  if (!spans || spans.length === 0) {
    highlightedEssayBody.innerHTML = "<p>No text available.</p>";
    return;
  }

  let currentP = document.createElement("p");
  let lastParaIdx = spans[0].sentence_idx >= 0 ? 0 : 0;

  spans.forEach((span, idx) => {
    // Check if new paragraph
    if (idx > 0 && spans[idx].text.startsWith("\n\n")) {
      highlightedEssayBody.appendChild(currentP);
      currentP = document.createElement("p");
    }

    const spanEl = document.createElement("span");
    spanEl.className = `essay-span ${span.overall_severity}`;
    spanEl.id = `ui_${span.span_id}`;
    spanEl.textContent = span.text + " ";
    spanEl.title = `Sentence #${span.sentence_idx + 1} (${span.overall_severity.toUpperCase()})`;

    spanEl.addEventListener("click", () => {
      selectSpan(span.span_id);
    });

    currentP.appendChild(spanEl);
  });

  highlightedEssayBody.appendChild(currentP);
}

function selectSpan(spanId) {
  selectedSpanId = spanId;

  // Highlight active span in reader
  document.querySelectorAll(".essay-span").forEach(el => el.classList.remove("selected"));
  const activeEl = document.getElementById(`ui_${spanId}`);
  if (activeEl) {
    activeEl.classList.add("selected");
  }

  const span = currentAnalysis.highlighted_spans.find(s => s.span_id === spanId);
  if (!span) return;

  selectedSpanBadge.textContent = `Sentence #${span.sentence_idx + 1} (${span.overall_severity.toUpperCase()})`;
  selectedSentenceText.textContent = `"${span.text}"`;

  // Render Evidence Items
  evidenceItemsContainer.innerHTML = "";
  if (!span.evidence_items || span.evidence_items.length === 0) {
    evidenceItemsContainer.innerHTML = `
      <div class="evidence-item-card human-skewed">
        <div class="item-card-header">
          <span>Standard Admissions Expression</span>
          <span>Aligned with Human Distribution</span>
        </div>
        <p class="item-card-desc">This sentence does not exhibit anomalous abstract vocabulary concentration or formulaic syntactical wrappers.</p>
      </div>
    `;
    return;
  }

  span.evidence_items.forEach(item => {
    const card = document.createElement("div");
    card.className = `evidence-item-card ${item.direction === 'AI-skewed' ? 'ai-skewed' : 'human-skewed'}`;
    
    card.innerHTML = `
      <div class="item-card-header">
        <span>${item.feature_display_name}</span>
        <span>${item.evidence_strength} (${item.direction})</span>
      </div>
      <p class="item-card-desc">${item.explanation}</p>
      <div class="item-card-stats">
        <span>Observed: ${item.observed_value}</span>
        <span>Human Ref: ${item.reference_mean} (±${item.reference_std})</span>
        <span>Z-Score: ${item.deviation_z > 0 ? '+' : ''}${item.deviation_z}σ</span>
      </div>
    `;
    evidenceItemsContainer.appendChild(card);
  });
}

function renderGlobalFactors(evidence) {
  globalFactorsList.innerHTML = "";
  
  const aiItems = evidence.top_ai_evidence || [];
  const humanItems = evidence.top_human_evidence || [];

  aiItems.slice(0, 3).forEach(item => {
    const row = document.createElement("div");
    row.className = "factor-pill ai";
    row.innerHTML = `
      <span>⬆ ${formatFeatureName(item.feature)}</span>
      <span>Obs: ${item.observed_value} (Ref: ${item.reference_mean})</span>
    `;
    globalFactorsList.appendChild(row);
  });

  humanItems.slice(0, 3).forEach(item => {
    const row = document.createElement("div");
    row.className = "factor-pill human";
    row.innerHTML = `
      <span>⬇ ${formatFeatureName(item.feature)}</span>
      <span>Obs: ${item.observed_value} (Ref: ${item.reference_mean})</span>
    `;
    globalFactorsList.appendChild(row);
  });
}

function formatFeatureName(feat) {
  return feat
    .replace(/^surface_/, "")
    .replace(/^discourse_/, "")
    .replace(/^dist_/, "")
    .replace(/_/g, " ");
}
