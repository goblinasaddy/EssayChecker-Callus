// ==========================================================================
// EssayChecker — Two-Level Evidence Attribution Engine (Frontend Logic)
// ==========================================================================

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

const emptyPlaceholder = document.getElementById("emptyPlaceholder");
const analysisView = document.getElementById("analysisView");

const verdictCard = document.getElementById("verdictCard");
const verdictBadge = document.getElementById("verdictBadge");
const verdictSummary = document.getElementById("verdictSummary");
const probNum = document.getElementById("probNum");
const probCaption = document.getElementById("probCaption");
const globalDriversList = document.getElementById("globalDriversList");

const sumHigh = document.getElementById("sumHigh");
const sumMed = document.getElementById("sumMed");
const sumHuman = document.getElementById("sumHuman");
const sumNeutral = document.getElementById("sumNeutral");

const highlightedEssayBody = document.getElementById("highlightedEssayBody");
const selectedSpanBadge = document.getElementById("selectedSpanBadge");
const selectedSentenceText = document.getElementById("selectedSentenceText");
const evidenceItemsContainer = document.getElementById("evidenceItemsContainer");
const disclaimerText = document.getElementById("disclaimerText");

// Initialize on Load
document.addEventListener("DOMContentLoaded", () => {
  setupInputListeners();
});

function setupInputListeners() {
  essayInput.addEventListener("input", updateCounts);
  
  btnClear.addEventListener("click", () => {
    essayInput.value = "";
    updateCounts();
    showEmptyState();
  });

  btnAnalyze.addEventListener("click", runAnalysis);
}

function updateCounts() {
  const text = essayInput.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  wordCountEl.textContent = words;
  charCountEl.textContent = text.length;
}

async function runAnalysis() {
  const text = essayInput.value.trim();
  if (!text || text.split(/\s+/).length < 20) {
    alert("Please provide an essay with at least 20 words for meaningful feature analysis.");
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
      alert(`Analysis Notice: ${data.message || 'Unable to process input'}`);
    }
  } catch (err) {
    console.error("Analysis failed:", err);
    alert("Failed to connect to the analysis engine. Please ensure the backend service is running.");
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(loading) {
  if (loading) {
    btnAnalyze.disabled = true;
    btnAnalyzeText.textContent = "Analyzing...";
    analyzeSpinner.style.display = "inline-block";
  } else {
    btnAnalyze.disabled = false;
    btnAnalyzeText.textContent = "Analyze Essay";
    analyzeSpinner.style.display = "none";
  }
}

function showEmptyState() {
  emptyPlaceholder.style.display = "block";
  analysisView.style.display = "none";
}

function renderAnalysisResults(data) {
  emptyPlaceholder.style.display = "none";
  analysisView.style.display = "flex";

  const assessment = data.assessment;
  const evidence = data.evidence || {};
  const spans = data.highlighted_spans || [];

  // 1. Overall Decision
  verdictBadge.textContent = assessment.category;
  verdictBadge.className = "verdict-title";
  verdictCard.className = "card verdict-box";

  if (assessment.verdict_code === "LIKELY_AI_GENERATED") {
    verdictBadge.classList.add("ai");
    verdictCard.classList.add("verdict-ai");
  } else if (assessment.verdict_code === "LIKELY_AI_ASSISTED") {
    verdictBadge.classList.add("assisted");
    verdictCard.classList.add("verdict-assisted");
  } else if (assessment.verdict_code === "UNCERTAIN") {
    verdictBadge.classList.add("uncertain");
    verdictCard.classList.add("verdict-uncertain");
  } else if (assessment.verdict_code === "INSUFFICIENT_EVIDENCE") {
    verdictBadge.classList.add("indeterminate");
    verdictCard.classList.add("verdict-indeterminate");
  }

  verdictSummary.textContent = assessment.summary;

  if (assessment.is_indeterminate) {
    probNum.textContent = "N/A";
    probCaption.textContent = "Short Text (<75 words)";
  } else {
    const probPct = (assessment.calibrated_ai_probability * 100).toFixed(1);
    probNum.textContent = `${probPct}%`;
    probCaption.textContent = `Confidence: ${assessment.confidence_level}`;
  }

  // 2. Render Top Global Drivers
  renderGlobalDrivers(evidence);

  // 3. Summary Counts
  const highCount = spans.filter(s => s.overall_severity === "high").length;
  const medCount = spans.filter(s => s.overall_severity === "medium").length;
  const humanCount = spans.filter(s => s.overall_severity === "human_grounded").length;
  const neutralCount = spans.filter(s => s.overall_severity === "neutral").length;

  sumHigh.textContent = `${highCount} Red (AI-Skewed)`;
  sumMed.textContent = `${medCount} Amber (Moderate)`;
  sumHuman.textContent = `${humanCount} Human Grounded`;
  sumNeutral.textContent = `${neutralCount} Neutral`;

  // 4. Render In-Text Spans
  renderHighlightedText(spans, highCount + medCount);

  // 5. Disclaimers
  disclaimerText.textContent = `${data.disclaimers.probabilistic_warning} ${data.disclaimers.no_llm_judge_guarantee}`;

  // Select first flagged span (or span 0)
  const firstFlagged = spans.find(s => s.overall_severity === "high" || s.overall_severity === "medium") || spans[0];
  if (firstFlagged) {
    selectSpan(firstFlagged.span_id);
  }
}

function renderGlobalDrivers(evidence) {
  globalDriversList.innerHTML = "";
  
  const aiDrivers = evidence.top_ai_evidence || [];
  const humanDrivers = evidence.top_human_evidence || [];

  if (aiDrivers.length === 0 && humanDrivers.length === 0) {
    globalDriversList.innerHTML = `<span style="font-size:0.75rem;color:var(--text-dim);">No global drivers available for short text.</span>`;
    return;
  }

  // Display top 3 AI and top 2 Human drivers
  aiDrivers.slice(0, 3).forEach(d => {
    const pill = document.createElement("div");
    pill.className = "driver-pill ai";
    pill.innerHTML = `
      <span>+ ${formatFeatureName(d.feature)}</span>
      <span>Obs: ${d.observed_value} (Impact: +${d.impact_score})</span>
    `;
    globalDriversList.appendChild(pill);
  });

  humanDrivers.slice(0, 2).forEach(d => {
    const pill = document.createElement("div");
    pill.className = "driver-pill human";
    pill.innerHTML = `
      <span>- ${formatFeatureName(d.feature)}</span>
      <span>Obs: ${d.observed_value} (Impact: ${d.impact_score})</span>
    `;
    globalDriversList.appendChild(pill);
  });
}

function renderHighlightedText(spans, flaggedCount) {
  highlightedEssayBody.innerHTML = "";
  
  if (!spans || spans.length === 0) {
    highlightedEssayBody.innerHTML = "<p>No text content found.</p>";
    return;
  }

  let currentP = document.createElement("p");

  spans.forEach((span, idx) => {
    if (idx > 0 && span.text.startsWith("\n\n")) {
      highlightedEssayBody.appendChild(currentP);
      currentP = document.createElement("p");
    }

    const spanEl = document.createElement("span");
    spanEl.className = `span-item ${span.overall_severity}`;
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

  document.querySelectorAll(".span-item").forEach(el => el.classList.remove("selected"));
  const activeEl = document.getElementById(`ui_${spanId}`);
  if (activeEl) {
    activeEl.classList.add("selected");
  }

  const span = currentAnalysis.highlighted_spans.find(s => s.span_id === spanId);
  if (!span) return;

  const severityLabel = span.overall_severity === "high" ? "AI-SKEWED — HIGH" :
                        span.overall_severity === "medium" ? "AI-SKEWED — MODERATE" :
                        span.overall_severity === "human_grounded" ? "HUMAN GROUNDED" : "NEUTRAL";

  selectedSpanBadge.textContent = `Sentence #${span.sentence_idx + 1} (${severityLabel})`;
  selectedSentenceText.textContent = `"${span.text}"`;

  // Render Evidence Items
  evidenceItemsContainer.innerHTML = "";
  if (!span.evidence_items || span.evidence_items.length === 0) {
    evidenceItemsContainer.innerHTML = `
      <div class="evidence-row-card human-skewed">
        <div class="row-head">
          <span>Standard Admissions Expression</span>
          <span>Aligned with Human Distribution</span>
        </div>
        <p class="row-desc">This sentence exhibits natural vocabulary and syntax aligned with the human reference baseline without anomalous statistical deviations.</p>
      </div>
    `;
    return;
  }

  span.evidence_items.forEach(item => {
    const card = document.createElement("div");
    card.className = `evidence-row-card ${item.direction === 'AI-skewed' ? 'ai-skewed' : 'human-skewed'}`;
    
    card.innerHTML = `
      <div class="row-head">
        <span>${item.feature_display_name}</span>
        <span>${item.evidence_strength} (${item.direction})</span>
      </div>
      <p class="row-desc">${item.explanation}</p>
      <div class="row-stats">
        <span>Observed: ${item.observed_value}</span>
        <span>Ref Mean: ${item.reference_mean} (±${item.reference_std})</span>
        <span>Z-Score: ${item.deviation_z > 0 ? '+' : ''}${item.deviation_z}σ</span>
      </div>
    `;
    evidenceItemsContainer.appendChild(card);
  });
}

function formatFeatureName(feat) {
  return feat
    .replace(/^surface_/, "")
    .replace(/^discourse_/, "")
    .replace(/^dist_/, "")
    .replace(/_/g, " ");
}
