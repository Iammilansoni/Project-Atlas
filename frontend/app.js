/* PROJECT ATLAS — Frontend Logic */

const API_BASE = '';  // Same origin — FastAPI serves both

// ─── DOM refs ───────────────────────────────────────────────────
const searchForm    = document.getElementById('searchForm');
const goalInput     = document.getElementById('goalInput');
const submitBtn     = document.getElementById('submitBtn');
const loadingSection  = document.getElementById('loadingSection');
const resultsSection  = document.getElementById('resultsSection');
const errorSection    = document.getElementById('errorSection');
const statusBadge   = document.getElementById('statusBadge');
const statusText    = document.getElementById('statusText');

// Loading steps
const loadingSteps = [
  document.getElementById('lStep1'),
  document.getElementById('lStep2'),
  document.getElementById('lStep3'),
  document.getElementById('lStep4'),
];

// ─── Status Check ───────────────────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('not ok');
    const data = await res.json();

    statusBadge.className = 'status-badge status-ready';
    statusText.textContent = `Ready · ${data.course_count.toLocaleString()} courses · ${data.llm_mode.toUpperCase()}`;
  } catch {
    statusBadge.className = 'status-badge status-error';
    statusText.textContent = 'Index not ready';
  }
}

// ─── Loading animation ──────────────────────────────────────────
let stepInterval = null;
function startLoadingAnimation() {
  let step = 0;
  loadingSteps.forEach(s => s.className = 'loading-step');

  stepInterval = setInterval(() => {
    if (step > 0) loadingSteps[step - 1].className = 'loading-step done';
    if (step < loadingSteps.length) {
      loadingSteps[step].className = 'loading-step active';
      step++;
    } else {
      clearInterval(stepInterval);
    }
  }, 900);
}
function stopLoadingAnimation() {
  clearInterval(stepInterval);
  loadingSteps.forEach(s => s.className = 'loading-step done');
}

// ─── Show / Hide helpers ────────────────────────────────────────
function showSection(id) {
  ['loadingSection', 'resultsSection', 'errorSection'].forEach(s => {
    document.getElementById(s).classList.toggle('hidden', s !== id);
  });
}
function hideAllSections() {
  ['loadingSection', 'resultsSection', 'errorSection'].forEach(s =>
    document.getElementById(s).classList.add('hidden'));
}

// ─── Render helpers ─────────────────────────────────────────────
function makeTags(arr, container, extraClass = '') {
  container.innerHTML = '';
  (arr || []).slice(0, 6).forEach(skill => {
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.textContent = skill;
    container.appendChild(tag);
  });
}

function renderModeBadge(mode) {
  const badge = document.getElementById('llmModeBadge');
  const labels = {
    claude: '◈ Claude Sonnet',
    gemini: '✦ Gemini 2.5 Flash',
    deterministic: '⚡ Smart Ranker',
  };
  badge.textContent = labels[mode] || mode;
  badge.className = `mode-badge mode-${mode}`;
}

function renderSkillGap(sg) {
  document.getElementById('currentLevel').textContent = sg.current_level;
  makeTags(sg.target_skills, document.getElementById('targetSkills'));
  makeTags(sg.missing_skills, document.getElementById('missingSkills'), 'missing');
  document.getElementById('recommendedFocus').textContent = sg.recommended_focus;
}

function renderStep(step, delay) {
  const card = document.createElement('div');
  card.className = 'step-card';
  card.style.animationDelay = `${delay}ms`;

  const skillsHtml = (step.skills_gained || []).slice(0, 5)
    .map(s => `<span class="tag">${s}</span>`).join('');

  card.innerHTML = `
    <div class="step-number">${step.step}</div>
    <div class="step-content">
      <div class="step-top">
        <div class="step-title">${step.course_title}</div>
      </div>
      <div class="step-institution">by ${step.institution}</div>
      <div class="step-meta">
        <span class="meta-chip meta-rating">★ ${step.rating || 'N/A'} · ${step.rating_label}</span>
        <span class="meta-chip meta-level">${step.level}</span>
        <span class="meta-chip meta-duration">⏱ ${step.duration}</span>
      </div>
      <div class="step-why">${step.why_this_course}</div>
      <div class="step-skills">${skillsHtml}</div>
      ${step.review_highlight ? `<div class="step-review">"${step.review_highlight}"</div>` : ''}
    </div>
  `;
  return card;
}

function renderResults(data) {
  // Header
  document.getElementById('resultGoal').textContent = data.goal;
  renderModeBadge(data.llm_mode);
  document.getElementById('timelineBadge').textContent = data.estimated_timeline;

  // Skill gap
  renderSkillGap(data.skill_gap_analysis);

  // Steps
  document.getElementById('hoursEstimate').textContent = `~${data.total_hours_estimate}`;
  const container = document.getElementById('stepsContainer');
  container.innerHTML = '';
  (data.learning_path || []).forEach((step, i) => {
    container.appendChild(renderStep(step, i * 100));
  });

  // Pro tip
  document.getElementById('proTipText').textContent = data.pro_tip;

  showSection('resultsSection');
}

// ─── Form Submit ────────────────────────────────────────────────
searchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const goal = goalInput.value.trim();
  if (!goal) return;

  submitBtn.disabled = true;
  showSection('loadingSection');
  startLoadingAnimation();

  try {
    const res = await fetch(`${API_BASE}/recommend-path`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal }),
    });

    stopLoadingAnimation();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    stopLoadingAnimation();
    showSection('errorSection');
    document.getElementById('errorMsg').textContent = `Error: ${err.message}`;
  } finally {
    submitBtn.disabled = false;
  }
});

// ─── Suggestion chips ───────────────────────────────────────────
document.querySelectorAll('.suggestion-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    goalInput.value = chip.dataset.goal;
    goalInput.focus();
  });
});

// ─── Reset buttons ──────────────────────────────────────────────
document.getElementById('resetBtn').addEventListener('click', () => {
  hideAllSections();
  goalInput.value = '';
  goalInput.focus();
});
document.getElementById('errorRetryBtn').addEventListener('click', () => {
  hideAllSections();
  goalInput.focus();
});

// ─── Init ────────────────────────────────────────────────────────
checkStatus();
setInterval(checkStatus, 30_000);
