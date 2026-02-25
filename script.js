const temperature = document.getElementById('temperature');
const steps = document.getElementById('steps');
const guidance = document.getElementById('guidance');
const modelProfile = document.getElementById('modelProfile');
const carOnlyMode = document.getElementById('carOnlyMode');

const temperatureValue = document.getElementById('temperatureValue');
const stepsValue = document.getElementById('stepsValue');
const guidanceValue = document.getElementById('guidanceValue');

const runModel = document.getElementById('runModel');
const resetForm = document.getElementById('resetForm');
const statusBadge = document.getElementById('statusBadge');
const previewText = document.getElementById('previewText');
const runtime = document.getElementById('runtime');
const confidence = document.getElementById('confidence');
const memory = document.getElementById('memory');
const maskFocus = document.getElementById('maskFocus');
const objectConsistency = document.getElementById('objectConsistency');
const seedValue = document.getElementById('seedValue');
const themeToggle = document.getElementById('themeToggle');

const MODEL_PROFILES = {
  balanced: { runtimeFactor: 1, confidenceBoost: 0, memoryOffset: 0 },
  quality: { runtimeFactor: 1.32, confidenceBoost: 4.5, memoryOffset: 220 },
  fast: { runtimeFactor: 0.72, confidenceBoost: -2.8, memoryOffset: -140 }
};

function updateSliderValues() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  stepsValue.textContent = steps.value;
  guidanceValue.textContent = Number(guidance.value).toFixed(1);
}

function setStatus(mode, text) {
  statusBadge.className = `status-badge ${mode}`;
  statusBadge.textContent = text;
}

function computeSeed(prompt, style) {
  const sourceText = `${prompt}-${style}-${modelProfile.value}-${carOnlyMode.checked}`;
  let hash = 0;

  for (let i = 0; i < sourceText.length; i += 1) {
    hash = (hash << 5) - hash + sourceText.charCodeAt(i);
    hash |= 0;
  }

  return Math.abs(hash).toString().padStart(8, '0').slice(0, 8);
}

function calculateInferenceResult() {
  const prompt = document.getElementById('prompt').value.trim();
  const style = document.getElementById('style').value;
  const profile = MODEL_PROFILES[modelProfile.value];

  if (!prompt) {
    return {
      isValid: false,
      message: 'Formuliere zuerst einen Prompt, damit das Modell arbeiten kann.'
    };
  }

  const stepsCount = Number(steps.value);
  const temperatureValueNumber = Number(temperature.value);
  const guidanceValueNumber = Number(guidance.value);
  const carOnlyActive = carOnlyMode.checked;

  const runtimeMs = Math.round(
    (180 + stepsCount * 12 + guidanceValueNumber * 16 + temperatureValueNumber * 120) *
      profile.runtimeFactor *
      (carOnlyActive ? 0.86 : 1.08)
  );

  const confidenceValue = Math.max(
    55,
    Math.min(
      99.9,
      71 +
        stepsCount * 0.2 +
        guidanceValueNumber * 0.85 -
        temperatureValueNumber * 9 +
        profile.confidenceBoost +
        (carOnlyActive ? 3.6 : -1.1)
    )
  );

  const memoryValue = Math.max(
    320,
    Math.round(460 + stepsCount * 5 + guidanceValueNumber * 9 + profile.memoryOffset + (carOnlyActive ? 75 : 180))
  );

  const maskFocusValue = Math.max(
    30,
    Math.min(99.9, 58 + guidanceValueNumber * 2.2 + (carOnlyActive ? 18 : -6))
  );

  const objectConsistencyValue = Math.max(
    40,
    Math.min(
      99.9,
      64 + stepsCount * 0.24 + guidanceValueNumber * 0.6 + profile.confidenceBoost - temperatureValueNumber * 6
    )
  );

  return {
    isValid: true,
    prompt,
    style,
    runtimeMs,
    confidenceValue,
    memoryValue,
    maskFocusValue,
    objectConsistencyValue,
    seed: computeSeed(prompt, style)
  };
}

function renderInferenceResult(result) {
  if (!result.isValid) {
    previewText.textContent = result.message;
    setStatus('idle', 'Prompt fehlt');
    return;
  }

  const scopeText = carOnlyMode.checked ? 'Car-Only Segmentierung aktiv' : 'Gesamtszene aktiv';

  previewText.textContent = `Ergebnis bereit: ${scopeText}. Stil ${result.style}, Profil ${modelProfile.value}, Kreativität ${Number(
    temperature.value
  ).toFixed(2)}, Guidance ${Number(guidance.value).toFixed(1)}, Schritte ${steps.value}. Prompt: "${result.prompt}"`;

  runtime.textContent = `${result.runtimeMs} ms`;
  confidence.textContent = `${result.confidenceValue.toFixed(1)} %`;
  memory.textContent = `${result.memoryValue} MB`;
  maskFocus.textContent = `${result.maskFocusValue.toFixed(1)} %`;
  objectConsistency.textContent = `${result.objectConsistencyValue.toFixed(1)} %`;
  seedValue.textContent = result.seed;
  setStatus('done', 'Abgeschlossen');
}

function runInference() {
  const result = calculateInferenceResult();

  if (!result.isValid) {
    renderInferenceResult(result);
    return;
  }

  setStatus('running', 'Läuft');
  previewText.textContent = 'Das Modell verarbeitet deine Eingabe...';

  window.setTimeout(() => {
    renderInferenceResult(result);
  }, 550);
}

function resetInterface() {
  document.getElementById('prompt').value = '';
  document.getElementById('style').selectedIndex = 0;
  modelProfile.value = 'balanced';
  carOnlyMode.checked = false;
  temperature.value = '0.6';
  steps.value = '35';
  guidance.value = '7.5';
  updateSliderValues();

  runtime.textContent = '-- ms';
  confidence.textContent = '-- %';
  memory.textContent = '-- MB';
  maskFocus.textContent = '-- %';
  objectConsistency.textContent = '-- %';
  seedValue.textContent = '--';

  previewText.textContent = 'Starte das Modell, um eine Ausgabevorschau zu sehen.';
  setStatus('idle', 'Bereit');
}

function toggleTheme() {
  document.body.classList.toggle('light-mode');
  const isLight = document.body.classList.contains('light-mode');
  themeToggle.textContent = isLight ? 'Night Mode aus' : 'Night Mode';
}

temperature.addEventListener('input', updateSliderValues);
steps.addEventListener('input', updateSliderValues);
guidance.addEventListener('input', updateSliderValues);
runModel.addEventListener('click', runInference);
resetForm.addEventListener('click', resetInterface);
themeToggle.addEventListener('click', toggleTheme);

updateSliderValues();
