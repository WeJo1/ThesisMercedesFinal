const temperature = document.getElementById('temperature');
const steps = document.getElementById('steps');
const temperatureValue = document.getElementById('temperatureValue');
const stepsValue = document.getElementById('stepsValue');
const runModel = document.getElementById('runModel');
const resetForm = document.getElementById('resetForm');
const statusBadge = document.getElementById('statusBadge');
const previewText = document.getElementById('previewText');
const runtime = document.getElementById('runtime');
const confidence = document.getElementById('confidence');
const memory = document.getElementById('memory');
const themeToggle = document.getElementById('themeToggle');

function updateSliderValues() {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
  stepsValue.textContent = steps.value;
}

function setStatus(mode, text) {
  statusBadge.className = `status-badge ${mode}`;
  statusBadge.textContent = text;
}

function simulateInference() {
  const prompt = document.getElementById('prompt').value.trim();
  const style = document.getElementById('style').value;

  if (!prompt) {
    previewText.textContent = 'Formuliere zuerst einen Prompt, damit das Modell arbeiten kann.';
    setStatus('idle', 'Prompt fehlt');
    return;
  }

  setStatus('running', 'Läuft');
  previewText.textContent = 'Das Modell verarbeitet deine Eingabe...';

  const fakeRuntime = Math.floor(250 + Math.random() * 750);
  const fakeConfidence = (82 + Math.random() * 16).toFixed(1);
  const fakeMemory = Math.floor(512 + Number(steps.value) * 3 + Number(temperature.value) * 80);

  window.setTimeout(() => {
    previewText.textContent = `Ergebnis bereit: Stil ${style}, Kreativität ${Number(
      temperature.value
    ).toFixed(2)}, Schritte ${steps.value}. Prompt: "${prompt}"`;
    runtime.textContent = `${fakeRuntime} ms`;
    confidence.textContent = `${fakeConfidence} %`;
    memory.textContent = `${fakeMemory} MB`;
    setStatus('done', 'Abgeschlossen');
  }, 700);
}

function resetInterface() {
  document.getElementById('prompt').value = '';
  document.getElementById('style').selectedIndex = 0;
  temperature.value = '0.6';
  steps.value = '35';
  updateSliderValues();
  runtime.textContent = '-- ms';
  confidence.textContent = '-- %';
  memory.textContent = '-- MB';
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
runModel.addEventListener('click', simulateInference);
resetForm.addEventListener('click', resetInterface);
themeToggle.addEventListener('click', toggleTheme);

updateSliderValues();
