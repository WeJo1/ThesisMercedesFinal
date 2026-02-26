const refImage = document.getElementById('refImage');
const genImage = document.getElementById('genImage');
const lpipsNet = document.getElementById('lpipsNet');
const carOnlyMode = document.getElementById('carOnlyMode');
const carMode = document.getElementById('carMode');
const maskSource = document.getElementById('maskSource');

const runModel = document.getElementById('runModel');
const resetForm = document.getElementById('resetForm');
const statusBadge = document.getElementById('statusBadge');
const previewText = document.getElementById('previewText');
const themeToggle = document.getElementById('themeToggle');
const refPreview = document.getElementById('refPreview');
const genPreview = document.getElementById('genPreview');
const carOnlyPreviewSection = document.getElementById('carOnlyPreviewSection');
const carRefPreview = document.getElementById('carRefPreview');
const carGenPreview = document.getElementById('carGenPreview');
const carMaskPreview = document.getElementById('carMaskPreview');

const lpipsValue = document.getElementById('lpips');
const lpipsSimilarity = document.getElementById('lpipsSimilarity');
const ssim = document.getElementById('ssim');
const deltaE = document.getElementById('deltaE');
const lpipsCar = document.getElementById('lpipsCar');
const maskIou = document.getElementById('maskIou');
const maskDice = document.getElementById('maskDice');

const fallbackApiOrigins = ['http://127.0.0.1:4173', 'http://localhost:4173'];

function logBrowser(message, details = null) {
  if (details === null) {
    console.log(`[CompareGUI] ${message}`);
    return;
  }
  console.log(`[CompareGUI] ${message}`, details);
}

function warnBrowser(message, details = null) {
  if (details === null) {
    console.warn(`[CompareGUI] ${message}`);
    return;
  }
  console.warn(`[CompareGUI] ${message}`, details);
}

function setStatus(mode, text) {
  statusBadge.className = `status-badge ${mode}`;
  statusBadge.textContent = text;
}

function formatMetricPair(mainValue, percentValue) {
  const numericMain = Number(mainValue);
  const numericPercent = Number(percentValue);

  if (Number.isNaN(numericMain) || Number.isNaN(numericPercent)) {
    return '--';
  }

  return `${numericMain.toFixed(4)} (${numericPercent.toFixed(2)} %)`;
}

function setMetric(target, value, suffix = '') {
  if (value === null || value === undefined || value === '') {
    target.textContent = '--';
    return;
  }

  const numeric = Number(value);
  if (!Number.isNaN(numeric)) {
    target.textContent = `${numeric.toFixed(4)}${suffix}`;
    return;
  }

  target.textContent = `${value}${suffix}`;
}

function isZipFile(file) {
  return file.name.toLowerCase().endsWith('.zip');
}

function showPreview(input, imgTarget) {
  const [file] = input.files;
  if (!file || isZipFile(file)) {
    imgTarget.removeAttribute('src');
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    imgTarget.src = String(reader.result);
  };
  reader.readAsDataURL(file);
}

function setPreviewImage(imgTarget, value) {
  if (!value) {
    imgTarget.removeAttribute('src');
    return;
  }
  imgTarget.src = value;
}

function updateCarOnlyPreview(data) {
  const hasCarPreview = Boolean(data?.car_only_ref_preview || data?.car_only_gen_preview || data?.car_only_mask_preview);
  if (!hasCarPreview) {
    carOnlyPreviewSection.hidden = true;
    setPreviewImage(carRefPreview, null);
    setPreviewImage(carGenPreview, null);
    setPreviewImage(carMaskPreview, null);
    return;
  }

  setPreviewImage(carRefPreview, data.car_only_ref_preview);
  setPreviewImage(carGenPreview, data.car_only_gen_preview);
  setPreviewImage(carMaskPreview, data.car_only_mask_preview);
  carOnlyPreviewSection.hidden = false;
}

async function parseCompareResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }

  const text = await response.text();
  return { error: text.trim() };
}

function getComparisonError(response, data) {
  const defaultMessage = 'Vergleich fehlgeschlagen';
  const serverMessage = data?.error || data?.message;

  if (serverMessage) {
    if (response.status === 501) {
      return 'POST wird vom aktiven Server nicht unterstützt. Starte stattdessen gui_server.py.';
    }
    return String(serverMessage);
  }

  if (response.status === 404) {
    return 'API-Route /api/compare fehlt. Starte gui_server.py statt python -m http.server.';
  }

  if (response.status === 501) {
    return 'POST wird vom aktiven Server nicht unterstützt. Starte stattdessen gui_server.py.';
  }

  return `${defaultMessage} (HTTP ${response.status})`;
}

function getApiCandidates() {
  const candidates = [];
  const sameOriginApi = `${window.location.origin}/api/compare`;

  candidates.push(sameOriginApi);

  fallbackApiOrigins.forEach((origin) => {
    candidates.push(`${origin}/api/compare`);
  });

  return [...new Set(candidates)];
}

async function sendComparisonRequest(payload) {
  const apiCandidates = getApiCandidates();
  let lastError = null;

  for (const apiUrl of apiCandidates) {
    try {
      logBrowser('Sende Anfrage an API', { apiUrl });
      const response = await fetch(apiUrl, { method: 'POST', body: payload });
      const data = await parseCompareResponse(response);

      logBrowser('Empfange API-Antwort', {
        apiUrl,
        status: response.status,
        ok: response.ok,
        data,
      });

      if (response.ok) {
        return data;
      }

      const errorMessage = getComparisonError(response, data);
      const shouldTryFallback = (response.status === 404 || response.status === 501) && apiUrl !== apiCandidates[apiCandidates.length - 1];
      if (shouldTryFallback) {
        warnBrowser('Primäre API nicht kompatibel. Nutze Fallback.', {
          apiUrl,
          status: response.status,
          errorMessage,
        });
        continue;
      }

      throw new Error(errorMessage);
    } catch (error) {
      lastError = error;
      const canRetry = apiUrl !== apiCandidates[apiCandidates.length - 1];
      if (canRetry) {
        warnBrowser('API-Anfrage fehlgeschlagen. Probiere nächsten Endpunkt.', {
          apiUrl,
          error: error.message,
        });
        continue;
      }

      if (error instanceof TypeError) {
        throw new Error(
          `Backend nicht erreichbar. Starte gui_server.py und prüfe Port 4173. Versuchte Endpunkte: ${apiCandidates.join(', ')}`,
        );
      }

      break;
    }
  }

  throw lastError || new Error('Vergleich fehlgeschlagen');
}

async function runComparison() {
  const [refFile] = refImage.files;
  const [genFile] = genImage.files;

  if (!refFile || !genFile) {
    setStatus('idle', 'Bilder fehlen');
    previewText.textContent = 'Wähle zuerst Referenz- und Generated-Bild aus.';
    return;
  }

  setStatus('running', 'Vergleiche');
  previewText.textContent = 'Berechne Metriken mit Python-Backend...';

  const payload = new FormData();
  payload.append('ref_image', refFile);
  payload.append('gen_image', genFile);
  payload.append('lpips_net', lpipsNet.value);
  payload.append('enable_car_only', String(carOnlyMode.checked));
  payload.append('car_mode', carMode.value);
  payload.append('mask_source', maskSource.value);

  try {
    logBrowser('Starte Vergleich', {
      refFile: refFile.name,
      genFile: genFile.name,
      lpipsNet: lpipsNet.value,
      enableCarOnly: carOnlyMode.checked,
      carMode: carMode.value,
      maskSource: maskSource.value,
    });
    const data = await sendComparisonRequest(payload);

    setMetric(lpipsValue, data.lpips);
    setMetric(lpipsSimilarity, data.lpips_similarity_percent, ' %');
    ssim.textContent = formatMetricPair(data.ssim, data.ssim_percent);
    setMetric(deltaE, data.delta_e_ciede2000);
    lpipsCar.textContent = formatMetricPair(data.lpips_car_only, data.lpips_car_only_similarity_percent);
    setMetric(maskIou, data.mask_iou);
    setMetric(maskDice, data.mask_dice);
    updateCarOnlyPreview(data);

    previewText.textContent = `Vergleich abgeschlossen für ${data.filename}. Ergebnisse aus image_metrics.py wurden geladen.`;
    setStatus('done', 'Fertig');
    logBrowser('Zeige Vergleichsergebnis', data);
  } catch (error) {
    setStatus('idle', 'Fehler');
    updateCarOnlyPreview(null);
    previewText.textContent = `Fehler: ${error.message}`;
    console.error('[CompareGUI] Vergleich abgebrochen', error);
  }
}

function resetInterface() {
  refImage.value = '';
  genImage.value = '';
  lpipsNet.value = 'alex';
  carOnlyMode.checked = false;
  carMode.value = 'neutralize_crop';
  maskSource.value = 'ref';

  refPreview.removeAttribute('src');
  genPreview.removeAttribute('src');
  updateCarOnlyPreview(null);

  [lpipsValue, lpipsSimilarity, ssim, deltaE, lpipsCar, maskIou, maskDice].forEach((node) => {
    node.textContent = node.id.includes('Similarity') ? '-- %' : '--';
  });

  previewText.textContent = 'Lade zwei Bilder hoch und starte den Vergleich.';
  setStatus('idle', 'Bereit');
}

function toggleTheme() {
  document.body.classList.toggle('light-mode');
  const isLight = document.body.classList.contains('light-mode');
  themeToggle.textContent = isLight ? 'Night Mode aus' : 'Night Mode';
}

refImage.addEventListener('change', () => showPreview(refImage, refPreview));
genImage.addEventListener('change', () => showPreview(genImage, genPreview));
runModel.addEventListener('click', runComparison);
resetForm.addEventListener('click', resetInterface);
themeToggle.addEventListener('click', toggleTheme);
