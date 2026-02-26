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
const loadingIndicator = document.getElementById('loadingIndicator');
const refPreview = document.getElementById('refPreview');
const genPreview = document.getElementById('genPreview');
const carOnlyPreviewSection = document.getElementById('carOnlyPreviewSection');
const carRefPreview = document.getElementById('carRefPreview');
const carGenPreview = document.getElementById('carGenPreview');
const comparisonSection = document.getElementById('comparisonSection');
const comparisonList = document.getElementById('comparisonList');
const metricInfoBoxes = document.querySelectorAll('.metric-info');

const lpipsValue = document.getElementById('lpips');
const ssim = document.getElementById('ssim');
const deltaE = document.getElementById('deltaE');
const lpipsCar = document.getElementById('lpipsCar');
const maskIou = document.getElementById('maskIou');
const maskDice = document.getElementById('maskDice');

const fallbackApiOrigins = ['http://127.0.0.1:4173', 'http://localhost:4173'];
let isComparisonRunning = false;

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


function setLoadingState(isLoading) {
  loadingIndicator.hidden = !isLoading;
  runModel.disabled = isLoading;
}

function startCalculation(message) {
  isComparisonRunning = true;
  setStatus('running', 'Vergleiche');
  setLoadingState(true);
  previewText.textContent = message;
}

function stopCalculation() {
  isComparisonRunning = false;
  setLoadingState(false);
}

function closeMetricInfoBoxes(exceptBox = null) {
  metricInfoBoxes.forEach((box) => {
    if (box === exceptBox) {
      return;
    }
    box.open = false;
  });
}

function formatMetricPair(mainValue, percentValue) {
  const numericMain = Number(mainValue);
  const numericPercent = Number(percentValue);

  if (Number.isNaN(numericMain) || Number.isNaN(numericPercent)) {
    return '--';
  }

  return `${numericMain.toFixed(4)} (${numericPercent.toFixed(2)} %)`;
}


function formatNumeric(value, digits = 4, suffix = '') {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return '--';
  }
  return `${numeric.toFixed(digits)}${suffix}`;
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
  const hasCarPreview = Boolean(data?.car_only_ref_preview || data?.car_only_gen_preview);
  if (!hasCarPreview) {
    carOnlyPreviewSection.hidden = true;
    setPreviewImage(carRefPreview, null);
    setPreviewImage(carGenPreview, null);
    return;
  }

  setPreviewImage(carRefPreview, data.car_only_ref_preview);
  setPreviewImage(carGenPreview, data.car_only_gen_preview);
  carOnlyPreviewSection.hidden = false;
}

function renderMetrics(data) {
  lpipsValue.textContent = formatMetricPair(data.lpips, data.lpips_similarity_percent);
  ssim.textContent = formatMetricPair(data.ssim, data.ssim_percent);
  deltaE.textContent = formatMetricPair(data.delta_e_ciede2000, data.delta_e_similarity_percent);
  lpipsCar.textContent = formatMetricPair(data.lpips_car_only, data.lpips_car_only_similarity_percent);
  setMetric(maskIou, data.mask_iou);
  setMetric(maskDice, data.mask_dice);
}

function renderComparisonList(comparisons) {
  if (!Array.isArray(comparisons) || comparisons.length <= 1) {
    comparisonSection.hidden = true;
    comparisonList.innerHTML = '';
    return;
  }

  comparisonSection.hidden = false;
  comparisonList.innerHTML = '';

  comparisons.forEach((item, index) => {
    const detailsNode = document.createElement('details');
    detailsNode.className = 'comparison-item';
    detailsNode.open = index === 0;

    const summaryNode = document.createElement('summary');
    summaryNode.textContent = item.filename || `Paar ${index + 1}`;
    detailsNode.append(summaryNode);

    const contentNode = document.createElement('div');
    contentNode.className = 'comparison-item-content';
    const hasPairPreview = Boolean(item.ref_preview || item.gen_preview);
    contentNode.innerHTML = `
      ${
        hasPairPreview
          ? `<div class="comparison-mini-grid">
        <img src="${item.ref_preview || ''}" alt="Referenz ${item.filename || ''}" />
        <img src="${item.gen_preview || ''}" alt="Generated ${item.filename || ''}" />
      </div>`
          : '<p class="comparison-note">Vorschau für Batch-Elemente deaktiviert, um Stabilität und Geschwindigkeit zu verbessern.</p>'
      }
      <ul>
        <li>LPIPS: ${formatMetricPair(item.lpips, item.lpips_similarity_percent)}</li>
        <li>SSIM: ${formatMetricPair(item.ssim, item.ssim_percent)}</li>
        <li>ΔE CIEDE2000: ${formatMetricPair(item.delta_e_ciede2000, item.delta_e_similarity_percent)}</li>
      </ul>
    `;
    detailsNode.append(contentNode);

    detailsNode.addEventListener('toggle', () => {
      if (!detailsNode.open) {
        return;
      }

      comparisons.forEach((_, innerIndex) => {
        if (innerIndex === index) {
          return;
        }
        const sibling = comparisonList.children[innerIndex];
        if (sibling) {
          sibling.open = false;
        }
      });

      renderMetrics(item);
      if (item.ref_preview || item.gen_preview) {
        setPreviewImage(refPreview, item.ref_preview);
        setPreviewImage(genPreview, item.gen_preview);
      }
      updateCarOnlyPreview(item);
      previewText.textContent = `Vergleich abgeschlossen für ${item.filename}.`; 
    });

    comparisonList.append(detailsNode);
  });
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
  if (isComparisonRunning) {
    return;
  }

  const [refFile] = refImage.files;
  const [genFile] = genImage.files;

  if (!refFile || !genFile) {
    setStatus('idle', 'Bilder fehlen');
    previewText.textContent = 'Wähle zuerst Referenz- und Generated-Bild aus.';
    return;
  }

  startCalculation('Berechne Metriken mit Python-Backend...');

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
    const comparisons = Array.isArray(data.comparisons) && data.comparisons.length > 0 ? data.comparisons : [data];
    const firstComparison = comparisons[0];

    renderMetrics(firstComparison);
    setPreviewImage(refPreview, firstComparison.ref_preview);
    setPreviewImage(genPreview, firstComparison.gen_preview);
    updateCarOnlyPreview(firstComparison);
    renderComparisonList(comparisons);

    const isBatch = Boolean(data.batch_mode && data.comparison_count > 1);
    if (isBatch) {
      const previewHint = data.batch_previews_limited
        ? ' Vorschauen wurden nur für das erste Paar geladen, um Abstürze bei großen ZIP-Dateien zu vermeiden.'
        : '';
      previewText.textContent = `Vergleich abgeschlossen: ${data.comparison_count} passende Dateipaare ausgewertet.${previewHint}`;
    } else {
      previewText.textContent = `Vergleich abgeschlossen für ${firstComparison.filename}. Ergebnisse aus image_metrics.py wurden geladen.`;
    }

    setStatus('done', 'Fertig');
    logBrowser('Zeige Vergleichsergebnis', data);
  } catch (error) {
    setStatus('idle', 'Fehler');
    updateCarOnlyPreview(null);
    comparisonSection.hidden = true;
    comparisonList.innerHTML = '';
    previewText.textContent = `Fehler: ${error.message}`;
    console.error('[CompareGUI] Vergleich abgebrochen', error);
  } finally {
    stopCalculation();
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
  comparisonSection.hidden = true;
  comparisonList.innerHTML = '';

  [lpipsValue, ssim, deltaE, lpipsCar, maskIou, maskDice].forEach((node) => {
    node.textContent = node.id.includes('Similarity') ? '-- %' : '--';
  });

  previewText.textContent = 'Lade zwei Bilder hoch und starte den Vergleich.';
  setStatus('idle', 'Bereit');
  stopCalculation();
}

metricInfoBoxes.forEach((box) => {
  box.addEventListener('toggle', () => {
    if (box.open) {
      closeMetricInfoBoxes(box);
    }
  });
});

document.addEventListener('click', (event) => {
  const clickedInsideInfoBox = event.target instanceof Element && event.target.closest('.metric-info');
  if (!clickedInsideInfoBox) {
    closeMetricInfoBoxes();
  }
});

document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') {
    closeMetricInfoBoxes();
  }
});

refImage.addEventListener('change', () => showPreview(refImage, refPreview));
genImage.addEventListener('change', () => showPreview(genImage, genPreview));
runModel.addEventListener('click', runComparison);
resetForm.addEventListener('click', resetInterface);

stopCalculation();
