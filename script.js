const refImage = document.getElementById('refImage');
const genImage = document.getElementById('genImage');
const lpipsNet = document.getElementById('lpipsNet');
const lpipsTrainMode = document.getElementById('lpipsTrainMode');
const enableHeatmap = document.getElementById('enableHeatmap');
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
const spatialSection = document.getElementById('spatialSection');
const spatialMeta = document.getElementById('spatialMeta');
const spatialMatrix = document.getElementById('spatialMatrix');
const spatialHeatmapCanvas = document.getElementById('spatialHeatmapCanvas');
const heatmapDetails = document.getElementById('heatmapDetails');
const matrixDetails = document.getElementById('matrixDetails');
const comparisonSection = document.getElementById('comparisonSection');
const comparisonList = document.getElementById('comparisonList');
const metricInfoBoxes = document.querySelectorAll('.metric-info');
const brandIcons = document.querySelector('.brand-icons');
const topbar = document.querySelector('.topbar');
const contentGrid = document.querySelector('.content-grid');

const lpipsValue = document.getElementById('lpips');
const ssim = document.getElementById('ssim');
const deltaE = document.getElementById('deltaE');
const lpipsCar = document.getElementById('lpipsCar');
const maskIou = document.getElementById('maskIou');
const maskDice = document.getElementById('maskDice');

const fallbackApiOrigins = ['http://127.0.0.1:4173', 'http://localhost:4173'];
let isComparisonRunning = false;
let lastSpatialPayload = null;

const mercedesStarSvgPath = 'icons/stern.svg';

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

function startBrandIntroAnimation() {
  if (!brandIcons) {
    return;
  }

  if (brandIcons.classList.contains('is-intro-done')) {
    return;
  }

  brandIcons.classList.add('is-intro');

  const markIntroAsDone = () => {
    brandIcons.classList.remove('is-intro');
    brandIcons.classList.add('is-intro-done');
  };

  brandIcons.addEventListener('animationend', markIntroAsDone, { once: true });
}

function setBrandLoadingSpin(isLoading) {
  if (!brandIcons) {
    return;
  }

  if (isLoading) {
    brandIcons.classList.remove('is-intro');
    brandIcons.classList.add('is-intro-done');
  }

  brandIcons.classList.toggle('is-loading-spin', isLoading);
}

function ensureMainBoardVisible() {
  if (!contentGrid) {
    return;
  }

  contentGrid.hidden = false;
  contentGrid.style.removeProperty('display');
}

function setHeaderLoadingState(isLoading) {
  if (!topbar) {
    return;
  }
  topbar.classList.toggle('is-loading', isLoading);
}

function enforceMercedesStarSvg() {
  document.documentElement.style.setProperty('--mercedes-star', `url("${mercedesStarSvgPath}")`);
}


function setLoadingState(isLoading) {
  loadingIndicator.hidden = !isLoading;
  loadingIndicator.setAttribute('aria-hidden', String(!isLoading));
  runModel.disabled = isLoading;
}

function startCalculation(message) {
  isComparisonRunning = true;
  setStatus('running', 'Vergleiche');
  setHeaderLoadingState(true);
  setBrandLoadingSpin(true);
  setLoadingState(true);
  previewText.textContent = message;
}

function stopCalculation() {
  isComparisonRunning = false;
  setHeaderLoadingState(false);
  setBrandLoadingSpin(false);
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

function hasMatchingComparisonTypes(refFile, genFile) {
  const comparesFolders = isZipFile(refFile) && isZipFile(genFile);
  const comparesImages = !isZipFile(refFile) && !isZipFile(genFile);
  return comparesFolders || comparesImages;
}

function updatePreviewState(imgTarget, hasImage) {
  const previewFigure = imgTarget.closest('.preview-figure');
  if (!previewFigure) {
    return;
  }

  previewFigure.classList.toggle('has-image', hasImage);
}

function syncHeatmapPreviewSize() {
  if (!spatialHeatmapCanvas || !refPreview) {
    return;
  }

  const referenceWidth = Math.round(refPreview.clientWidth);
  const referenceHeight = Math.round(refPreview.clientHeight);
  if (referenceWidth <= 0 || referenceHeight <= 0) {
    return;
  }

  spatialHeatmapCanvas.style.width = `${referenceWidth}px`;
  spatialHeatmapCanvas.style.height = `${referenceHeight}px`;

  const pixelRatio = window.devicePixelRatio || 1;
  spatialHeatmapCanvas.width = Math.max(1, Math.round(referenceWidth * pixelRatio));
  spatialHeatmapCanvas.height = Math.max(1, Math.round(referenceHeight * pixelRatio));
}

function showPreview(input, imgTarget) {
  const [file] = input.files;
  if (!file || isZipFile(file)) {
    setPreviewImage(imgTarget, null);
    return;
  }
  const previewUrl = URL.createObjectURL(file);
  imgTarget.dataset.objectUrl = previewUrl;
  setPreviewImage(imgTarget, previewUrl);
}

function setPreviewImage(imgTarget, value) {
  const previousObjectUrl = imgTarget.dataset.objectUrl;
  if (previousObjectUrl && previousObjectUrl !== value) {
    URL.revokeObjectURL(previousObjectUrl);
    delete imgTarget.dataset.objectUrl;
  }

  if (!value) {
    imgTarget.removeAttribute('src');
    updatePreviewState(imgTarget, false);
    syncHeatmapPreviewSize();
    return;
  }
  imgTarget.src = value;
  updatePreviewState(imgTarget, true);
  syncHeatmapPreviewSize();
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

function formatSpatialValue(value) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return '--';
  }
  return numeric.toFixed(3);
}

function getPercentileThreshold(flatValues, percentile) {
  if (!Array.isArray(flatValues) || flatValues.length === 0) {
    return Number.NaN;
  }

  const sortedValues = [...flatValues].sort((a, b) => a - b);
  const clampedPercentile = Math.min(Math.max(percentile, 0), 1);
  const index = Math.floor((sortedValues.length - 1) * clampedPercentile);
  return sortedValues[index];
}

function renderSpatialHeatmap(values, minValue, maxValue) {
  syncHeatmapPreviewSize();
  const context = spatialHeatmapCanvas.getContext('2d');
  if (!context) {
    return;
  }

  const rows = values.length;
  const cols = values[0].length;
  const imageData = context.createImageData(cols, rows);
  const range = Math.max(maxValue - minValue, 1e-8);

  values.forEach((row, rowIndex) => {
    row.forEach((cellValue, colIndex) => {
      const normalized = Math.min(Math.max((Number(cellValue) - minValue) / range, 0), 1);
      const r = Math.round(255 * normalized);
      const g = Math.round(255 * (1 - Math.abs(normalized - 0.5) * 2));
      const b = Math.round(255 * (1 - normalized));
      const idx = (rowIndex * cols + colIndex) * 4;
      imageData.data[idx] = r;
      imageData.data[idx + 1] = g;
      imageData.data[idx + 2] = b;
      imageData.data[idx + 3] = 255;
    });
  });

  const offscreen = document.createElement('canvas');
  offscreen.width = cols;
  offscreen.height = rows;
  const offscreenContext = offscreen.getContext('2d');
  if (!offscreenContext) {
    return;
  }
  offscreenContext.putImageData(imageData, 0, 0);

  context.clearRect(0, 0, spatialHeatmapCanvas.width, spatialHeatmapCanvas.height);
  context.imageSmoothingEnabled = false;
  context.drawImage(offscreen, 0, 0, spatialHeatmapCanvas.width, spatialHeatmapCanvas.height);
}

function renderSpatialMatrixTable(values, minValue, maxValue) {
  const rowCount = values.length;
  const colCount = values[0].length;
  const maxRows = Math.min(rowCount, 32);
  const maxCols = Math.min(colCount, 32);
  const range = Math.max(maxValue - minValue, 1e-8);
  const highThreshold = minValue + range * 0.75;
  const veryHighThreshold = minValue + range * 0.9;

  const tableNode = document.createElement('table');
  tableNode.className = 'spatial-table';
  const bodyNode = document.createElement('tbody');

  for (let rowIndex = 0; rowIndex < maxRows; rowIndex += 1) {
    const rowNode = document.createElement('tr');
    for (let colIndex = 0; colIndex < maxCols; colIndex += 1) {
      const cellNode = document.createElement('td');
      const numericValue = Number(values[rowIndex][colIndex]);
      cellNode.textContent = formatSpatialValue(numericValue);
      if (numericValue >= veryHighThreshold) {
        cellNode.classList.add('spatial-cell-very-high');
      } else if (numericValue >= highThreshold) {
        cellNode.classList.add('spatial-cell-high');
      }
      rowNode.append(cellNode);
    }
    bodyNode.append(rowNode);
  }

  tableNode.append(bodyNode);
  spatialMatrix.innerHTML = '';
  spatialMatrix.append(tableNode);

  if (rowCount > maxRows || colCount > maxCols) {
    const noteNode = document.createElement('p');
    noteNode.className = 'comparison-note';
    noteNode.textContent = `Anzeige gekürzt auf ${maxRows}x${maxCols} von ${rowCount}x${colCount}.`;
    spatialMatrix.append(noteNode);
  }
}

function updateSpatialOutput(data) {
  const values = data?.lpips_spatial_map?.values;
  if (!Array.isArray(values) || values.length === 0 || !Array.isArray(values[0])) {
    spatialSection.hidden = true;
    spatialMeta.textContent = '--';
    spatialMatrix.innerHTML = '';
    const context = spatialHeatmapCanvas.getContext('2d');
    if (context) {
      context.clearRect(0, 0, spatialHeatmapCanvas.width, spatialHeatmapCanvas.height);
    }
    return;
  }

  const minValue = Number(data.lpips_spatial_map?.min ?? Math.min(...values.flat()));
  const maxValue = Number(data.lpips_spatial_map?.max ?? Math.max(...values.flat()));
  const rows = Number(data.lpips_spatial_map?.rows ?? values.length);
  const cols = Number(data.lpips_spatial_map?.cols ?? values[0].length);

  spatialMeta.textContent = `Matrix: ${rows}x${cols} | min=${formatSpatialValue(minValue)} | max=${formatSpatialValue(maxValue)}`;
  renderSpatialMatrixTable(values, minValue, maxValue);
  renderSpatialHeatmap(values, minValue, maxValue);
  if (heatmapDetails) {
    heatmapDetails.open = false;
  }
  if (matrixDetails) {
    matrixDetails.open = false;
  }
  spatialSection.hidden = false;
}

function renderMetrics(data) {
  const hasCarOnlyMetric = Boolean(data?.car_only_enabled);
  const maskMetricScope = data?.mask_metric_scope || 'none';
  const hasCarMaskMetrics = Boolean(data?.mask_metrics_available) && maskMetricScope === 'car_mask';

  lpipsValue.textContent = formatMetricPair(data.lpips, data.lpips_similarity_percent);
  ssim.textContent = formatMetricPair(data.ssim, data.ssim_percent);
  deltaE.textContent = formatMetricPair(data.delta_e_ciede2000, data.delta_e_similarity_percent);
  lpipsCar.textContent = hasCarOnlyMetric
    ? formatMetricPair(data.lpips_car_only, data.lpips_car_only_similarity_percent)
    : '--';
  if (hasCarMaskMetrics) {
    setMetric(maskIou, data.mask_iou);
    setMetric(maskDice, data.mask_dice);
    maskIou.removeAttribute('title');
    maskDice.removeAttribute('title');
    return;
  }

  maskIou.textContent = '--';
  maskDice.textContent = '--';
  maskIou.title = 'Nur verfügbar, wenn Car-only/Fahrzeugsegmentierung erfolgreich war.';
  maskDice.title = 'Nur verfügbar, wenn Car-only/Fahrzeugsegmentierung erfolgreich war.';
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
          : '<p class="comparison-note">Für dieses Paar ist keine Vorschau verfügbar.</p>'
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
      updateSpatialOutput(item);
      previewText.textContent = `Vergleich abgeschlossen für ${item.filename}. LPIPS-Spatialdaten sichtbar: ${Boolean(item.lpips_spatial_map)}. Ablage: ${item.run_dir || '--'}`;
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
          `Server nicht erreichbar. Starte gui_server.py und prüfe Port 4173. Versuchte Endpunkte: ${apiCandidates.join(', ')}`,
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
    stopCalculation();
    setStatus('idle', 'Bilder fehlen');
    previewText.textContent = 'Wähle zuerst Referenz- und Vergleichsbild aus.';
    return;
  }

  if (!hasMatchingComparisonTypes(refFile, genFile)) {
    stopCalculation();
    setStatus('idle', 'Typ prüfen');
    previewText.textContent = 'Vergleiche entweder zwei Bilder oder zwei ZIP-Ordner. Mischformen sind nicht erlaubt.';
    return;
  }

  startCalculation('Metriken werden berechnet...');

  const payload = new FormData();
  payload.append('ref_image', refFile);
  payload.append('gen_image', genFile);
  payload.append('lpips_net', lpipsNet.value);
  payload.append('lpips_train_mode', lpipsTrainMode.value);
  payload.append('enable_heatmap', String(enableHeatmap.checked));
  payload.append('enable_car_only', String(carOnlyMode.checked));
  payload.append('car_mode', carMode.value);
  payload.append('mask_source', maskSource.value);

  try {
    logBrowser('Starte Vergleich', {
      refFile: refFile.name,
      genFile: genFile.name,
      lpipsNet: lpipsNet.value,
      lpipsTrainMode: lpipsTrainMode.value,
      enableHeatmap: enableHeatmap.checked,
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
    updateSpatialOutput(firstComparison);
    renderComparisonList(comparisons);

    const isBatch = Boolean(data.batch_mode && data.comparison_count > 1);
    if (isBatch) {
      const previewHint = data.batch_previews_limited
        ? ' Vorschauen wurden nur für das erste Paar geladen.'
        : '';
      previewText.textContent = `Vergleich abgeschlossen: ${data.comparison_count} Dateipaare ausgewertet.${previewHint} LPIPS-Spatialdaten sichtbar: ${Boolean(firstComparison.lpips_spatial_map)}. Ablage: ${data.run_dir}`;
    } else {
      previewText.textContent = `Vergleich abgeschlossen für ${firstComparison.filename}. LPIPS-Spatialdaten sichtbar: ${Boolean(firstComparison.lpips_spatial_map)}. Ablage: ${data.run_dir}`;
    }

    setStatus('done', 'Fertig');
    logBrowser('Zeige Vergleichsergebnis', data);
  } catch (error) {
    setStatus('idle', 'Fehler');
    updateCarOnlyPreview(null);
    updateSpatialOutput(null);
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
  lpipsTrainMode.value = 'lin';
  enableHeatmap.checked = true;
  carOnlyMode.checked = false;
  carMode.value = 'neutralize_crop';
  maskSource.value = 'ref';
  overlayOpacity.value = '0.55';

  refPreview.removeAttribute('src');
  genPreview.removeAttribute('src');
  updateCarOnlyPreview(null);
  updateSpatialOutput(null);
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
overlayOpacity.addEventListener('input', handleOverlayOpacityChange);

[refPreview, genPreview, carRefPreview, carGenPreview].forEach((imgNode) => {
  updatePreviewState(imgNode, Boolean(imgNode.getAttribute('src')));
});

stopCalculation();
ensureMainBoardVisible();
startBrandIntroAnimation();
enforceMercedesStarSvg();
syncHeatmapPreviewSize();
window.addEventListener('resize', syncHeatmapPreviewSize);
