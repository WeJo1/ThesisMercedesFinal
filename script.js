const refImage = document.getElementById('refImage');
const genImage = document.getElementById('genImage');
const lpipsNet = document.getElementById('lpipsNet');
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
const spatialSummary = document.getElementById('spatialSummary');
const spatialHotspots = document.getElementById('spatialHotspots');
const spatialAggregated = document.getElementById('spatialAggregated');
const spatialLocalInspector = document.getElementById('spatialLocalInspector');
const spatialMatrixNotice = document.getElementById('spatialMatrixNotice');
const spatialHeatmapCanvas = document.getElementById('spatialHeatmapCanvas');
const heatmapDetails = document.getElementById('heatmapDetails');
const aggregatedDetails = document.getElementById('aggregatedDetails');
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
const largeSpatialCellLimit = 12000;
const spatialHotspotLimit = 10;
const localInspectorRadius = 2;

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

function isSpatialDomReady() {
  return Boolean(
    spatialSection
    && heatmapDetails
    && aggregatedDetails
    && matrixDetails
    && spatialMeta
    && spatialSummary
    && spatialHotspots
    && spatialAggregated
    && spatialLocalInspector
    && spatialMatrixNotice
    && spatialMatrix
    && spatialHeatmapCanvas,
  );
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

function isVisiblePreviewImage(imgNode) {
  if (!imgNode || !imgNode.getAttribute('src')) {
    return false;
  }

  const rect = imgNode.getBoundingClientRect();
  return rect.width > 0 && rect.height > 0;
}

function getActiveHeatmapPreviewImage() {
  if (isVisiblePreviewImage(genPreview)) {
    return genPreview;
  }

  if (isVisiblePreviewImage(refPreview)) {
    return refPreview;
  }

  return null;
}

function getRenderedImageSize(imgNode) {
  if (!imgNode) {
    return null;
  }

  const rect = imgNode.getBoundingClientRect();
  const widthFromRect = Math.round(rect.width);
  const heightFromRect = Math.round(rect.height);

  if (widthFromRect > 0 && heightFromRect > 0) {
    return { width: widthFromRect, height: heightFromRect };
  }

  const fallbackWidth = Math.round(imgNode.clientWidth || imgNode.naturalWidth || 0);
  const fallbackHeight = Math.round(imgNode.clientHeight || imgNode.naturalHeight || 0);
  if (fallbackWidth > 0 && fallbackHeight > 0) {
    return { width: fallbackWidth, height: fallbackHeight };
  }

  return null;
}

function syncHeatmapPreviewSize() {
  if (!spatialHeatmapCanvas || !spatialSection) {
    return;
  }

  const targetPreview = getActiveHeatmapPreviewImage();
  const renderedSize = getRenderedImageSize(targetPreview);
  if (!renderedSize) {
    spatialSection.style.removeProperty('--spatial-target-width');
    spatialHeatmapCanvas.style.removeProperty('width');
    spatialHeatmapCanvas.style.removeProperty('height');
    return;
  }

  const { width: referenceWidth, height: referenceHeight } = renderedSize;

  spatialSection.style.setProperty('--spatial-target-width', `${referenceWidth}px`);
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

function formatPercent(value, digits = 2) {
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return '--';
  }
  return `${numeric.toFixed(digits)} %`;
}

function getPercentileValue(sortedValues, percentile) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
    return Number.NaN;
  }
  const clampedPercentile = Math.min(Math.max(percentile, 0), 1);
  const index = Math.floor((sortedValues.length - 1) * clampedPercentile);
  return sortedValues[index];
}

function computeSpatialStats(flatValues) {
  const safeValues = Array.isArray(flatValues) ? flatValues.filter((value) => Number.isFinite(value)) : [];
  if (safeValues.length === 0) {
    return null;
  }

  const sortedValues = [...safeValues].sort((a, b) => a - b);
  const count = sortedValues.length;
  const sum = sortedValues.reduce((acc, value) => acc + value, 0);
  const min = sortedValues[0];
  const max = sortedValues[count - 1];
  const mean = sum / count;
  const p90 = getPercentileValue(sortedValues, 0.9);
  const p95 = getPercentileValue(sortedValues, 0.95);
  const p99 = getPercentileValue(sortedValues, 0.99);
  const median = getPercentileValue(sortedValues, 0.5);
  const aboveP95 = safeValues.filter((value) => value > p95).length;
  const aboveP99 = safeValues.filter((value) => value > p99).length;

  return {
    count,
    min,
    max,
    mean,
    median,
    p90,
    p95,
    p99,
    range: max - min,
    aboveP95Share: (aboveP95 / count) * 100,
    aboveP99Share: (aboveP99 / count) * 100,
  };
}

function computeHotspots(values, maxCount = spatialHotspotLimit, globalMax = Number.NaN) {
  if (!Array.isArray(values) || values.length === 0 || !Array.isArray(values[0])) {
    return [];
  }

  const hotspots = [];
  for (let rowIndex = 0; rowIndex < values.length; rowIndex += 1) {
    for (let colIndex = 0; colIndex < values[rowIndex].length; colIndex += 1) {
      const numericValue = Number(values[rowIndex][colIndex]);
      if (!Number.isFinite(numericValue)) {
        continue;
      }
      hotspots.push({ row: rowIndex, col: colIndex, value: numericValue });
    }
  }

  hotspots.sort((left, right) => {
    if (right.value !== left.value) {
      return right.value - left.value;
    }
    if (left.row !== right.row) {
      return left.row - right.row;
    }
    return left.col - right.col;
  });

  const denominator = Math.max(Number(globalMax), 1e-8);
  return hotspots.slice(0, Math.max(1, maxCount)).map((entry, index) => {
    const normalizedToMax = entry.value / denominator;
    const category = normalizedToMax >= 0.9 ? 'sehr hoch' : normalizedToMax >= 0.75 ? 'hoch' : 'moderat';
    return {
      ...entry,
      rank: index + 1,
      normalizedToMax,
      category,
    };
  });
}

function buildAggregatedSpatialGrid(values, targetRows = 12, targetCols = 12, mode = 'mean') {
  if (!Array.isArray(values) || values.length === 0 || !Array.isArray(values[0])) {
    return null;
  }
  const sourceRows = values.length;
  const sourceCols = values[0].length;
  const rowBins = Math.max(1, Math.min(targetRows, sourceRows));
  const colBins = Math.max(1, Math.min(targetCols, sourceCols));
  const aggregatedValues = Array.from({ length: rowBins }, () => Array(colBins).fill(0));

  for (let aggRow = 0; aggRow < rowBins; aggRow += 1) {
    const rowStart = Math.floor((aggRow * sourceRows) / rowBins);
    const rowEnd = Math.floor(((aggRow + 1) * sourceRows) / rowBins);
    for (let aggCol = 0; aggCol < colBins; aggCol += 1) {
      const colStart = Math.floor((aggCol * sourceCols) / colBins);
      const colEnd = Math.floor(((aggCol + 1) * sourceCols) / colBins);
      let sum = 0;
      let count = 0;
      let max = -Infinity;

      for (let rowIndex = rowStart; rowIndex < Math.max(rowEnd, rowStart + 1); rowIndex += 1) {
        for (let colIndex = colStart; colIndex < Math.max(colEnd, colStart + 1); colIndex += 1) {
          const numericValue = Number(values[rowIndex][colIndex]);
          if (!Number.isFinite(numericValue)) {
            continue;
          }
          sum += numericValue;
          count += 1;
          max = Math.max(max, numericValue);
        }
      }

      aggregatedValues[aggRow][aggCol] = mode === 'max' ? max : sum / Math.max(count, 1);
    }
  }

  return {
    rows: rowBins,
    cols: colBins,
    values: aggregatedValues,
    mode,
  };
}

function buildSpatialAnalysis(values) {
  if (!Array.isArray(values) || values.length === 0 || !Array.isArray(values[0])) {
    return null;
  }
  const rows = values.length;
  const cols = values[0].length;
  const flatValues = [];
  values.forEach((row) => {
    row.forEach((cellValue) => {
      const numericValue = Number(cellValue);
      if (Number.isFinite(numericValue)) {
        flatValues.push(numericValue);
      }
    });
  });

  const stats = computeSpatialStats(flatValues);
  if (!stats) {
    return null;
  }

  const min = stats.min;
  const max = stats.max;
  const hotspotEntries = computeHotspots(values, spatialHotspotLimit, max);
  const aggregatedGrid = buildAggregatedSpatialGrid(values, 12, 12, 'mean');

  return {
    values,
    rows,
    cols,
    min,
    max,
    mean: stats.mean,
    median: stats.median,
    p90: stats.p90,
    p95: stats.p95,
    p99: stats.p99,
    range: stats.range,
    flatValues,
    hotspotEntries,
    aggregatedGrid,
    thresholdStats: {
      aboveP95Share: stats.aboveP95Share,
      aboveP99Share: stats.aboveP99Share,
    },
    totalCells: rows * cols,
  };
}

function renderSpatialHeatmap(values, minValue, maxValue) {
  syncHeatmapPreviewSize();
  const context = spatialHeatmapCanvas.getContext('2d');
  if (!context) {
    return;
  }

  const pixelRatio = window.devicePixelRatio || 1;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

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

  const drawWidth = spatialHeatmapCanvas.width / pixelRatio;
  const drawHeight = spatialHeatmapCanvas.height / pixelRatio;
  context.clearRect(0, 0, drawWidth, drawHeight);
  context.imageSmoothingEnabled = false;
  context.drawImage(offscreen, 0, 0, drawWidth, drawHeight);
}

function renderExactSpatialMatrix(values, minValue, maxValue) {
  const rowCount = values.length;
  const colCount = values[0].length;
  const range = Math.max(maxValue - minValue, 1e-8);
  const highThreshold = minValue + range * 0.75;
  const veryHighThreshold = minValue + range * 0.9;

  const tableNode = document.createElement('table');
  tableNode.className = 'spatial-table';
  const bodyNode = document.createElement('tbody');

  for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
    const rowNode = document.createElement('tr');
    for (let colIndex = 0; colIndex < colCount; colIndex += 1) {
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
}

function renderSpatialSummary(analysis) {
  if (!spatialSummary || !analysis) {
    return;
  }
  const summaryEntries = [
    ['Matrix', `${analysis.rows} × ${analysis.cols}`],
    ['Min', formatSpatialValue(analysis.min)],
    ['Max', formatSpatialValue(analysis.max)],
    ['Median', formatSpatialValue(analysis.median)],
  ];

  spatialSummary.innerHTML = summaryEntries
    .map(([label, value]) => `<article class="spatial-stat-card"><p class="spatial-stat-label">${label}</p><p class="spatial-stat-value">${value}</p></article>`)
    .join('');
}

function renderSpatialHotspots(analysis) {
  if (!spatialHotspots || !analysis) {
    return;
  }
  const rows = analysis.hotspotEntries
    .map((entry) => {
      return `<tr>
        <td>${entry.rank}</td>
        <td>${entry.row}</td>
        <td>${entry.col}</td>
        <td>${formatSpatialValue(entry.value)}</td>
        <td><span class="spatial-badge spatial-badge-${entry.category.replace(' ', '-')}">${entry.category}</span></td>
      </tr>`;
    })
    .join('');

  spatialHotspots.innerHTML = `<h4>Hotspots (Top ${spatialHotspotLimit})</h4>
    <div class="spatial-hotspot-table-wrap">
      <table class="spatial-hotspot-table">
        <thead><tr><th>Rang</th><th>Row</th><th>Col</th><th>Wert</th><th>Klasse</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function renderAggregatedSpatialGrid(analysis) {
  if (!spatialAggregated || !analysis?.aggregatedGrid) {
    return;
  }
  const grid = analysis.aggregatedGrid;
  const minValue = Math.min(...grid.values.flat());
  const maxValue = Math.max(...grid.values.flat());
  const range = Math.max(maxValue - minValue, 1e-8);

  const tableRows = grid.values
    .map((row) => {
      const cells = row
        .map((value) => {
          const normalized = Math.min(Math.max((value - minValue) / range, 0), 1);
          const alpha = 0.18 + normalized * 0.72;
          const color = `rgba(${Math.round(255 * normalized)}, ${Math.round(185 - normalized * 85)}, ${Math.round(255 * (1 - normalized))}, ${alpha})`;
          return `<td style="background:${color}" title="${formatSpatialValue(value)}">${formatSpatialValue(value)}</td>`;
        })
        .join('');
      return `<tr>${cells}</tr>`;
    })
    .join('');

  spatialAggregated.innerHTML = `<p class="spatial-meta">Blöcke: ${grid.rows}×${grid.cols} (${grid.mode})</p><table class="spatial-aggregated-table"><tbody>${tableRows}</tbody></table>`;
}

function renderLocalSpatialInspector(analysis, centerRow, centerCol) {
  if (!spatialLocalInspector || !analysis) {
    return;
  }
  const rowStart = Math.max(0, centerRow - localInspectorRadius);
  const rowEnd = Math.min(analysis.rows - 1, centerRow + localInspectorRadius);
  const colStart = Math.max(0, centerCol - localInspectorRadius);
  const colEnd = Math.min(analysis.cols - 1, centerCol + localInspectorRadius);

  const localRows = [];
  for (let rowIndex = rowStart; rowIndex <= rowEnd; rowIndex += 1) {
    const cells = [];
    for (let colIndex = colStart; colIndex <= colEnd; colIndex += 1) {
      const isSelected = rowIndex === centerRow && colIndex === centerCol;
      cells.push(`<td class="${isSelected ? 'is-selected' : ''}">${formatSpatialValue(analysis.values[rowIndex][colIndex])}</td>`);
    }
    localRows.push(`<tr>${cells.join('')}</tr>`);
  }

  spatialLocalInspector.innerHTML = `<h4>Lokale Inspektion</h4>
    <p class="spatial-meta">Zelle [row=${centerRow}, col=${centerCol}] = ${formatSpatialValue(analysis.values[centerRow][centerCol])}</p>
    <table class="spatial-local-table"><tbody>${localRows.join('')}</tbody></table>`;
  spatialLocalInspector.hidden = false;
}

function resetSpatialOutput() {
  if (!isSpatialDomReady()) {
    warnBrowser('Spatial-Ausgabe kann nicht zurückgesetzt werden: DOM-Container fehlen.');
    lastSpatialPayload = null;
    return;
  }

  spatialSection.hidden = true;
  spatialMeta.textContent = '--';
  spatialSummary.innerHTML = '';
  spatialHotspots.innerHTML = '';
  spatialAggregated.innerHTML = '';
  spatialLocalInspector.innerHTML = '';
  spatialLocalInspector.hidden = true;
  spatialMatrixNotice.textContent = 'Diese Ansicht rendert die gesamte Matrix erst bei Bedarf.';
  spatialMatrix.innerHTML = '';
  spatialMatrix.dataset.fullRendered = 'false';
  spatialMatrix.dataset.spatialKey = '';
  if (matrixDetails) {
    matrixDetails.open = false;
  }
  if (aggregatedDetails) {
    aggregatedDetails.open = false;
  }
  if (heatmapDetails) {
    heatmapDetails.open = true;
  }
  const context = spatialHeatmapCanvas.getContext('2d');
  if (context) {
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, spatialHeatmapCanvas.width, spatialHeatmapCanvas.height);
  }
  lastSpatialPayload = null;
}

function getSpatialCellFromCanvasEvent(analysis, event) {
  const rect = spatialHeatmapCanvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const col = Math.min(analysis.cols - 1, Math.max(0, Math.floor((x / rect.width) * analysis.cols)));
  const row = Math.min(analysis.rows - 1, Math.max(0, Math.floor((y / rect.height) * analysis.rows)));
  return { row, col };
}

function prepareExactMatrixPanel(analysis) {
  if (!spatialMatrixNotice) {
    return;
  }
  const isLargeMatrix = analysis.totalCells > largeSpatialCellLimit;
  spatialMatrix.innerHTML = '';
  spatialMatrix.dataset.fullRendered = 'false';
  spatialMatrix.dataset.spatialKey = `${analysis.rows}x${analysis.cols}:${analysis.min}:${analysis.max}`;
  spatialMatrixNotice.innerHTML = isLargeMatrix
    ? `Diese Vollmatrix ist sehr groß (${analysis.totalCells} Zellen) und kann die Darstellung verlangsamen.`
    : 'Diese Ansicht rendert die gesamte Matrix erst bei Bedarf.';
}

function attachSpatialDetailListeners() {
  if (matrixDetails) {
    matrixDetails.addEventListener('toggle', () => {
      if (!matrixDetails.open || !lastSpatialPayload) {
        return;
      }
      if (spatialMatrix.dataset.fullRendered === 'true') {
        return;
      }
      renderExactSpatialMatrix(lastSpatialPayload.values, lastSpatialPayload.min, lastSpatialPayload.max);
      spatialMatrix.dataset.fullRendered = 'true';
    });
  }

  if (spatialHeatmapCanvas) {
    spatialHeatmapCanvas.addEventListener('click', (event) => {
      if (!lastSpatialPayload) {
        return;
      }
      const pickedCell = getSpatialCellFromCanvasEvent(lastSpatialPayload, event);
      if (!pickedCell) {
        return;
      }
      renderLocalSpatialInspector(lastSpatialPayload, pickedCell.row, pickedCell.col);
    });
  }
}

function updateSpatialOutput(data) {
  if (!isSpatialDomReady()) {
    warnBrowser('Spatial-Ausgabe übersprungen: DOM-Container fehlen oder sind veraltet.');
    return;
  }

  const values = data?.lpips_spatial_map?.values;
  const analysis = buildSpatialAnalysis(values);
  if (!analysis) {
    resetSpatialOutput();
    return;
  }

  lastSpatialPayload = analysis;
  spatialMeta.textContent = `Matrix: ${analysis.rows}x${analysis.cols} | min=${formatSpatialValue(analysis.min)} | max=${formatSpatialValue(analysis.max)} | mean=${formatSpatialValue(analysis.mean)}`;
  renderSpatialHeatmap(analysis.values, analysis.min, analysis.max);
  renderSpatialSummary(analysis);
  renderSpatialHotspots(analysis);
  renderAggregatedSpatialGrid(analysis);
  spatialLocalInspector.innerHTML = '';
  spatialLocalInspector.hidden = true;
  prepareExactMatrixPanel(analysis);

  if (heatmapDetails) {
    heatmapDetails.open = true;
  }
  if (aggregatedDetails) {
    aggregatedDetails.open = false;
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
      previewText.textContent = createCompletionMessage({
        filename: item.filename,
        runDir: item.run_dir,
      });
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

function createCompletionMessage({ filename, comparisonCount, isBatch, batchPreviewsLimited, runDir }) {
  const normalizedFilename = filename || 'ausgewähltes Paar';
  const storageHint = runDir ? ` Ergebnisse liegen unter: ${runDir}.` : '';

  if (isBatch) {
    const previewHint = batchPreviewsLimited
      ? ' Lade weitere Paare über die Liste links.'
      : '';
    return `Vergleich abgeschlossen. Werte ${comparisonCount} Dateipaare aus.${previewHint}${storageHint}`;
  }

  return `Vergleich abgeschlossen für ${normalizedFilename}.${storageHint}`;
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
  payload.append('enable_heatmap', String(enableHeatmap.checked));
  payload.append('enable_car_only', String(carOnlyMode.checked));
  payload.append('car_mode', carMode.value);
  payload.append('mask_source', maskSource.value);

  try {
    logBrowser('Starte Vergleich', {
      refFile: refFile.name,
      genFile: genFile.name,
      lpipsNet: lpipsNet.value,
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
    previewText.textContent = createCompletionMessage({
      filename: firstComparison.filename,
      comparisonCount: data.comparison_count,
      isBatch,
      batchPreviewsLimited: Boolean(data.batch_previews_limited),
      runDir: data.run_dir,
    });

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
  enableHeatmap.checked = true;
  carOnlyMode.checked = false;
  carMode.value = 'neutralize_crop';
  maskSource.value = 'ref';

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
attachSpatialDetailListeners();

[refPreview, genPreview, carRefPreview, carGenPreview].forEach((imgNode) => {
  updatePreviewState(imgNode, Boolean(imgNode.getAttribute('src')));
  imgNode.addEventListener('load', syncHeatmapPreviewSize);
});

stopCalculation();
ensureMainBoardVisible();
startBrandIntroAnimation();
enforceMercedesStarSvg();
syncHeatmapPreviewSize();
window.addEventListener('resize', syncHeatmapPreviewSize);
