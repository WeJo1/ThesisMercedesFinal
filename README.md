# ThesisMercedesFinal – Tool-Handhabung

Nutze dieses Projekt auf zwei Arten:
- Starte die **Web-GUI** (`gui_server.py`) und arbeite im Browser.
- Starte die **CLI** (`image_metrics.py`) direkt im Terminal.

## 1) Projekt einrichten

Führe die Befehle im Projektordner aus:

```bash
cd /workspace/ThesisMercedesFinal
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> Hinweis: Für `--enable-car-only` brauchst du ein funktionierendes `torch` + `torchvision` Setup.

## 2) Web-GUI starten

Starte den Server:

```bash
cd /workspace/ThesisMercedesFinal
python3 gui_server.py
```

Öffne dann im Browser:

```text
http://localhost:4173
```

Optional: Nutze einen anderen Port.

```bash
cd /workspace/ThesisMercedesFinal
PORT=8080 python3 gui_server.py
```

Dann öffne:

```text
http://localhost:8080
```

### Falls „Seite nicht erreichbar" erscheint

Führe diese Schritte der Reihe nach aus:

1. Starte den Server neu.
2. Öffne **nicht** `http://0.0.0.0:4173`, sondern `http://localhost:4173` oder `http://127.0.0.1:4173`.
3. Prüfe im Terminal, ob der Port erreichbar ist.

```bash
cd /workspace/ThesisMercedesFinal
python3 gui_server.py
```

Öffne in einem zweiten Terminal:

```bash
curl -I http://127.0.0.1:4173
```

Wenn `HTTP/1.0 200 OK` erscheint, läuft der Server korrekt.

Wenn du in Docker/Remote arbeitest, leite den Port 4173 an deinen Host weiter und rufe danach lokal `http://localhost:4173` auf.

## 3) CLI-Hilfe anzeigen

Lass dir alle verfügbaren CLI-Optionen anzeigen:

```bash
cd /workspace/ThesisMercedesFinal
python3 image_metrics.py --help
```

## 4) Einzelvergleich (zwei Bilder)

Vergleiche genau ein Referenzbild mit einem generierten Bild:

```bash
cd /workspace/ThesisMercedesFinal
python3 image_metrics.py \
  --ref "reference/schwarz.png" \
  --gen "generated/schwarz.png" \
  --out "normalized" \
  --output-csv "image_metrics_results.csv"
```

## 5) Batchvergleich (zwei Ordner)

Vergleiche alle gleichnamigen Bilder aus zwei Ordnern:

```bash
cd /workspace/ThesisMercedesFinal
python3 image_metrics.py \
  --reference-dir "reference" \
  --generated-dir "generated" \
  --out "normalized" \
  --output-csv "image_metrics_results.csv"
```

## 6) Car-only-Metriken aktivieren

Aktiviere die fahrzeugfokussierte Auswertung:

```bash
cd /workspace/ThesisMercedesFinal
python3 image_metrics.py \
  --reference-dir "reference" \
  --generated-dir "generated" \
  --enable-car-only \
  --car-mode neutralize_crop \
  --mask-source union \
  --car-only-dir car_only \
  --output-csv image_metrics_results.csv
```

Kurzform:

```bash
python3 image_metrics.py --car-only
```

## 7) Wichtige Optionen (CLI)

Nutze diese Parameter je nach Bedarf:

- `--mode letterbox` – normalisiere Bildgrößen (Default: `letterbox`)
- `--lpips-net {alex|vgg|squeeze}` – wähle LPIPS-Backbone
- `--lpips-train-mode {lin|tune|scratch}` – nutze aktuell `lin` (tune/scratch sind im Skript bewusst nicht implementiert)
- LPIPS läuft mit `spatial=True` und erzeugt standardmäßig Heatmaps
- `--lpips-heatmap-dir <ordner>` – setze Zielordner für LPIPS-Heatmaps (`none` deaktiviert)
- `--use-gpu` – nutze CUDA, wenn verfügbar
- `--seed <int>` – setze einen festen Seed für reproduzierbare Läufe
- `--deterministic` – aktiviere deterministische Backends (langsamer, aber stabiler reproduzierbar)
- `--skip-hausdorff` – überspringe Hausdorff für schnellere Batchläufe
- `--debug-dir <ordner>` – speichere Masken/Crop-Debugdaten
- `--mask-score-threshold <wert>` – setze Mindestscore für Fahrzeugsegmentierung
- `--mask-threshold <wert>` – setze Pixel-Schwelle der Segmentierungsmaske
- `--roi-min-size-px <wert>` – halte die Car-ROI mindestens auf dieser Kantenlänge
- `--roi-square` – erzwinge quadratische ROI für robustere LPIPS/SSIM-Vergleiche

## 8) Ergebnisse prüfen

Öffne die CSV-Ergebnisse:

```bash
cd /workspace/ThesisMercedesFinal
python3 - <<'PY'
import pandas as pd
print(pd.read_csv('image_metrics_results.csv').head())
PY
```

## 9) LPIPS-Modus `lin` verständlich erklärt

Nutze diese Faustregel:
- **Vorherige Situation im Skript:** Es gab keine explizite Trennung zwischen den Konzepten `lin`, `tune`, `scratch`. Dadurch konnte der Eindruck entstehen, dass alle drei Modi direkt implementiert sind.
- **Aktuelle Situation im Skript:** Erlaube bewusst nur `lin` für die Inferenz in diesem Tool. Wenn `tune` oder `scratch` gewählt wird, stoppe den Lauf mit klarer Fehlermeldung.

Was bedeutet `lin` konkret?
- Lade das offizielle LPIPS-Inferenzmodell mit trainierten linearen Kalibrierungsschichten.
- Nutze `spatial=True`, damit zusätzlich eine Distanzkarte/Heatmap entsteht.
- Nutze den gewählten Backbone (`alex`, `vgg`, `squeeze`) nur als Feature-Extraktor im LPIPS-Setup.

Was ist **nicht** enthalten?
- Kein eigenes Nachtrainieren (`tune`) im Skript.
- Kein Training von Grund auf (`scratch`) im Skript.

Merke:
- Wenn du nur Bildpaare bewerten willst, nutze `--lpips-train-mode lin`.
- Wenn du wirklich trainieren willst (`tune`/`scratch`), implementiere dafür einen separaten Trainings-Workflow.

## 10) Typischer Ablauf (kurz)

1. Aktiviere die Umgebung.
2. Starte `python3 gui_server.py` **oder** führe `python3 image_metrics.py ...` aus.
3. Prüfe `image_metrics_results.csv`.
4. Wiederhole den Lauf mit angepassten Optionen.
