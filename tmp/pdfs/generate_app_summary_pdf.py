from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


OUT_PATH = Path("output/pdf/app_summary_one_page.pdf")


def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def draw_heading(pdf: canvas.Canvas, text: str, x: float, y: float, size: int = 12) -> float:
    pdf.setFont("Helvetica-Bold", size)
    pdf.drawString(x, y, text)
    return y - (size + 4)


def draw_paragraph(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
    leading: int = 13,
    space_after: int = 2,
) -> float:
    lines = wrap_text(text, font_name, font_size, width)
    pdf.setFont(font_name, font_size)
    for line in lines:
        pdf.drawString(x, y, line)
        y -= leading
    return y - space_after


def draw_bullets(
    pdf: canvas.Canvas,
    bullets: list[str],
    x: float,
    y: float,
    width: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
    leading: int = 12,
) -> float:
    bullet_prefix = "- "
    bullet_width = stringWidth(bullet_prefix, font_name, font_size)
    text_width = width - bullet_width - 2

    pdf.setFont(font_name, font_size)
    for bullet in bullets:
        lines = wrap_text(bullet, font_name, font_size, text_width)
        for idx, line in enumerate(lines):
            if idx == 0:
                pdf.drawString(x, y, bullet_prefix + line)
            else:
                pdf.drawString(x + bullet_width, y, line)
            y -= leading
        y -= 1
    return y


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(OUT_PATH), pagesize=A4)
    page_width, page_height = A4
    margin = 42
    x = margin
    width = page_width - (2 * margin)
    y = page_height - margin

    y = draw_heading(pdf, "App Summary - ThesisMercedesFinal", x, y, size=16)
    y = draw_paragraph(
        pdf,
        "Single-page overview based only on repository evidence.",
        x,
        y,
        width,
        font_name="Helvetica-Oblique",
        font_size=9,
        leading=11,
        space_after=8,
    )

    y = draw_heading(pdf, "What It Is", x, y, size=12)
    y = draw_paragraph(
        pdf,
        "ThesisMercedesFinal is a local image comparison app that scores similarity between reference and generated images.",
        x,
        y,
        width,
    )
    y = draw_paragraph(
        pdf,
        "It provides both a browser GUI and a CLI, backed by the same Python metrics pipeline.",
        x,
        y,
        width,
        space_after=6,
    )

    y = draw_heading(pdf, "Who It Is For", x, y, size=12)
    y = draw_bullets(
        pdf,
        [
            "Primary user/persona: Not found in repo.",
            "Likely users (inferred): thesis researchers or engineers evaluating generated vehicle imagery.",
        ],
        x,
        y,
        width,
    )

    y = draw_heading(pdf, "What It Does", x, y, size=12)
    y = draw_bullets(
        pdf,
        [
            "Compares one image pair or batch inputs (folder mode in CLI, ZIP upload in GUI).",
            "Computes LPIPS, SSIM, DeltaE CIEDE2000, mask IoU, and mask Dice metrics.",
            "Normalizes generated images to reference size via letterbox before scoring.",
            "Supports optional car-only scoring using Mask R-CNN segmentation and two car modes.",
            "Shows a metric dashboard with pair previews and optional car-only previews.",
            "Exports results to CSV and writes normalized and optional car-only output images.",
        ],
        x,
        y,
        width,
    )

    y = draw_heading(pdf, "How It Works (Repo Evidence)", x, y, size=12)
    y = draw_bullets(
        pdf,
        [
            "Frontend (`index.html`, `script.js`, `styles.css`) collects files/options and sends POST `/api/compare`.",
            "API server (`gui_server.py`) serves static files, parses multipart uploads, and validates image and ZIP inputs.",
            "Server executes `image_metrics.py` as a subprocess with selected options.",
            "Metrics engine computes scores, writes CSV plus preview files, and returns row data.",
            "Server embeds preview images as base64 data URLs in JSON; frontend renders metrics and previews.",
            "Data store, auth service, and external persistence: Not found in repo.",
        ],
        x,
        y,
        width,
    )

    y = draw_heading(pdf, "How To Run (Minimal)", x, y, size=12)
    y = draw_bullets(
        pdf,
        [
            "Install dependencies: `py -3 -m pip install numpy pandas pillow scikit-image tqdm lpips torch torchvision`",
            "Start GUI server: `py -3 gui_server.py`",
            "Open `http://localhost:4173` in a browser.",
            "CLI alternative: `py -3 image_metrics.py --help`",
        ],
        x,
        y,
        width,
    )

    y = draw_paragraph(
        pdf,
        "Evidence files: README.md, gui_server.py, image_metrics.py, index.html, script.js, styles.css",
        x,
        y,
        width,
        font_name="Helvetica-Oblique",
        font_size=8,
        leading=10,
        space_after=0,
    )

    if y < margin:
        raise RuntimeError("Layout overflow: content exceeds one page.")

    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    main()
