import argparse
import os
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import color
from skimage.filters import threshold_otsu
from skimage.metrics import hausdorff_distance, structural_similarity
from tqdm import tqdm

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
LETTERBOX_PAD_COLOR = (127, 127, 127)  # Neutralgrau reduziert harte Kontraste im Randbereich.


def load_image(path):
    """Lade ein Bild robust als RGB-Array in float32 [0,1] und ignoriere DPI."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr


def np_to_pil_uint8(img):
    """Konvertiere ein Float-Bild [0,1] nach PIL-Image (uint8)."""
    clipped = np.clip(img, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")


def normalize_pair(ref_img, gen_img, mode="letterbox", pad_color=LETTERBOX_PAD_COLOR):
    """
    Normalisiere ein Bildpaar auf identische Pixelgröße.

    - ref_norm bleibt exakt in Referenzgröße.
    - gen_norm erhält exakt Breite/Höhe der Referenz.
    - mode='letterbox': Seitenverhältnis beibehalten + Padding.
    - mode='stretch': Direktes Resize ohne Seitenverhältnis.
    """
    if mode not in {"letterbox", "stretch"}:
        raise ValueError("mode muss 'letterbox' oder 'stretch' sein")

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm = ref_img.copy()
    gen_pil = np_to_pil_uint8(gen_img)

    if mode == "stretch":
        resized = gen_pil.resize((ref_w, ref_h), resample=Image.Resampling.LANCZOS)
        gen_norm = np.asarray(resized, dtype=np.float32) / 255.0
        return ref_norm, gen_norm

    scale = min(ref_w / gen_w, ref_h / gen_h)
    scaled_w = max(1, int(round(gen_w * scale)))
    scaled_h = max(1, int(round(gen_h * scale)))

    resized = gen_pil.resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (ref_w, ref_h), color=pad_color)

    offset_x = (ref_w - scaled_w) // 2
    offset_y = (ref_h - scaled_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    gen_norm = np.asarray(canvas, dtype=np.float32) / 255.0
    return ref_norm, gen_norm


def save_normalized_pair(ref_norm, gen_norm, basename, out_dir):
    """Speichere normalisierte Bilder mit eindeutigen Dateinamen."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir_path / f"{basename}_ref_norm.png"
    gen_path = out_dir_path / f"{basename}_gen_norm.png"

    np_to_pil_uint8(ref_norm).save(ref_path)
    np_to_pil_uint8(gen_norm).save(gen_path)

    return str(ref_path), str(gen_path)


def compute_ssim(ref, gen):
    """Berechne SSIM auf normalisierten RGB-Bildern."""
    try:
        return float(structural_similarity(ref, gen, data_range=1.0, channel_axis=-1))
    except TypeError:
        return float(structural_similarity(ref, gen, data_range=1.0, multichannel=True))


def init_lpips_model(net="alex", use_gpu=False):
    """Initialisiere LPIPS-Modell."""
    model = lpips.LPIPS(net=net)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def numpy_to_lpips_tensor(img):
    """Konvertiere HWC [0,1] nach NCHW [-1,1] für LPIPS."""
    chw = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).float()
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def compute_lpips(ref, gen, lpips_model, use_gpu=False):
    """Berechne LPIPS auf normalisierten Bildern."""
    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    return float(dist.item())


def create_foreground_mask(img, min_coverage=0.01):
    """Erzeuge eine robuste Vordergrundmaske für geometrische Checks."""
    gray = color.rgb2gray(img)
    otsu = threshold_otsu(gray)

    mask_dark = gray <= otsu
    mask_light = gray > otsu

    coverage_dark = float(np.mean(mask_dark))
    coverage_light = float(np.mean(mask_light))

    if coverage_dark < min_coverage and coverage_light < min_coverage:
        fallback = gray <= np.mean(gray)
        return fallback.astype(bool)

    if coverage_dark <= coverage_light:
        return mask_dark.astype(bool)
    return mask_light.astype(bool)


def safe_centroid(mask):
    """Berechne den Schwerpunkt einer Bool-Maske robust."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return np.array([h / 2.0, w / 2.0], dtype=np.float32)
    return np.array([float(np.mean(ys)), float(np.mean(xs))], dtype=np.float32)


def compute_geometric_metrics(ref, gen):
    """Berechne geometrische Metriken auf normalisierten Bildern."""
    ref_mask = create_foreground_mask(ref)
    gen_mask = create_foreground_mask(gen)

    ref_area = int(np.sum(ref_mask))
    gen_area = int(np.sum(gen_mask))
    intersection = int(np.sum(ref_mask & gen_mask))
    union = int(np.sum(ref_mask | gen_mask))

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2 * intersection) / (ref_area + gen_area)) if (ref_area + gen_area) > 0 else 0.0
    area_ratio = float(gen_area / ref_area) if ref_area > 0 else 0.0

    ref_center = safe_centroid(ref_mask)
    gen_center = safe_centroid(gen_mask)

    h, w = ref_mask.shape
    diag = float(np.hypot(h, w)) if (h > 0 and w > 0) else 1.0
    centroid_distance_px = float(np.linalg.norm(ref_center - gen_center))
    centroid_distance_norm = float(centroid_distance_px / diag)

    hausdorff_px = float(hausdorff_distance(ref_mask, gen_mask))
    hausdorff_norm = float(hausdorff_px / diag)

    return {
        "mask_iou": iou,
        "mask_dice": dice,
        "mask_area_ratio": area_ratio,
        "centroid_distance_px": centroid_distance_px,
        "centroid_distance_norm": centroid_distance_norm,
        "hausdorff_px": hausdorff_px,
        "hausdorff_norm": hausdorff_norm,
    }


def evaluate_pair(ref_path, gen_path, lpips_model, mode="letterbox", out_dir="normalized", use_gpu=False):
    """Evaluiere ein einzelnes Bildpaar inklusive Normalisierung und Export."""
    ref_img = load_image(ref_path)
    gen_img = load_image(gen_path)

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm, gen_norm = normalize_pair(ref_img, gen_img, mode=mode)
    norm_h, norm_w = ref_norm.shape[:2]

    basename = Path(ref_path).stem
    ref_norm_path, gen_norm_path = save_normalized_pair(ref_norm, gen_norm, basename, out_dir)

    ssim_val = compute_ssim(ref_norm, gen_norm)
    lpips_val = compute_lpips(ref_norm, gen_norm, lpips_model, use_gpu=use_gpu)
    geometric = compute_geometric_metrics(ref_norm, gen_norm)

    print("------------------------------------------------------------")
    print(f"Pair: {basename}")
    print(f"  Reference original : {ref_w}x{ref_h}")
    print(f"  Generated original : {gen_w}x{gen_h}")
    print(f"  Normalized sizes   : {norm_w}x{norm_h}")
    print(f"  SSIM               : {ssim_val:.6f}")
    print(f"  LPIPS              : {lpips_val:.6f}")
    print(f"  Saved ref_norm     : {ref_norm_path}")
    print(f"  Saved gen_norm     : {gen_norm_path}")

    return {
        "filename": Path(ref_path).name,
        "reference_width": ref_w,
        "reference_height": ref_h,
        "generated_width": gen_w,
        "generated_height": gen_h,
        "normalized_width": norm_w,
        "normalized_height": norm_h,
        "normalization_mode": mode,
        "ssim": ssim_val,
        "lpips": lpips_val,
        **geometric,
        "ref_norm_path": ref_norm_path,
        "gen_norm_path": gen_norm_path,
    }


def evaluate_folders(reference_dir, generated_dir, output_csv, lpips_model, mode="letterbox", out_dir="normalized", use_gpu=False):
    """Evaluiere alle passenden Bildpaare zweier Ordner."""
    ref_dir = Path(reference_dir)
    gen_dir = Path(generated_dir)

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference-Ordner existiert nicht: {ref_dir}")
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated-Ordner existiert nicht: {gen_dir}")

    ref_files = sorted([p for p in ref_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS])
    if not ref_files:
        raise RuntimeError(f"Keine Bilder im Reference-Ordner gefunden: {ref_dir}")

    results = []
    for ref_path in tqdm(ref_files, desc="Berechne Metriken"):
        gen_path = gen_dir / ref_path.name
        if not gen_path.exists():
            print(f"[WARNUNG] Überspringe {ref_path.name}: kein passendes Generated-Bild gefunden.")
            continue

        result = evaluate_pair(
            ref_path=str(ref_path),
            gen_path=str(gen_path),
            lpips_model=lpips_model,
            mode=mode,
            out_dir=out_dir,
            use_gpu=use_gpu,
        )
        results.append(result)

    if not results:
        raise RuntimeError("Keine auswertbaren Bildpaare gefunden.")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, float_format="%.6f")

    print("============================================================")
    print(f"[INFO] Ergebnisse gespeichert: {output_csv}")
    print(df[["filename", "ssim", "lpips", "mask_iou", "mask_dice"]].head())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleiche Referenz- und generierte Bilder mit automatischer Größen-Normalisierung.",
        epilog="Falls skimage fehlt: pip install scikit-image",
    )

    parser.add_argument("--mode", choices=["letterbox", "stretch"], default="letterbox", help="Normalisierungsmodus")
    parser.add_argument("--out", default="normalized", help="Output-Ordner für normalisierte Bilder")
    parser.add_argument("--output-csv", default="image_metrics_results.csv", help="CSV-Datei für Metrikergebnisse")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="Backbone für LPIPS")
    parser.add_argument("--use-gpu", action="store_true", help="Nutze CUDA, falls verfügbar")

    parser.add_argument("--ref", help="Pfad zu einem einzelnen Referenzbild")
    parser.add_argument("--gen", help="Pfad zu einem einzelnen Generated-Bild")
    parser.add_argument("--reference-dir", default="reference", help="Ordner mit Referenzbildern")
    parser.add_argument("--generated-dir", default="generated", help="Ordner mit Generated-Bildern")

    return parser.parse_args()


def main():
    args = parse_args()

    print("============================================================")
    print("[INFO] Starte Bildmetrik-Berechnung")
    print(f"[INFO] Mode            : {args.mode}")
    print(f"[INFO] Normalized out  : {args.out}")
    print(f"[INFO] Output CSV      : {args.output_csv}")
    print("============================================================")

    lpips_model = init_lpips_model(net=args.lpips_net, use_gpu=args.use_gpu)

    if args.ref or args.gen:
        if not (args.ref and args.gen):
            raise ValueError("Setze für Einzelvergleich beide Parameter: --ref und --gen")

        result = evaluate_pair(
            ref_path=args.ref,
            gen_path=args.gen,
            lpips_model=lpips_model,
            mode=args.mode,
            out_dir=args.out,
            use_gpu=args.use_gpu,
        )
        pd.DataFrame([result]).to_csv(args.output_csv, index=False, float_format="%.6f")
        print(f"[INFO] Einzelvergleich gespeichert: {args.output_csv}")
        return

    evaluate_folders(
        reference_dir=args.reference_dir,
        generated_dir=args.generated_dir,
        output_csv=args.output_csv,
        lpips_model=lpips_model,
        mode=args.mode,
        out_dir=args.out,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
