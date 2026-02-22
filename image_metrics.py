import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import torch
import lpips

from skimage.metrics import structural_similarity
from skimage import color


def load_image(path):
    """Bild als RGB-Array (float32, [0,1]) laden."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_to_match(img, target_shape):
    """img (np.array) auf target_shape (H,W) skalieren."""
    h, w = target_shape[:2]
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_img = pil_img.resize((w, h), Image.BICUBIC)
    return np.asarray(pil_img, dtype=np.float32) / 255.0


def compute_ssim(ref, gen):
    """SSIM für zwei RGB-Bilder berechnen."""
    try:
        ssim_val = structural_similarity(
            ref, gen, data_range=1.0, channel_axis=-1
        )
    except TypeError:
        ssim_val = structural_similarity(
            ref, gen, data_range=1.0, multichannel=True
        )
    return float(ssim_val)


def compute_delta_e(ref, gen):
    """Durchschnittliche ΔE00 (CIEDE2000) über alle Pixel."""
    ref_lab = color.rgb2lab(ref)
    gen_lab = color.rgb2lab(gen)
    delta_e = color.deltaE_ciede2000(ref_lab, gen_lab)
    return float(np.mean(delta_e))


def init_lpips_model(net='alex', use_gpu=False):
    print(f"[INFO] Initialisiere LPIPS-Modell ({net}), GPU: {use_gpu}")
    model = lpips.LPIPS(net=net)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def numpy_to_lpips_tensor(img):
    img_t = torch.from_numpy(img.transpose(2, 0, 1)).float() * 2.0 - 1.0
    return img_t.unsqueeze(0)


def compute_lpips(ref, gen, lpips_model, use_gpu=False):
    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    return float(dist.item())


def evaluate_folder_pairs(
    reference_dir="reference",
    generated_dir="generated",
    output_csv="image_metrics_results.csv",
    use_gpu=False,
    lpips_net="alex"
):
    print("====================================")
    print("[INFO] Starte Bildmetrik-Berechnung")
    print("Arbeitsverzeichnis:", os.getcwd())
    print("Reference-Ordner :", os.path.abspath(reference_dir))
    print("Generated-Ordner :", os.path.abspath(generated_dir))
    print("Output-Datei     :", os.path.abspath(output_csv))
    print("====================================")

    if not os.path.isdir(reference_dir):
        print(f"[FEHLER] Reference-Ordner '{reference_dir}' existiert nicht!")
        return

    if not os.path.isdir(generated_dir):
        print(f"[FEHLER] Generated-Ordner '{generated_dir}' existiert nicht!")
        return

    ref_files = sorted([
        f for f in os.listdir(reference_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))
    ])

    print(f"[INFO] Gefundene Bilder in '{reference_dir}': {len(ref_files)}")
    if ref_files:
        print("  Beispiele:", ref_files[:5])
    else:
        print("[WARNUNG] Keine Bilder im Reference-Ordner gefunden. Breche ab.")
        return

    lpips_model = init_lpips_model(net=lpips_net, use_gpu=use_gpu)

    results = []

    for filename in tqdm(ref_files, desc="Berechne Metriken"):
        ref_path = os.path.join(reference_dir, filename)
        gen_path = os.path.join(generated_dir, filename)

        if not os.path.exists(gen_path):
            print(f"[WARNUNG] Kein passendes generiertes Bild für {filename} gefunden, überspringe.")
            continue

        ref_img = load_image(ref_path)
        gen_img = load_image(gen_path)

        if ref_img.shape != gen_img.shape:
            print(f"[INFO] Größe unterscheidet sich für {filename}, skaliere generiertes Bild.")
            gen_img = resize_to_match(gen_img, ref_img.shape)

        ssim_val = compute_ssim(ref_img, gen_img)
        delta_e_val = compute_delta_e(ref_img, gen_img)
        lpips_val = compute_lpips(ref_img, gen_img, lpips_model, use_gpu=use_gpu)

        results.append({
            "filename": filename,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "deltaE00_mean": delta_e_val,
        })

    if not results:
        print("[WARNUNG] Keine Ergebnisse – vermutlich keine passenden Paare gefunden.")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, float_format="%.6f")

    print("\n[INFO] Fertig! Ergebnisse gespeichert in:", output_csv)
    print("\n[INFO] Erste Zeilen der Tabelle:")
    print(df.head())


if __name__ == "__main__":
    REF_DIR = "reference"
    GEN_DIR = "generated"
    OUTPUT_CSV = "image_metrics_results.csv"

    evaluate_folder_pairs(
        reference_dir=REF_DIR,
        generated_dir=GEN_DIR,
        output_csv=OUTPUT_CSV,
        use_gpu=False,
        lpips_net="alex",
    )
