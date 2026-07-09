import argparse
import errno
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.filters import threshold_otsu
from skimage.metrics import hausdorff_distance, structural_similarity
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, disk, remove_small_holes, remove_small_objects
from tqdm import tqdm

TORCH_IMPORT_ERROR = None

try:
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None
    TORCH_IMPORT_ERROR = str(exc)

try:
    import lpips
except Exception:  # noqa: BLE001
    lpips = None

try:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
except Exception:
    MaskRCNN_ResNet50_FPN_V2_Weights = None
    maskrcnn_resnet50_fpn_v2 = None

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
LETTERBOX_PAD_COLOR = (127, 127, 127)
COCO_VEHICLE_CLASSES = {3, 4, 6, 8}
CSV_COLUMN_ORDER = [
    "filename",
    "reference_width",
    "reference_height",
    "generated_width",
    "generated_height",
    "normalized_width",
    "normalized_height",
    "metric_scale_factor",
    "normalization_mode",
    "main_metric_scope",
    "content_mask_area_px",
    "content_mask_area_ratio",
    "ssim",
    "ssim_percent",
    "lpips",
    "lpips_similarity_percent",
    "lpips_map_mean",
    "lpips_foreground",
    "lpips_foreground_similarity_percent",
    "delta_e_ciede2000",
    "delta_e_similarity_percent",
    "lpips_car_only",
    "lpips_car_only_similarity_percent",
    "ssim_car_only",
    "mask_metric_scope",
    "mask_iou",
    "mask_dice",
    "mask_area_ratio",
    "centroid_distance_px",
    "centroid_distance_norm",
    "hausdorff_px",
    "hausdorff_norm",
    "car_mask_area_ratio",
    "car_bbox",
    "car_fallback_reason",
    "ref_norm_path",
    "gen_norm_path",
    "car_only_ref_path",
    "car_only_gen_path",
    "lpips_spatial_path",
]


IMPORTANT_RESULT_COLUMNS = [
    "filename",
    "ssim_percent",
    "lpips_similarity_percent",
    "delta_e_similarity_percent",
    "lpips_car_only_similarity_percent",
    "ssim_car_only",
    "mask_iou",
    "mask_dice",
    "mask_metric_scope",
    "content_mask_area_ratio",
    "ref_norm_path",
    "gen_norm_path",
    "car_only_ref_path",
    "car_only_gen_path",
    "lpips_spatial_path",
]

SUMMARY_COLUMN_ORDER = [
    "filename",
    "lpips",
    "lpips_similarity_percent",
    "delta_e_ciede2000",
    "delta_e_similarity_percent",
    "lpips_car_only",
    "lpips_car_only_similarity_percent",
    "mask_iou",
    "mask_metric_scope",
    "ref_norm_path",
    "gen_norm_path",
    "car_only_ref_path",
    "car_only_gen_path",
    "lpips_spatial_path",
]

DISTANCE_COLUMNS = {
    "ssim",
    "lpips",
    "lpips_map_mean",
    "lpips_foreground",
    "delta_e_ciede2000",
    "lpips_car_only",
    "ssim_car_only",
    "mask_iou",
    "mask_dice",
    "mask_area_ratio",
    "centroid_distance_px",
    "centroid_distance_norm",
    "hausdorff_px",
    "hausdorff_norm",
    "car_mask_area_ratio",
    "content_mask_area_ratio",
    "metric_scale_factor",
}

PERCENT_COLUMNS = {
    "ssim_percent",
    "lpips_similarity_percent",
    "lpips_foreground_similarity_percent",
    "delta_e_similarity_percent",
    "lpips_car_only_similarity_percent",
}

EXCEL_COLUMN_LABELS = {
    "filename": "Dateiname",
    "reference_width": "Referenz Breite px",
    "reference_height": "Referenz Höhe px",
    "generated_width": "Generated Breite px",
    "generated_height": "Generated Höhe px",
    "normalized_width": "Normalisiert Breite px",
    "normalized_height": "Normalisiert Höhe px",
    "metric_scale_factor": "Metrik Skalierungsfaktor",
    "normalization_mode": "Normalisierung",
    "main_metric_scope": "Hauptmetrik Bereich",
    "content_mask_area_px": "Content-Maske Fläche px",
    "content_mask_area_ratio": "Content-Maske Anteil",
    "ssim": "SSIM Distanz/Rohwert",
    "ssim_percent": "SSIM Ähnlichkeit (%)",
    "lpips": "LPIPS Distanz",
    "lpips_similarity_percent": "LPIPS Ähnlichkeit (%)",
    "lpips_map_mean": "LPIPS Spatial Mittelwert",
    "lpips_foreground": "LPIPS Vordergrund",
    "lpips_foreground_similarity_percent": "LPIPS Vordergrund Ähnlichkeit (%)",
    "delta_e_ciede2000": "Delta E CIEDE2000",
    "delta_e_similarity_percent": "Delta E Ähnlichkeit (%)",
    "lpips_car_only": "LPIPS Fahrzeugmodus Distanz",
    "lpips_car_only_similarity_percent": "LPIPS Fahrzeugmodus Ähnlichkeit (%)",
    "ssim_car_only": "SSIM Fahrzeugmodus",
    "mask_metric_scope": "Maskenmetrik Bereich",
    "mask_iou": "Mask IoU",
    "mask_dice": "Mask Dice",
    "mask_area_ratio": "Maskenflächen Verhältnis",
    "centroid_distance_px": "Schwerpunktdistanz px",
    "centroid_distance_norm": "Schwerpunktdistanz normiert",
    "hausdorff_px": "Hausdorff Distanz px",
    "hausdorff_norm": "Hausdorff Distanz normiert",
    "car_mask_area_ratio": "Fahrzeugmaske Anteil",
    "car_bbox": "Fahrzeug Bounding Box",
    "car_fallback_reason": "Fahrzeug Fallback Grund",
    "ref_norm_path": "Pfad Referenz normalisiert",
    "gen_norm_path": "Pfad Generated normalisiert",
    "car_only_ref_path": "Pfad Fahrzeugmodus Referenz",
    "car_only_gen_path": "Pfad Fahrzeugmodus Generated",
    "lpips_spatial_path": "Pfad Abweichungsmatrix",
}

CAR_ONLY_EXPORT_COLUMNS = {
    "lpips_car_only",
    "lpips_car_only_similarity_percent",
    "ssim_car_only",
    "car_mask_area_ratio",
    "car_bbox",
    "car_fallback_reason",
    "car_only_ref_path",
    "car_only_gen_path",
}

CSV_FLOAT_COLUMNS = [
    "content_mask_area_ratio",
    "ssim",
    "ssim_percent",
    "lpips",
    "lpips_similarity_percent",
    "lpips_map_mean",
    "lpips_foreground",
    "lpips_foreground_similarity_percent",
    "delta_e_ciede2000",
    "delta_e_similarity_percent",
    "lpips_car_only",
    "lpips_car_only_similarity_percent",
    "ssim_car_only",
    "mask_iou",
    "mask_dice",
    "mask_area_ratio",
    "centroid_distance_px",
    "centroid_distance_norm",
    "hausdorff_px",
    "hausdorff_norm",
    "car_mask_area_ratio",
    "metric_scale_factor",
]


def build_result_dataframe(results):
    if not results:
        raise ValueError("results darf nicht leer sein.")

    df = pd.DataFrame(results)

    for column in CSV_COLUMN_ORDER:
        if column not in df.columns:
            df[column] = None

    df = df.loc[:, CSV_COLUMN_ORDER]

    for column in CSV_FLOAT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def is_no_space_error(exc):
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return True
    message = str(exc).lower()
    return "no space left on device" in message or "errno 28" in message


def get_no_space_message():
    return (
        "Nicht genügend Speicherplatz vorhanden. "
        "Bitte alte Analyseordner im Ordner runs löschen oder Speicherplatz freigeben."
    )


def build_excel_output_paths(output_csv):
    output_path = Path(output_csv)
    stem = output_path.stem
    parent = output_path.parent if output_path.parent != Path("") else Path(".")
    full_xlsx = parent / f"{stem}.xlsx"
    summary_stem = stem.replace("_results", "_summary")
    if summary_stem == stem:
        summary_stem = f"{stem}_summary"
    summary_xlsx = parent / f"{summary_stem}.xlsx"
    return full_xlsx, summary_xlsx


def get_lpips_net_label(lpips_net):
    if lpips_net == "alex":
        return "alex (empfohlen)"
    return str(lpips_net)


def prepare_export_dataframe(df, include_car_only=True, summary=False):
    if summary:
        columns = [column for column in SUMMARY_COLUMN_ORDER if column in df.columns]
    else:
        priority_columns = [column for column in IMPORTANT_RESULT_COLUMNS if column in df.columns]
        remaining_columns = [column for column in df.columns if column not in priority_columns]
        columns = priority_columns + remaining_columns

    if not include_car_only:
        columns = [column for column in columns if column not in CAR_ONLY_EXPORT_COLUMNS]

    export_df = df.loc[:, columns].copy()
    for column in export_df.columns:
        if column in PERCENT_COLUMNS:
            export_df[column] = pd.to_numeric(export_df[column], errors="coerce").round(2)
        elif column in DISTANCE_COLUMNS:
            export_df[column] = pd.to_numeric(export_df[column], errors="coerce").round(4)

    return export_df


def rename_columns_for_excel(df):
    return df.rename(columns={column: EXCEL_COLUMN_LABELS.get(column, column) for column in df.columns})


def format_excel_sheet(writer, sheet_name, dataframe):
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    header_fill = PatternFill(fill_type="solid", fgColor="D9EAF7")
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")

    worksheet.row_dimensions[1].height = 32

    for index, column_name in enumerate(dataframe.columns, start=1):
        column_letter = get_column_letter(index)
        series = dataframe[column_name].fillna("")
        max_content_width = series.map(lambda value: len(str(value))).max() if not series.empty else 0
        header_width = len(str(column_name))
        width = max(max_content_width, header_width) + 2

        if "Pfad" in str(column_name) or "path" in str(column_name).lower():
            width = min(max(width, 35), 80)
        else:
            width = min(max(width, 12), 35)

        worksheet.column_dimensions[column_letter].width = width

        for cell in worksheet[column_letter]:
            if cell.row == 1:
                continue

            if isinstance(cell.value, (int, float)):
                column_label = str(column_name).lower()
                if "percent" in column_label or "ähnlichkeit" in column_label or "(%)" in column_label:
                    cell.number_format = "0.00"
                else:
                    cell.number_format = "0.0000"


def write_excel_workbook(path, df, sheet_name, include_car_only=True, lpips_net="alex", summary=False):
    export_df = prepare_export_dataframe(df, include_car_only=include_car_only, summary=summary)
    display_df = rename_columns_for_excel(export_df)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        display_df.to_excel(writer, sheet_name=sheet_name, index=False)
        format_excel_sheet(writer, sheet_name, display_df)

        metadata = pd.DataFrame(
            [
                {"Eigenschaft": "LPIPS CNN", "Wert": get_lpips_net_label(lpips_net)},
                {"Eigenschaft": "Fahrzeugmodus", "Wert": "aktiv" if include_car_only else "deaktiviert"},
            ]
        )
        metadata.to_excel(writer, sheet_name="Info", index=False)
        format_excel_sheet(writer, "Info", metadata)


def write_result_files(df, output_csv, include_car_only=True, lpips_net="alex"):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_df = prepare_export_dataframe(df, include_car_only=include_car_only, summary=True)
    summary_df.to_csv(
        output_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        float_format="%.6f",
        na_rep="",
    )

    full_xlsx, _summary_xlsx = build_excel_output_paths(output_path)
    output_paths = {"csv": str(output_path)}
    try:
        write_excel_workbook(full_xlsx, df, "Ergebnisse", include_car_only=include_car_only, lpips_net=lpips_net, summary=True)
        output_paths["xlsx"] = str(full_xlsx)
    except ModuleNotFoundError as exc:
        if exc.name != "openpyxl":
            raise
        print("[WARN] Excel-Ausgabe übersprungen: Installiere openpyxl für formatierte Ergebnistabellen.")

    return output_paths


def load_image(path):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr


def validate_image_for_metrics(img, image_name="image"):
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{image_name} muss ein NumPy-Array sein.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"{image_name} muss die Form (H, W, 3) haben. Aktuell: {img.shape}")

    if not np.issubdtype(img.dtype, np.floating):
        raise TypeError(f"{image_name} muss ein Float-Tensor/Array sein. Aktuell: {img.dtype}")

    min_val = float(np.min(img))
    max_val = float(np.max(img))
    if min_val < 0.0 or max_val > 1.0:
        raise ValueError(
            f"{image_name} enthält Werte außerhalb [0,1] (min={min_val:.4f}, max={max_val:.4f}). "
            "Nutze load_image()/Normierung vor der Metrik-Berechnung."
        )


def np_to_pil_uint8(img):
    clipped = np.clip(img, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")


def letterbox_to_canvas(img, target_w, target_h, pad_color=LETTERBOX_PAD_COLOR):
    src_h, src_w = img.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Bildgröße muss größer als 0 sein.")
    if target_w <= 0 or target_h <= 0:
        raise ValueError("Canvas-Größe muss größer als 0 sein.")

    scale = min(target_w / src_w, target_h / src_h)
    scaled_w = max(1, int(round(src_w * scale)))
    scaled_h = max(1, int(round(src_h * scale)))
    offset_x = (target_w - scaled_w) // 2
    offset_y = (target_h - scaled_h) // 2

    resized = np_to_pil_uint8(img).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
    canvas.paste(resized, (offset_x, offset_y))

    norm_img = np.asarray(canvas, dtype=np.float32) / 255.0
    content_mask = np.zeros((target_h, target_w), dtype=bool)
    content_mask[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = True
    debug = {
        "source_width": int(src_w),
        "source_height": int(src_h),
        "scale": float(scale),
        "scaled_width": int(scaled_w),
        "scaled_height": int(scaled_h),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
    }
    return norm_img, content_mask, debug


def normalize_pair(ref_img, gen_img, mode="letterbox", pad_color=LETTERBOX_PAD_COLOR, max_metric_long_edge=1600):
    ref_norm, gen_norm, content_mask, _ = normalize_pair_for_metrics(
        ref_img,
        gen_img,
        mode=mode,
        pad_color=pad_color,
        max_metric_long_edge=max_metric_long_edge,
    )
    return ref_norm, gen_norm, content_mask


def normalize_pair_for_metrics(ref_img, gen_img, mode="letterbox", pad_color=LETTERBOX_PAD_COLOR, max_metric_long_edge=1600):
    if mode != "letterbox":
        raise ValueError("mode muss 'letterbox' sein")

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]
    if ref_w <= 0 or ref_h <= 0 or gen_w <= 0 or gen_h <= 0:
        raise ValueError("Bildgrößen müssen größer als 0 sein.")

    if max_metric_long_edge is None:
        # None deaktiviert den Performance-Schutz bewusst: Die Metriken laufen dann in voller
        # Referenzauflösung. Das kann bei sehr großen Bildern weiterhin viel RAM benötigen.
        metric_scale = 1.0
        target_w = int(ref_w)
        target_h = int(ref_h)
    else:
        limit = int(max_metric_long_edge)
        if limit <= 0:
            raise ValueError("max_metric_long_edge muss > 0 sein oder None.")

        current_long_edge = max(ref_h, ref_w)
        if current_long_edge > limit:
            metric_scale = float(limit / current_long_edge)
            target_w = max(1, int(round(ref_w * metric_scale)))
            target_h = max(1, int(round(ref_h * metric_scale)))
        else:
            metric_scale = 1.0
            target_w = int(ref_w)
            target_h = int(ref_h)

    if target_w == ref_w and target_h == ref_h:
        ref_norm = ref_img
    else:
        ref_norm = np.asarray(
            np_to_pil_uint8(ref_img).resize((target_w, target_h), resample=Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0

    gen_scale = min(target_w / gen_w, target_h / gen_h)
    scaled_w = max(1, int(round(gen_w * gen_scale)))
    scaled_h = max(1, int(round(gen_h * gen_scale)))
    offset_x = (target_w - scaled_w) // 2
    offset_y = (target_h - scaled_h) // 2

    resized_gen = np_to_pil_uint8(gen_img).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), color=pad_color)
    canvas.paste(resized_gen, (offset_x, offset_y))

    gen_norm = np.asarray(canvas, dtype=np.float32) / 255.0
    content_mask = np.zeros((target_h, target_w), dtype=bool)
    content_mask[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = True

    return ref_norm, gen_norm, content_mask, metric_scale


def downscale_pair_for_metrics(ref_img, gen_img, content_mask=None, max_long_edge_px=1600):
    if max_long_edge_px is None:
        return ref_img, gen_img, content_mask, 1.0

    limit = int(max_long_edge_px)
    if limit <= 0:
        raise ValueError("max_long_edge_px muss > 0 sein.")

    h, w = ref_img.shape[:2]
    current_long_edge = max(h, w)
    if current_long_edge <= limit:
        return ref_img, gen_img, content_mask, 1.0

    scale = float(limit / current_long_edge)
    target_w = max(1, int(round(w * scale)))
    target_h = max(1, int(round(h * scale)))

    ref_resized = np.asarray(
        np_to_pil_uint8(ref_img).resize((target_w, target_h), resample=Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    gen_resized = np.asarray(
        np_to_pil_uint8(gen_img).resize((target_w, target_h), resample=Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0

    resized_mask = None
    if content_mask is not None:
        mask_img = Image.fromarray(np.asarray(content_mask, dtype=np.uint8) * 255, mode="L")
        resized_mask = np.asarray(
            mask_img.resize((target_w, target_h), resample=Image.Resampling.NEAREST),
            dtype=np.uint8,
        ).astype(bool)

    return ref_resized, gen_resized, resized_mask, scale


def save_normalized_pair(ref_norm, gen_norm, basename, out_dir):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir_path / f"{basename}_ref_norm.png"
    gen_path = out_dir_path / f"{basename}_gen_norm.png"

    np_to_pil_uint8(ref_norm).save(ref_path)
    np_to_pil_uint8(gen_norm).save(gen_path)

    return str(ref_path), str(gen_path)


def compute_ssim(ref, gen):
    try:
        value = structural_similarity(ref, gen, data_range=1.0, channel_axis=-1)
    except TypeError:
        value = structural_similarity(ref, gen, data_range=1.0, multichannel=True)
    return float(value)


def compute_masked_ssim(ref, gen, mask, neutral_value=0.5):
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        return compute_ssim(ref, gen)

    masked_ref = ref.copy()
    masked_gen = gen.copy()
    masked_ref[~metric_mask] = neutral_value
    masked_gen[~metric_mask] = neutral_value
    return compute_ssim(masked_ref, masked_gen)


def prepare_metric_mask(mask, ref, gen):
    if mask is None:
        return None

    expected_shape = ref.shape[:2]
    if gen.shape[:2] != expected_shape:
        raise ValueError(
            f"Maskenvalidierung fehlgeschlagen: ref/gen haben unterschiedliche Formen "
            f"({expected_shape} vs. {gen.shape[:2]})."
        )

    metric_mask = np.asarray(mask, dtype=bool)
    if metric_mask.shape != expected_shape:
        raise ValueError(
            f"Maskenvalidierung fehlgeschlagen: erwartete Form {expected_shape}, "
            f"erhalten {metric_mask.shape}."
        )

    if not np.any(metric_mask):
        return None

    return metric_mask


def init_lpips_model(net="alex", use_gpu=False):
    if lpips is None:
        raise RuntimeError("Paket 'lpips' nicht gefunden. Installiere die Abhängigkeiten aus requirements.txt.")

    if torch is None:
        raise RuntimeError(f"'torch' konnte nicht geladen werden ({TORCH_IMPORT_ERROR}). Installiere torch korrekt.")

    # Nutze immer den offiziellen LPIPS-Inferenzpfad (lin):
    # - lpips=True aktiviert die trainierten linearen Kalibrierungsschichten.
    # - pretrained=True lädt die vortrainierten Gewichte.
    # - spatial=True liefert zusätzlich eine räumliche Distanzkarte (Heatmap).
    # Dieses Tool trainiert keine LPIPS-Gewichte nach.
    model = lpips.LPIPS(net=net, spatial=True, lpips=True, pretrained=True)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def verify_lpips_forward(lpips_model, net="alex", use_gpu=False):
    if torch is None:
        raise RuntimeError(f"'torch' ist nicht verfügbar: {TORCH_IMPORT_ERROR}")

    dummy_ref = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    dummy_gen = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    dummy_ref = dummy_ref * 2.0 - 1.0
    dummy_gen = dummy_gen * 2.0 - 1.0

    if use_gpu and torch.cuda.is_available():
        dummy_ref = dummy_ref.cuda()
        dummy_gen = dummy_gen.cuda()

    with torch.no_grad():
        out = lpips_model(dummy_ref, dummy_gen)

    if out.ndim < 2:
        raise RuntimeError(f"LPIPS-Forward für net='{net}' liefert unerwartete Form: {tuple(out.shape)}")
def configure_determinism(seed=None, deterministic=False):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    if deterministic and torch is not None:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def run_lpips_pipeline_sanity_checks():
    import ast

    module_path = Path(__file__).resolve()
    module_text = module_path.read_text(encoding="utf-8")
    tree = ast.parse(module_text)

    if not any(isinstance(node, ast.FunctionDef) and node.name == "compute_lpips" for node in ast.walk(tree)):
        raise RuntimeError("Sanity-Check fehlgeschlagen: compute_lpips-Funktion nicht gefunden.")


def numpy_to_lpips_tensor(img):
    if torch is None:
        raise RuntimeError(f"'torch' ist nicht verfügbar: {TORCH_IMPORT_ERROR}")

    chw = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).float()
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def compute_lpips(ref, gen, lpips_model, use_gpu=False):
    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    return float(torch.mean(dist).item())


def compute_lpips_with_map(ref, gen, lpips_model, use_gpu=False):
    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    dist_value = float(torch.mean(dist).item())
    dist_map = dist.detach().float().cpu().numpy().squeeze()
    if dist_map.ndim == 0:
        dist_map = np.array([[float(dist_map)]], dtype=np.float32)
    return dist_value, dist_map


def masked_lpips(ref, gen, mask, lpips_model, use_gpu=False, mask_downsample="bilinear", eps=1e-8):
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        return compute_lpips(ref, gen, lpips_model, use_gpu=use_gpu)

    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    dist_map = dist.detach().float().cpu().numpy().squeeze()
    if dist_map.ndim == 0:
        dist_map = np.array([[float(dist_map)]], dtype=np.float32)
    elif dist_map.ndim == 1:
        dist_map = dist_map[np.newaxis, :]

    if mask_downsample == "nearest":
        resample = Image.Resampling.NEAREST
    else:
        resample = Image.Resampling.BILINEAR

    mask_img = Image.fromarray(metric_mask.astype(np.uint8) * 255, mode="L")
    mask_resized = np.asarray(
        mask_img.resize((dist_map.shape[1], dist_map.shape[0]), resample=resample),
        dtype=np.float32,
    ) / 255.0

    weighted_sum = float(np.sum(dist_map * mask_resized))
    mask_sum = float(np.sum(mask_resized))
    if mask_sum <= float(eps):
        return float(np.mean(dist_map))
    return float(weighted_sum / (mask_sum + float(eps)))


def compute_lpips_on_content(ref, gen, content_mask, lpips_model, use_gpu=False, mask_downsample="bilinear", eps=1e-8):
    metric_mask = prepare_metric_mask(content_mask, ref, gen)
    if metric_mask is None:
        return compute_lpips(ref, gen, lpips_model, use_gpu=use_gpu)

    return masked_lpips(
        ref,
        gen,
        metric_mask,
        lpips_model,
        use_gpu=use_gpu,
        mask_downsample=mask_downsample,
        eps=eps,
    )


def compute_mask_roi_bbox(mask, fallback_shape):
    metric_mask = np.asarray(mask, dtype=bool) if mask is not None else None
    if metric_mask is None or not np.any(metric_mask):
        h, w = fallback_shape
        return 0, 0, w, h

    ys, xs = np.where(metric_mask)
    x0 = int(np.min(xs))
    x1 = int(np.max(xs)) + 1
    y0 = int(np.min(ys))
    y1 = int(np.max(ys)) + 1
    return x0, y0, x1, y1


def crop_image_to_bbox(img, bbox):
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1, :]


def crop_mask_to_bbox(mask, bbox):
    if mask is None:
        return None
    x0, y0, x1, y1 = bbox
    return np.asarray(mask, dtype=bool)[y0:y1, x0:x1]


def compute_lpips_on_roi(ref, gen, roi_bbox, lpips_model, use_gpu=False):
    ref_roi = crop_image_to_bbox(ref, roi_bbox)
    gen_roi = crop_image_to_bbox(gen, roi_bbox)
    return compute_lpips(ref_roi, gen_roi, lpips_model, use_gpu=use_gpu)


def resize_mask_to_spatial_map(mask, target_shape):
    if mask is None:
        return None

    spatial_mask = np.asarray(mask, dtype=bool)
    if spatial_mask.ndim != 2:
        return None

    target_rows, target_cols = target_shape
    if target_rows <= 0 or target_cols <= 0:
        return None

    if spatial_mask.shape == (target_rows, target_cols):
        return spatial_mask

    mask_image = Image.fromarray(spatial_mask.astype(np.uint8) * 255, mode="L")
    resized_mask = np.asarray(
        mask_image.resize((target_cols, target_rows), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    return resized_mask.astype(bool)


def save_lpips_spatial_map(
    dist_map,
    path,
    overlay_mask=None,
    mask_mode=None,
    outline_mask=None,
    outline_mode=None,
):
    dist_map = np.asarray(dist_map, dtype=np.float32)
    if dist_map.ndim == 1:
        dist_map = dist_map[np.newaxis, :]
    elif dist_map.ndim == 0:
        dist_map = dist_map.reshape(1, 1)
    payload = {
        "rows": int(dist_map.shape[0]),
        "cols": int(dist_map.shape[1]),
        "min": float(np.min(dist_map)),
        "max": float(np.max(dist_map)),
        "values": dist_map.tolist(),
    }

    resized_overlay_mask = resize_mask_to_spatial_map(overlay_mask, dist_map.shape)
    if resized_overlay_mask is not None and np.any(resized_overlay_mask):
        payload["overlay_mask"] = resized_overlay_mask.astype(np.uint8).tolist()
        payload["mask_mode"] = str(mask_mode) if mask_mode else "overlay"

    resized_outline_mask = resize_mask_to_spatial_map(outline_mask, dist_map.shape)
    if resized_outline_mask is not None and np.any(resized_outline_mask):
        payload["outline_mask"] = resized_outline_mask.astype(np.uint8).tolist()
        payload["outline_mode"] = str(outline_mode) if outline_mode else "outline"

    path.write_text(json.dumps(payload), encoding="utf-8")


def build_vehicle_segmenter(use_gpu=False, score_threshold=0.5, mask_threshold=0.5):
    if torch is None:
        raise RuntimeError(f"torch ist nicht verfügbar ({TORCH_IMPORT_ERROR}). Deaktiviere --enable-car-only oder installiere torch korrekt.")

    if maskrcnn_resnet50_fpn_v2 is None:
        raise RuntimeError("torchvision MaskRCNN ist nicht verfügbar. Installiere torchvision >= 0.13.")

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    model.eval()
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    return {
        "model": model,
        "device": torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"),
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "cache": {},
    }


def segment_car_mask(image, segmenter, cache_key=None):
    if cache_key is not None and cache_key in segmenter["cache"]:
        return segmenter["cache"][cache_key].copy()

    img_t = torch.from_numpy(image.transpose(2, 0, 1)).float().to(segmenter["device"])
    with torch.no_grad():
        pred = segmenter["model"]([img_t])[0]

    labels = pred["labels"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    masks = pred["masks"].detach().cpu().numpy()

    valid = (scores >= segmenter["score_threshold"]) & np.isin(labels, list(COCO_VEHICLE_CLASSES))

    h, w = image.shape[:2]
    if np.any(valid):
        probs = masks[valid, 0, :, :]
        max_probs = np.max(probs, axis=0)
        combined = max_probs >= segmenter.get("mask_threshold", 0.5)
    else:
        combined = np.zeros((h, w), dtype=bool)

    if cache_key is not None:
        segmenter["cache"][cache_key] = combined.copy()
    return combined


def refine_car_mask(
    mask,
    ref_mask,
    gen_mask,
    grow_px=10,
    min_object_area=500,
    max_hole_area=3000,
    trim_px=1,
):
    merged_mask = (ref_mask | gen_mask).astype(bool)
    refined_mask = mask.astype(bool)

    if np.any(merged_mask):
        grow_disk = disk(max(1, int(grow_px)))
        grown_mask = binary_dilation(refined_mask, footprint=grow_disk)
        refined_mask = refined_mask | (merged_mask & grown_mask)

    refined_mask = binary_closing(refined_mask, footprint=disk(2))
    refined_mask = remove_small_objects(refined_mask, min_size=max(1, int(min_object_area)))
    refined_mask = remove_small_holes(refined_mask, area_threshold=max(1, int(max_hole_area)))

    if trim_px > 0:
        refined_mask = binary_erosion(refined_mask, footprint=disk(int(trim_px)))
        refined_mask = remove_small_objects(refined_mask, min_size=max(1, int(min_object_area)))

    return refined_mask.astype(bool)


def compute_mask_bbox(mask, pad_px=20, min_size_px=64, make_square=True):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    y0 = max(0, int(np.min(ys)) - pad_px)
    y1 = min(mask.shape[0], int(np.max(ys)) + 1 + pad_px)
    x0 = max(0, int(np.min(xs)) - pad_px)
    x1 = min(mask.shape[1], int(np.max(xs)) + 1 + pad_px)

    box_w = x1 - x0
    box_h = y1 - y0
    min_size = max(1, int(min_size_px))

    if make_square:
        target_size = max(box_w, box_h, min_size)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half = target_size / 2.0
        x0 = int(round(cx - half))
        x1 = int(round(cx + half))
        y0 = int(round(cy - half))
        y1 = int(round(cy + half))
    else:
        if box_w < min_size:
            add = min_size - box_w
            x0 -= add // 2
            x1 += add - (add // 2)
        if box_h < min_size:
            add = min_size - box_h
            y0 -= add // 2
            y1 += add - (add // 2)

    h, w = mask.shape
    if x0 < 0:
        x1 += -x0
        x0 = 0
    if y0 < 0:
        y1 += -y0
        y0 = 0
    if x1 > w:
        shift = x1 - w
        x0 = max(0, x0 - shift)
        x1 = w
    if y1 > h:
        shift = y1 - h
        y0 = max(0, y0 - shift)
        y1 = h

    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def apply_neutralize_crop(img, mask, bbox, neutral_value=0.5):
    x0, y0, x1, y1 = bbox
    img_crop = img[y0:y1, x0:x1, :].copy()
    mask_crop = mask[y0:y1, x0:x1]
    img_crop[~mask_crop] = neutral_value
    return img_crop, mask_crop


def apply_masked_car_crop(img, mask, bbox):
    x0, y0, x1, y1 = bbox
    img_crop = img[y0:y1, x0:x1, :].copy()
    mask_crop = mask[y0:y1, x0:x1]
    masked_crop = np.zeros_like(img_crop)
    masked_crop[mask_crop] = img_crop[mask_crop]
    return masked_crop, mask_crop


def save_mask_image(mask, path):
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def bbox_to_debug_dict(bbox, pad_px, min_size_px, square):
    return {
        "x0": bbox[0],
        "y0": bbox[1],
        "x1": bbox[2],
        "y1": bbox[3],
        "pad_px": int(pad_px),
        "min_size_px": int(min_size_px),
        "square": bool(square),
    }


def compute_car_only_metrics(
    ref_norm,
    gen_norm,
    ref_path,
    gen_path,
    lpips_model,
    segmenter,
    car_mode="neutralize_crop",
    mask_source="union",
    pad_px=20,
    neutral_value=0.5,
    min_mask_area=0,
    mask_downsample="bilinear",
    eps=1e-8,
    debug_dir=None,
    car_only_dir=None,
    use_gpu=False,
    mask_grow_px=10,
    mask_min_object_area=500,
    mask_max_hole_area=3000,
    mask_trim_px=1,
    roi_min_size_px=64,
    roi_square=True,
):
    empty_masks = {"ref_mask": None, "gen_mask": None, "car_mask": None}
    if segmenter is None:
        debug = {
            "mask_area_ratio": 0.0,
            "bbox": None,
            "metric_bbox": None,
            "ref_preview_bbox": None,
            "gen_preview_bbox": None,
            "fallback_reason": "Fahrzeugmodus deaktiviert",
        }
        return {
            "lpips_car_only": None,
            "ssim_car_only": None,
            "car_only_paths": {"ref": None, "gen": None},
            "debug": debug,
            "masks": empty_masks,
        }

    ref_mask = segment_car_mask(ref_norm, segmenter, cache_key=f"ref::{Path(ref_path).resolve()}")
    gen_mask = segment_car_mask(gen_norm, segmenter, cache_key=f"gen::{Path(gen_path).resolve()}")

    if mask_source == "ref":
        base_mask = ref_mask
    elif mask_source == "gen":
        base_mask = gen_mask
    else:
        base_mask = ref_mask | gen_mask

    mask = refine_car_mask(
        base_mask,
        ref_mask,
        gen_mask,
        grow_px=mask_grow_px,
        min_object_area=mask_min_object_area,
        max_hole_area=mask_max_hole_area,
        trim_px=mask_trim_px,
    )

    mask_area = int(np.sum(mask))
    total_area = int(mask.size)
    area_ratio = float(mask_area / total_area) if total_area > 0 else 0.0

    debug = {
        "mask_area_ratio": area_ratio,
        "bbox": None,
        "metric_bbox": None,
        "ref_preview_bbox": None,
        "gen_preview_bbox": None,
        "fallback_reason": None,
        "mask_refine": {
            "grow_px": mask_grow_px,
            "min_object_area": mask_min_object_area,
            "max_hole_area": mask_max_hole_area,
            "trim_px": mask_trim_px,
        },
    }
    if mask_area <= int(min_mask_area):
        debug["fallback_reason"] = f"Mask area zu klein ({mask_area} px)"
        return {
            "lpips_car_only": None,
            "ssim_car_only": None,
            "car_only_paths": {"ref": None, "gen": None},
            "debug": debug,
            "masks": {"ref_mask": ref_mask, "gen_mask": gen_mask, "car_mask": None},
        }

    adaptive_min_size = max(int(roi_min_size_px), int(min(ref_norm.shape[0], ref_norm.shape[1]) * 0.2))
    metric_bbox = compute_mask_bbox(mask, pad_px=pad_px, min_size_px=adaptive_min_size, make_square=roi_square)
    debug["metric_bbox"] = bbox_to_debug_dict(metric_bbox, pad_px, adaptive_min_size, roi_square)
    debug["bbox"] = debug["metric_bbox"]

    ref_preview_mask = ref_mask.astype(bool)
    gen_preview_mask = gen_mask.astype(bool)
    ref_preview_bbox = compute_mask_bbox(ref_preview_mask, pad_px=pad_px, min_size_px=adaptive_min_size, make_square=roi_square)
    gen_preview_bbox = compute_mask_bbox(gen_preview_mask, pad_px=pad_px, min_size_px=adaptive_min_size, make_square=roi_square)
    debug["ref_preview_bbox"] = bbox_to_debug_dict(ref_preview_bbox, pad_px, adaptive_min_size, roi_square)
    debug["gen_preview_bbox"] = bbox_to_debug_dict(gen_preview_bbox, pad_px, adaptive_min_size, roi_square)

    if car_mode == "neutralize_crop":
        ref_car, mask_crop = apply_neutralize_crop(ref_norm, mask, metric_bbox, neutral_value=neutral_value)
        gen_car, _ = apply_neutralize_crop(gen_norm, mask, metric_bbox, neutral_value=neutral_value)
        ref_preview, _ = apply_masked_car_crop(ref_norm, ref_preview_mask, ref_preview_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, gen_preview_mask, gen_preview_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    elif car_mode == "weighted_lpips":
        debug["fallback_reason"] = "car_mode=weighted_lpips ist deprecated und nutzt neutralize_crop."
        ref_car, mask_crop = apply_neutralize_crop(ref_norm, mask, metric_bbox, neutral_value=neutral_value)
        gen_car, _ = apply_neutralize_crop(gen_norm, mask, metric_bbox, neutral_value=neutral_value)
        ref_preview, _ = apply_masked_car_crop(ref_norm, ref_preview_mask, ref_preview_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, gen_preview_mask, gen_preview_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    elif car_mode == "roi_crop":
        x0, y0, x1, y1 = metric_bbox
        ref_car = ref_norm[y0:y1, x0:x1, :]
        gen_car = gen_norm[y0:y1, x0:x1, :]
        mask_crop = mask[y0:y1, x0:x1]
        ref_preview, _ = apply_masked_car_crop(ref_norm, ref_preview_mask, ref_preview_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, gen_preview_mask, gen_preview_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    else:
        raise ValueError("car_mode muss 'neutralize_crop', 'roi_crop' oder 'weighted_lpips' sein")

    stem = Path(ref_path).stem

    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        save_mask_image(mask, debug_path / f"{stem}_mask.png")
        with open(debug_path / f"{stem}_crop_box.json", "w", encoding="utf-8") as fp:
            json.dump(debug["bbox"], fp, indent=2)
        np_to_pil_uint8(ref_car).save(debug_path / f"{stem}_ref_neutral.png")
        np_to_pil_uint8(gen_car).save(debug_path / f"{stem}_gen_neutral.png")
        np_to_pil_uint8(np.repeat(mask_crop[..., None].astype(np.float32), 3, axis=2)).save(debug_path / f"{stem}_crop_mask.png")

    car_only_paths = {"ref": None, "gen": None}
    if car_only_dir:
        car_only_path = Path(car_only_dir)
        car_only_path.mkdir(parents=True, exist_ok=True)

        ref_car_path = car_only_path / f"{stem}_ref_car_only.png"
        gen_car_path = car_only_path / f"{stem}_gen_car_only.png"
        np_to_pil_uint8(ref_preview).save(ref_car_path)
        np_to_pil_uint8(gen_preview).save(gen_car_path)

        car_only_paths = {
            "ref": str(ref_car_path),
            "gen": str(gen_car_path),
        }

    return {
        "lpips_car_only": lpips_car,
        "ssim_car_only": ssim_car,
        "car_only_paths": car_only_paths,
        "debug": debug,
        "masks": {"ref_mask": ref_mask, "gen_mask": gen_mask, "car_mask": mask},
    }


def compute_delta_e(ref, gen):
    ref_lab = color.rgb2lab(ref)
    gen_lab = color.rgb2lab(gen)
    delta_map = color.deltaE_ciede2000(ref_lab, gen_lab)
    return float(np.mean(delta_map))


def compute_masked_delta_e(ref, gen, mask):
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        return compute_delta_e(ref, gen)

    ref_lab = color.rgb2lab(ref)
    gen_lab = color.rgb2lab(gen)
    delta_map = color.deltaE_ciede2000(ref_lab, gen_lab)
    return float(np.mean(delta_map[metric_mask]))


def convert_metrics_to_percent(ssim_val, lpips_val, delta_e_val):
    ssim_percent = float(np.clip(ssim_val * 100.0, 0.0, 100.0))
    lpips_similarity_percent = float(np.clip((1.0 - lpips_val) * 100.0, 0.0, 100.0))
    delta_e_similarity_percent = float(np.clip(100.0 - delta_e_val, 0.0, 100.0))

    return {
        "ssim_percent": ssim_percent,
        "lpips_similarity_percent": lpips_similarity_percent,
        "delta_e_similarity_percent": delta_e_similarity_percent,
    }


def convert_lpips_to_similarity_percent(lpips_val):
    if lpips_val is None:
        return None
    return float(np.clip((1.0 - lpips_val) * 100.0, 0.0, 100.0))


def compute_foreground_mask_union(ref, gen):
    ref_mask = create_foreground_mask(ref)
    gen_mask = create_foreground_mask(gen)
    union_mask = ref_mask | gen_mask

    if not np.any(union_mask):
        h, w = ref.shape[:2]
        return np.ones((h, w), dtype=bool)

    return union_mask.astype(bool)


def format_percent(percent_value):
    if percent_value is None:
        return "None"
    return f"{percent_value:.2f}%"


def create_foreground_mask(img, min_coverage=0.01, min_object_area=256, max_hole_area=1024):
    gray = color.rgb2gray(img)
    otsu = threshold_otsu(gray)

    mask_dark = gray <= otsu
    mask_light = gray > otsu

    coverage_dark = float(np.mean(mask_dark))
    coverage_light = float(np.mean(mask_light))

    if coverage_dark < min_coverage and coverage_light < min_coverage:
        fallback = gray <= np.mean(gray)
        base_mask = fallback.astype(bool)
    elif coverage_dark <= coverage_light:
        base_mask = mask_dark.astype(bool)
    else:
        base_mask = mask_light.astype(bool)

    base_mask = remove_small_objects(base_mask, min_size=max(1, int(min_object_area)))
    base_mask = remove_small_holes(base_mask, area_threshold=max(1, int(max_hole_area)))
    return base_mask.astype(bool)


def safe_centroid(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        center_y = h / 2.0
        center_x = w / 2.0
        return np.array([center_y, center_x], dtype=np.float32)

    mean_y = float(np.mean(ys))
    mean_x = float(np.mean(xs))
    return np.array([mean_y, mean_x], dtype=np.float32)


def build_empty_car_mask_metrics():
    return {
        "mask_iou": None,
        "mask_dice": None,
        "mask_area_ratio": None,
        "centroid_distance_px": None,
        "centroid_distance_norm": None,
        "hausdorff_px": None,
        "hausdorff_norm": None,
        "mask_metric_scope": "none",
    }


def compute_car_mask_metrics(ref_mask, gen_mask, include_hausdorff=True):
    ref_mask = np.asarray(ref_mask, dtype=bool)
    gen_mask = np.asarray(gen_mask, dtype=bool)
    if ref_mask.shape != gen_mask.shape:
        raise ValueError(f"Car-Masken müssen dieselbe Form haben ({ref_mask.shape} vs. {gen_mask.shape}).")

    ref_area = int(np.sum(ref_mask))
    gen_area = int(np.sum(gen_mask))
    intersection = int(np.sum(ref_mask & gen_mask))
    union = int(np.sum(ref_mask | gen_mask))

    if union > 0:
        iou = float(intersection / union)
    else:
        iou = 0.0

    if (ref_area + gen_area) > 0:
        dice = float((2 * intersection) / (ref_area + gen_area))
    else:
        dice = 0.0

    if ref_area > 0:
        area_ratio = float(gen_area / ref_area)
    else:
        area_ratio = 0.0

    ref_center = safe_centroid(ref_mask)
    gen_center = safe_centroid(gen_mask)

    h, w = ref_mask.shape
    if h > 0 and w > 0:
        diag = float(np.hypot(h, w))
    else:
        diag = 1.0

    centroid_distance_px = float(np.linalg.norm(ref_center - gen_center))
    centroid_distance_norm = float(centroid_distance_px / diag)

    hausdorff_px = None
    hausdorff_norm = None
    if include_hausdorff:
        hausdorff_px = float(hausdorff_distance(ref_mask, gen_mask))
        hausdorff_norm = float(hausdorff_px / diag)

    metrics = {
        "mask_iou": iou,
        "mask_dice": dice,
        "mask_area_ratio": area_ratio,
        "centroid_distance_px": centroid_distance_px,
        "centroid_distance_norm": centroid_distance_norm,
        "hausdorff_px": hausdorff_px,
        "hausdorff_norm": hausdorff_norm,
        "mask_metric_scope": "car_mask",
    }
    return metrics


def evaluate_pair(
    ref_path,
    gen_path,
    lpips_model,
    mode="letterbox",
    out_dir="Normalisiert",
    use_gpu=False,
    segmenter=None,
    car_mode="neutralize_crop",
    mask_source="union",
    pad_px=20,
    neutral_value=0.5,
    min_mask_area=0,
    mask_downsample="bilinear",
    eps=1e-8,
    debug_dir=None,
    car_only_dir=None,
    mask_grow_px=10,
    mask_min_object_area=500,
    mask_max_hole_area=3000,
    mask_trim_px=1,
    include_hausdorff=True,
    lpips_heatmap_dir=None,
    heatmap_focus_mode="global",
    roi_min_size_px=64,
    roi_square=True,
    max_metric_long_edge=1600,
    car_only_enabled=False,
):
    ref_img = load_image(ref_path)
    gen_img = load_image(gen_path)
    validate_image_for_metrics(ref_img, image_name="ref_img")
    validate_image_for_metrics(gen_img, image_name="gen_img")

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm, gen_norm, content_mask, metric_scale = normalize_pair_for_metrics(
        ref_img,
        gen_img,
        mode=mode,
        max_metric_long_edge=max_metric_long_edge,
    )
    validate_image_for_metrics(ref_norm, image_name="ref_norm")
    validate_image_for_metrics(gen_norm, image_name="gen_norm")
    norm_h, norm_w = ref_norm.shape[:2]

    basename = Path(ref_path).stem
    ref_norm_path, gen_norm_path = save_normalized_pair(ref_norm, gen_norm, basename, out_dir)

    valid_content_mask = prepare_metric_mask(content_mask, ref_norm, gen_norm)
    ssim_val = compute_masked_ssim(ref_norm, gen_norm, valid_content_mask, neutral_value=neutral_value)
    lpips_val = compute_lpips_on_content(
        ref_norm,
        gen_norm,
        valid_content_mask,
        lpips_model,
        use_gpu=use_gpu,
        mask_downsample=mask_downsample,
        eps=eps,
    )
    delta_e_val = compute_masked_delta_e(ref_norm, gen_norm, valid_content_mask)
    percent_metrics = convert_metrics_to_percent(ssim_val, lpips_val, delta_e_val)
    lpips_spatial_path = None
    lpips_map_mean = None
    lpips_map = None
    if lpips_heatmap_dir is not None:
        lpips_map_mean, lpips_map = compute_lpips_with_map(
            ref_norm,
            gen_norm,
            lpips_model,
            use_gpu=use_gpu,
        )
    if lpips_heatmap_dir is not None:
        heatmap_dir = Path(lpips_heatmap_dir)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        spatial_path = heatmap_dir / f"{basename}_lpips_spatial.json"
        lpips_spatial_path = str(spatial_path)

    foreground_mask = compute_foreground_mask_union(ref_norm, gen_norm)
    if valid_content_mask is not None:
        foreground_mask = foreground_mask & valid_content_mask
    lpips_foreground = masked_lpips(
        ref_norm,
        gen_norm,
        foreground_mask,
        lpips_model,
        use_gpu=use_gpu,
        mask_downsample=mask_downsample,
        eps=eps,
    )
    lpips_foreground_similarity_percent = convert_lpips_to_similarity_percent(lpips_foreground)

    active_segmenter = segmenter if car_only_enabled else None
    car_metrics = compute_car_only_metrics(
        ref_norm,
        gen_norm,
        ref_path,
        gen_path,
        lpips_model,
        active_segmenter,
        car_mode=car_mode,
        mask_source=mask_source,
        pad_px=pad_px,
        neutral_value=neutral_value,
        min_mask_area=min_mask_area,
        mask_downsample=mask_downsample,
        eps=eps,
        debug_dir=debug_dir if car_only_enabled else None,
        car_only_dir=car_only_dir if car_only_enabled else None,
        use_gpu=use_gpu,
        mask_grow_px=mask_grow_px,
        mask_min_object_area=mask_min_object_area,
        mask_max_hole_area=mask_max_hole_area,
        mask_trim_px=mask_trim_px,
        roi_min_size_px=roi_min_size_px,
        roi_square=roi_square,
    )
    lpips_car_only_similarity_percent = convert_lpips_to_similarity_percent(car_metrics["lpips_car_only"])

    car_masks = car_metrics.get("masks", {})
    ref_car_mask = car_masks.get("ref_mask")
    gen_car_mask = car_masks.get("gen_mask")
    car_focus_mask = car_masks.get("car_mask")

    valid_heatmap_focus_mode = {"global", "car_only"}
    if heatmap_focus_mode not in valid_heatmap_focus_mode:
        raise ValueError(
            f"heatmap_focus_mode muss einer von {sorted(valid_heatmap_focus_mode)} sein. "
            f"Aktuell: {heatmap_focus_mode}"
        )

    heatmap_overlay_mask = None
    heatmap_mask_mode = None
    heatmap_outline_mask = None
    heatmap_outline_mode = None
    if heatmap_focus_mode == "car_only" and car_focus_mask is not None and np.any(car_focus_mask):
        heatmap_overlay_mask = car_focus_mask
        heatmap_mask_mode = "car_focus"
        heatmap_outline_mask = car_focus_mask
        heatmap_outline_mode = "car_outline"

    if lpips_heatmap_dir is not None and lpips_spatial_path and lpips_map_mean is not None and lpips_map is not None:
        save_lpips_spatial_map(
            lpips_map,
            Path(lpips_spatial_path),
            overlay_mask=heatmap_overlay_mask,
            mask_mode=heatmap_mask_mode,
            outline_mask=heatmap_outline_mask,
            outline_mode=heatmap_outline_mode,
        )

    has_valid_car_masks = ref_car_mask is not None and gen_car_mask is not None and np.any(ref_car_mask) and np.any(gen_car_mask)
    if has_valid_car_masks:
        geometric = compute_car_mask_metrics(ref_car_mask, gen_car_mask, include_hausdorff=include_hausdorff)
    else:
        geometric = build_empty_car_mask_metrics()

    print("------------------------------------------------------------")
    print(f"Pair: {basename}")
    print(f"  Reference original : {ref_w}x{ref_h}")
    print(f"  Generated original : {gen_w}x{gen_h}")
    print(f"  Normalized sizes   : {norm_w}x{norm_h}")
    print(f"  Metric scale factor: {metric_scale:.6f}")
    if valid_content_mask is not None:
        content_area_px = int(np.sum(valid_content_mask))
        content_area_ratio = float(content_area_px / valid_content_mask.size)
        print("  Main scope         : content_mask")
        print(f"  Content area (px)  : {content_area_px}")
        print(f"  Content area (%)   : {content_area_ratio * 100.0:.2f}%")
    else:
        content_area_px = int(ref_norm.shape[0] * ref_norm.shape[1])
        content_area_ratio = 1.0
        print("  Main scope         : full_frame (Fallback)")
    print(f"  SSIM               : {ssim_val:.6f}")
    print(f"  SSIM (%)           : {percent_metrics['ssim_percent']:.2f}%")
    print(f"  LPIPS              : {lpips_val:.6f}")
    print(f"  LPIPS Similarity % : {percent_metrics['lpips_similarity_percent']:.2f}%")
    if lpips_map_mean is not None:
        print(f"  LPIPS map mean     : {lpips_map_mean:.6f}")
        print(f"  LPIPS Spatial map  : {lpips_spatial_path}")
    print(f"  LPIPS foreground   : {lpips_foreground:.6f}")
    print(f"  LPIPS foreground % : {format_percent(lpips_foreground_similarity_percent)}")
    if car_only_enabled and active_segmenter is not None:
        print(f"  Mask area (%)      : {car_metrics['debug']['mask_area_ratio'] * 100.0:.2f}%")
        print(f"  BBox (Metrik)      : {car_metrics['debug']['metric_bbox']}")
        print(f"  BBox (Preview Ref) : {car_metrics['debug']['ref_preview_bbox']}")
        print(f"  BBox (Preview Gen) : {car_metrics['debug']['gen_preview_bbox']}")
        print(f"  LPIPS Fahrzeugmodus     : {car_metrics['lpips_car_only']}")
        print(f"  LPIPS Fahrzeugmodus (%) : {format_percent(lpips_car_only_similarity_percent)}")
        if car_metrics.get("car_only_paths", {}).get("ref"):
            print(f"  Fahrzeugmodus Ref saved : {car_metrics['car_only_paths']['ref']}")
            print(f"  Fahrzeugmodus Gen saved : {car_metrics['car_only_paths']['gen']}")
    elif car_only_enabled:
        print("  Fahrzeugmodus           : deaktiviert (Segmentierung nicht verfügbar)")
    else:
        print("  Fahrzeugmodus           : deaktiviert")
    print(f"  Mask metric scope  : {geometric['mask_metric_scope']}")
    print(f"  Delta E (CIEDE2000): {delta_e_val:.6f}")
    print(f"  Delta E Similarity %: {percent_metrics['delta_e_similarity_percent']:.2f}%")
    print(f"  Saved ref_norm     : {ref_norm_path}")
    print(f"  Saved gen_norm     : {gen_norm_path}")

    result = {
        "filename": Path(ref_path).name,
        "reference_width": ref_w,
        "reference_height": ref_h,
        "generated_width": gen_w,
        "generated_height": gen_h,
        "normalized_width": norm_w,
        "normalized_height": norm_h,
        "metric_scale_factor": metric_scale,
        "normalization_mode": mode,
        "main_metric_scope": "content_mask" if valid_content_mask is not None else "full_frame_fallback",
        "content_mask_area_px": content_area_px,
        "content_mask_area_ratio": content_area_ratio,
        "ssim": ssim_val,
        "ssim_percent": percent_metrics["ssim_percent"],
        "lpips": lpips_val,
        "lpips_map_mean": lpips_map_mean,
        "lpips_spatial_path": lpips_spatial_path,
        "lpips_similarity_percent": percent_metrics["lpips_similarity_percent"],
        "lpips_foreground": lpips_foreground,
        "lpips_foreground_similarity_percent": lpips_foreground_similarity_percent,
        "delta_e_ciede2000": delta_e_val,
        "delta_e_similarity_percent": percent_metrics["delta_e_similarity_percent"],
        "ref_norm_path": ref_norm_path,
        "gen_norm_path": gen_norm_path,
        "lpips_car_only": car_metrics["lpips_car_only"],
        "lpips_car_only_similarity_percent": lpips_car_only_similarity_percent,
        "ssim_car_only": car_metrics["ssim_car_only"],
        "car_mask_area_ratio": car_metrics["debug"]["mask_area_ratio"],
        "car_bbox": json.dumps(car_metrics["debug"]["bbox"]) if car_metrics["debug"]["bbox"] else None,
        "car_metric_bbox": json.dumps(car_metrics["debug"]["metric_bbox"]) if car_metrics["debug"]["metric_bbox"] else None,
        "car_ref_preview_bbox": json.dumps(car_metrics["debug"]["ref_preview_bbox"]) if car_metrics["debug"]["ref_preview_bbox"] else None,
        "car_gen_preview_bbox": json.dumps(car_metrics["debug"]["gen_preview_bbox"]) if car_metrics["debug"]["gen_preview_bbox"] else None,
        "car_fallback_reason": car_metrics["debug"]["fallback_reason"],
        "car_only_ref_path": car_metrics["car_only_paths"]["ref"] if car_metrics.get("car_only_paths") else None,
        "car_only_gen_path": car_metrics["car_only_paths"]["gen"] if car_metrics.get("car_only_paths") else None,
    }
    result.update(geometric)
    return result


def evaluate_folders(
    reference_dir,
    generated_dir,
    output_csv,
    lpips_model,
    mode="letterbox",
    out_dir="Normalisiert",
    use_gpu=False,
    segmenter=None,
    car_mode="neutralize_crop",
    mask_source="union",
    pad_px=20,
    neutral_value=0.5,
    min_mask_area=0,
    mask_downsample="bilinear",
    eps=1e-8,
    debug_dir=None,
    car_only_dir=None,
    mask_grow_px=10,
    mask_min_object_area=500,
    mask_max_hole_area=3000,
    mask_trim_px=1,
    include_hausdorff=True,
    lpips_heatmap_dir=None,
    heatmap_focus_mode="global",
    roi_min_size_px=64,
    roi_square=True,
    max_metric_long_edge=1600,
    car_only_enabled=False,
    lpips_net="alex",
):
    ref_dir = Path(reference_dir)
    gen_dir = Path(generated_dir)

    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference-Ordner existiert nicht: {ref_dir}")
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"Generated-Ordner existiert nicht: {gen_dir}")

    ref_files = []
    for path in ref_dir.iterdir():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            ref_files.append(path)
    ref_files.sort()

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
            segmenter=segmenter,
            car_mode=car_mode,
            mask_source=mask_source,
            pad_px=pad_px,
            neutral_value=neutral_value,
            min_mask_area=min_mask_area,
            mask_downsample=mask_downsample,
            eps=eps,
            debug_dir=debug_dir,
            car_only_dir=car_only_dir,
            mask_grow_px=mask_grow_px,
            mask_min_object_area=mask_min_object_area,
            mask_max_hole_area=mask_max_hole_area,
            mask_trim_px=mask_trim_px,
            include_hausdorff=include_hausdorff,
            lpips_heatmap_dir=lpips_heatmap_dir,
            heatmap_focus_mode=heatmap_focus_mode,
            roi_min_size_px=roi_min_size_px,
            roi_square=roi_square,
            max_metric_long_edge=max_metric_long_edge,
            car_only_enabled=car_only_enabled,
        )
        results.append(result)

    if not results:
        raise RuntimeError("Keine auswertbaren Bildpaare gefunden.")

    df = build_result_dataframe(results)
    output_paths = write_result_files(df, output_csv, include_car_only=car_only_enabled, lpips_net=lpips_net)

    print("============================================================")
    print(f"[INFO] Summary-CSV gespeichert: {output_paths['csv']}")
    preview_columns = [
        "filename",
        "ssim",
        "ssim_percent",
        "lpips",
        "lpips_similarity_percent",
        "lpips_foreground",
        "lpips_foreground_similarity_percent",
        "delta_e_ciede2000",
        "delta_e_similarity_percent",
        "mask_iou",
        "mask_dice",
    ]
    if car_only_enabled:
        preview_columns.insert(9, "lpips_car_only")
        preview_columns.insert(10, "lpips_car_only_similarity_percent")
    print(df[[column for column in preview_columns if column in df.columns]].head())


def parse_optional_metric_long_edge(value):
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return int(value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Vergleiche Referenz- und generierte Bilder mit automatischer Größen-Normalisierung.",
        epilog="Falls skimage fehlt: pip install scikit-image",
    )

    parser.add_argument(
        "--mode",
        choices=["letterbox"],
        default="letterbox",
        help="Normalisierungsmodus",
    )
    parser.add_argument("--out", default="Normalisiert", help="Output-Ordner für normalisierte Bilder")
    parser.add_argument("--output-csv", default="Ergebnisse.csv", help="CSV-Datei für Metrikergebnisse")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="Backbone für LPIPS; alex ist empfohlen")
    parser.add_argument("--lpips-heatmap-dir", default="Abweichungsmatrix", help="Ausgabeordner für LPIPS-Heatmaps (setze 'none' zum Deaktivieren)")
    parser.add_argument("--use-gpu", action="store_true", help="Nutze CUDA, falls verfügbar")
    parser.add_argument("--seed", type=int, default=None, help="Setze optionalen Zufalls-Seed für reproduzierbare Läufe")
    parser.add_argument("--deterministic", action="store_true", help="Aktiviere deterministische Backends (langsamer, aber reproduzierbarer)")
    parser.add_argument("--enable-car-only", action="store_true", help="Aktiviere Fahrzeugmodus Metriken (LPIPS/SSIM); ist standardmäßig aktiv")
    parser.add_argument("--car-only", action="store_true", help="Kurzform für --enable-car-only")
    parser.add_argument("--disable-car-only", "--no-car-only", action="store_true", help="Deaktiviere Fahrzeugmodus Metriken und blende Fahrzeugmodus Ergebniswerte aus")
    parser.add_argument(
        "--car-mode",
        default="neutralize_crop",
        choices=["neutralize_crop", "roi_crop", "weighted_lpips"],
        help="Auto-fokussierte Fahrzeugmodus-Berechnung (weighted_lpips ist deprecated und wird auf neutralize_crop umgebogen)",
    )
    parser.add_argument("--mask-source", default="union", choices=["ref", "gen", "union"], help="Quelle für die Auto-Maske")
    parser.add_argument("--pad-px", type=int, default=20, help="Padding für Car-Crop-BBox")
    parser.add_argument("--neutral-value", type=float, default=0.5, help="Neutralwert für Hintergrundpixel [0..1]")
    parser.add_argument("--min-mask-area", type=int, default=0, help="Minimale Maskenfläche in Pixel")
    parser.add_argument("--mask-downsample", default="bilinear", choices=["bilinear", "nearest"], help="Deprecated: ohne produktive Wirkung")
    parser.add_argument("--mask-grow-px", type=int, default=10, help="Erweitere die Car-Maske lokal um X Pixel")
    parser.add_argument("--mask-min-object-area", type=int, default=500, help="Entferne sehr kleine Maskeninseln")
    parser.add_argument("--mask-max-hole-area", type=int, default=3000, help="Fülle kleine Löcher in der Car-Maske")
    parser.add_argument("--mask-trim-px", type=int, default=1, help="Schneide Maskenrand um X Pixel ein für sauberere Konturen")
    parser.add_argument("--roi-min-size-px", type=int, default=64, help="Minimale Kantenlänge der Car-ROI in Pixel")
    parser.add_argument("--roi-square", action="store_true", help="Erzwinge quadratische Car-ROI für stabilere Vergleiche")
    parser.add_argument(
        "--max-metric-long-edge",
        type=parse_optional_metric_long_edge,
        default=1600,
        help="Skaliere normalisierte Bilder vor der Metrik-Berechnung auf diese maximale Kantenlänge (Performance-Schutz); 'none' deaktiviert den Schutz.",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Deprecated: ohne produktive Wirkung")
    parser.add_argument("--mask-score-threshold", type=float, default=0.5, help="Score-Schwelle für Vehicle-Segmentierung")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Pixel-Schwelle der Segmentierungsmaske [0..1]")
    parser.add_argument("--debug-dir", default=None, help="Optionales Debug-Verzeichnis für Masken/Crops")
    parser.add_argument("--car-only-dir", default="Fahrzeugmodus", help="Verzeichnis für gespeicherte Fahrzeugmodus-Crops")
    parser.add_argument(
        "--skip-hausdorff",
        action="store_true",
        help="Überspringe Hausdorff-Distanz für schnellere Batch-Berechnung",
    )

    parser.add_argument("--ref", help="Pfad zu einem einzelnen Referenzbild")
    parser.add_argument("--gen", help="Pfad zu einem einzelnen Generated-Bild")
    parser.add_argument("--reference-dir", default="reference", help="Ordner mit Referenzbildern")
    parser.add_argument("--generated-dir", default="generated", help="Ordner mit Generated-Bildern")

    args = parser.parse_args()
    if isinstance(args.lpips_heatmap_dir, str) and args.lpips_heatmap_dir.strip().lower() == "none":
        args.lpips_heatmap_dir = None

    use_car_specific_option = any(
        [
            args.car_mode != "neutralize_crop",
            args.mask_source != "union",
            args.pad_px != 20,
            args.neutral_value != 0.5,
            args.min_mask_area != 0,
            args.mask_downsample != "bilinear",
            args.mask_grow_px != 10,
            args.mask_min_object_area != 500,
            args.mask_max_hole_area != 3000,
            args.mask_trim_px != 1,
            args.roi_min_size_px != 64,
            args.roi_square,
            args.eps != 1e-8,
            args.mask_score_threshold != 0.5,
            args.mask_threshold != 0.5,
            args.debug_dir is not None,
        ]
    )

    args.enable_car_only = not args.disable_car_only
    if args.car_only or args.enable_car_only or use_car_specific_option:
        args.enable_car_only = True
    if args.disable_car_only:
        args.enable_car_only = False
    return args


def main():
    args = parse_args()
    configure_determinism(seed=args.seed, deterministic=args.deterministic)

    print("============================================================")
    print("[INFO] Starte Bildmetrik-Berechnung")
    print(f"[INFO] Mode            : {args.mode}")
    print(f"[INFO] Normalized out  : {args.out}")
    print(f"[INFO] Output Basis    : {args.output_csv}")
    print(f"[INFO] LPIPS Net       : {get_lpips_net_label(args.lpips_net)}")
    print("[INFO] LPIPS Setup     : offizielles vortrainiertes Inferenzmodell (lin, kein Training im Tool)")
    print(f"[INFO] LPIPS Heatmaps  : {args.lpips_heatmap_dir}")
    print(f"[INFO] Seed            : {args.seed}")
    print(f"[INFO] Deterministisch : {args.deterministic}")
    print(f"[INFO] ROI min-size px : {args.roi_min_size_px}")
    print(f"[INFO] ROI square      : {args.roi_square}")
    print(f"[INFO] Max metric edge : {args.max_metric_long_edge}")
    print(f"[INFO] Fahrzeugmodus aktiv  : {args.enable_car_only}")
    print("============================================================")
    run_lpips_pipeline_sanity_checks()
    print("[INFO] Sanity-Check     : LPIPS-Pipeline geprüft")

    lpips_model = init_lpips_model(net=args.lpips_net, use_gpu=args.use_gpu)
    verify_lpips_forward(lpips_model, net=args.lpips_net, use_gpu=args.use_gpu)
    segmenter = None
    needs_car_segmenter = args.enable_car_only or (args.lpips_heatmap_dir is not None)
    if needs_car_segmenter:
        if args.enable_car_only:
            print("[INFO] Fahrzeugmodus wird aktiviert. Einfacher Aufruf: python image_metrics.py --car-only")
        try:
            segmenter = build_vehicle_segmenter(
                use_gpu=args.use_gpu,
                score_threshold=args.mask_score_threshold,
                mask_threshold=args.mask_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Fahrzeugsegmentierung nicht verfügbar: {exc}")
            if args.enable_car_only:
                print("[WARN] Fahrzeugmodus wurde deaktiviert, Heatmaps laufen global ohne Fahrzeugkontur weiter.")
                args.enable_car_only = False

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
            segmenter=segmenter,
            car_mode=args.car_mode,
            mask_source=args.mask_source,
            pad_px=args.pad_px,
            neutral_value=args.neutral_value,
            min_mask_area=args.min_mask_area,
            mask_downsample=args.mask_downsample,
            mask_grow_px=args.mask_grow_px,
            mask_min_object_area=args.mask_min_object_area,
            mask_max_hole_area=args.mask_max_hole_area,
            mask_trim_px=args.mask_trim_px,
            eps=args.eps,
            debug_dir=args.debug_dir,
            car_only_dir=args.car_only_dir if args.enable_car_only else None,
            include_hausdorff=not args.skip_hausdorff,
            lpips_heatmap_dir=args.lpips_heatmap_dir,
            heatmap_focus_mode="car_only" if args.enable_car_only else "global",
            roi_min_size_px=args.roi_min_size_px,
            roi_square=args.roi_square,
            max_metric_long_edge=args.max_metric_long_edge,
            car_only_enabled=args.enable_car_only,
        )
        df = build_result_dataframe([result])
        output_paths = write_result_files(df, args.output_csv, include_car_only=args.enable_car_only, lpips_net=args.lpips_net)
        print(f"[INFO] Einzelvergleich Summary-CSV gespeichert: {output_paths['csv']}")
        return

    evaluate_folders(
        reference_dir=args.reference_dir,
        generated_dir=args.generated_dir,
        output_csv=args.output_csv,
        lpips_model=lpips_model,
        mode=args.mode,
        out_dir=args.out,
        use_gpu=args.use_gpu,
        segmenter=segmenter,
        car_mode=args.car_mode,
        mask_source=args.mask_source,
        pad_px=args.pad_px,
        neutral_value=args.neutral_value,
        min_mask_area=args.min_mask_area,
        mask_downsample=args.mask_downsample,
        mask_grow_px=args.mask_grow_px,
        mask_min_object_area=args.mask_min_object_area,
        mask_max_hole_area=args.mask_max_hole_area,
        mask_trim_px=args.mask_trim_px,
        eps=args.eps,
        debug_dir=args.debug_dir,
        car_only_dir=args.car_only_dir if args.enable_car_only else None,
        include_hausdorff=not args.skip_hausdorff,
        lpips_heatmap_dir=args.lpips_heatmap_dir,
        heatmap_focus_mode="car_only" if args.enable_car_only else "global",
        roi_min_size_px=args.roi_min_size_px,
        roi_square=args.roi_square,
        max_metric_long_edge=args.max_metric_long_edge,
        car_only_enabled=args.enable_car_only,
        lpips_net=args.lpips_net,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        if is_no_space_error(exc):
            print(get_no_space_message(), file=sys.stderr)
            sys.exit(1)
        raise
