import argparse
import inspect
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import color
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.metrics import hausdorff_distance, structural_similarity
from skimage.morphology import closing, dilation, disk, erosion, opening, remove_small_holes, remove_small_objects
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
DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME = "mercedes_reflection_robust_v1"
DEFAULT_MERCEDES_WEIGHT_PROFILE = {
    "enabled": True,
    "name": DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME,
    "base_vehicle_weight": 0.82,
    "edge_weight": 0.38,
    "max_weight": 2.4,
    "reflection_min_weight": 0.38,
    "reflection_edge_floor": 0.78,
    "reflection_highlight_start": 0.62,
    "reflection_highlight_width": 0.32,
    "reflection_low_saturation_limit": 0.55,
    "reflection_highlight_weight": 0.65,
    "reflection_low_saturation_weight": 0.35,
    "glass_interior_weight": 0.0,
    "glass_contour_weight_floor": 0.92,
    "glass_contour_edge_boost": 0.32,
    "glass_zone_y_min": 0.18,
    "glass_zone_y_max": 0.58,
    "glass_zone_x_margin": 0.10,
    "glass_saturation_limit": 0.50,
    "glass_dark_limit": 0.34,
    "glass_bright_start": 0.58,
    "glass_min_area_ratio": 0.002,
    "downweight_threshold": 0.75,
    "reflection_difference_start": 0.10,
    "reflection_difference_width": 0.22,
    "reflection_blur_sigma": 3.0,
    "reflection_edge_change_limit": 0.12,
    "reflection_absolute_edge_limit": 0.35,
    "zones": {
        "star_grill_front_center": {"cx": 0.50, "cy": 0.55, "sx": 0.16, "sy": 0.18, "weight": 0.55},
        "headlights_light_signature": {"x_offset": 0.26, "cy": 0.50, "sx": 0.11, "sy": 0.13, "weight": 0.45},
        "wheels_tires": {"x_offset": 0.29, "cy": 0.76, "sx": 0.12, "sy": 0.12, "weight": 0.42},
        "side_body_character_line": {"cx": 0.50, "cy": 0.67, "sx": 0.42, "sy": 0.10, "weight": 0.22},
    },
}

DEFAULT_PRODUCT_INTEGRITY_PROFILE_NAME = "mercedes_product_integrity_v1"
DEFAULT_PRODUCT_INTEGRITY_PROFILE = {
    "enabled_components": {
        "structure_only_score": True,
        "detail_zones_score": True,
        "color_reflection_score": True,
        "car_only_lpips_score": True,
    },
    "weights": {
        "structure_only_score": 0.50,
        "detail_zones_score": 0.40,
        "color_reflection_score": 0.05,
        "car_only_lpips_score": 0.05,
        "legacy_weighted_lpips_weight": 0.0,
    },
    "thresholds": {
        "passed_product_integrity_min": 95.0,
        "passed_structure_min": 90.0,
        "passed_detail_min": 94.0,
        "failed_product_integrity_min": 90.0,
        "failed_structure_min": 80.0,
        "failed_detail_min": 80.0,
        "warning_color_reflection_max": 78.0,
        "critical_zone_score_max": 82.0,
        "warning_zone_score_max": 94.0,
        "soft_pass_product_integrity_min": 95.0,
        "warning_wheel_score_max": 94.0,
        "critical_wheel_score_max": 90.0,
        "warning_critical_component_score_max": 94.0,
        "critical_component_score_max": 90.0,
    },
    "contour_warning": {
        "high_iou_min": 0.98,
        "high_dice_min": 0.99,
        "critical_zone_score_max": 72.0,
        "warning_zone_score_max": 86.0,
        "max_hausdorff_norm": 0.018,
        "max_centroid_norm": 0.006,
        "max_area_ratio_delta": 0.025,
    },
    "reflection_tolerance": {
        "low_edge_difference_bonus": 0.35,
        "tolerated_score_below": 82.0,
        "color_score_penalty_cap_when_structure_stable": 1.2,
        "stable_structure_for_color_cap": 90.0,
        "stable_detail_for_color_cap": 92.0,
        "glass_interior_color_weight": 0.12,
        "glass_contour_color_weight": 0.45,
        "paint_color_weight": 1.0,
        "color_score_error_scale": 0.22,
        "color_score_quantile": 0.90,
    },
    "detail_zone_weights": {
        "silhouette": 1.35,
        "front_rear": 1.25,
        "wheels_tires": 2.25,
        "window_line": 1.20,
        "body_lines": 1.00,
        "center_grill_emblem": 1.20,
    },
}

FINDING_SEVERITY_RANK = {"tolerated": 0, "warning": 1, "critical": 2}

FINDING_TEXTS = {
    "headlight_light_signature": {
        "critical": "Kritische Änderung an Scheinwerfer oder Lichtsignatur erkannt",
        "warning": "Mögliche Änderung an Scheinwerfer oder Lichtsignatur erkannt",
        "tolerated": "Mögliche Änderung an Scheinwerfer oder Lichtsignatur erkannt",
    },
    "grille_front_structure": {
        "critical": "Mögliche Änderung am Kühlergrill erkannt",
        "warning": "Mögliche Änderung am Kühlergrill erkannt",
        "tolerated": "Mögliche Änderung am Kühlergrill erkannt",
    },
    "emblem_front_structure": {
        "critical": "Mögliche Änderung an Mercedes-Stern oder Emblem erkannt",
        "warning": "Mögliche Änderung an Mercedes-Stern oder Emblem erkannt",
        "tolerated": "Mögliche Änderung an Mercedes-Stern oder Emblem erkannt",
    },
    "wheel_tire_structure": {
        "critical": "Kritische Änderung an Felgen-, Reifen- oder Radstruktur erkannt",
        "warning": "Kritische Änderung an Felgen-, Reifen- oder Radstruktur erkannt",
        "tolerated": "Mögliche Änderung an Felgen-, Reifen- oder Radstruktur erkannt",
    },
    "window_line": {
        "critical": "Fensterlinie oder Dach-/Säulenstruktur auffällig",
        "warning": "Fensterlinie oder Dach-/Säulenstruktur auffällig",
        "tolerated": "Fensterlinie oder Dach-/Säulenstruktur auffällig",
    },
    "body_line_door_gap": {
        "critical": "Karosserielinie oder Türfuge auffällig",
        "warning": "Karosserielinie oder Türfuge auffällig",
        "tolerated": "Karosserielinie oder Türfuge auffällig",
    },
    "silhouette_contour": {
        "critical": "Relevante Abweichung an Fahrzeugkontur erkannt",
        "warning": "Geringe Rand- oder Konturabweichung bei hoher Maskenüberlappung toleriert",
        "tolerated": "Geringe Rand- oder Konturabweichung bei hoher Maskenüberlappung toleriert",
    },
    "glass_reflection": {
        "tolerated": "Reflexions- oder Helligkeitsunterschied auf Glasfläche erkannt",
        "warning": "Reflexions- oder Helligkeitsunterschied auf Glasfläche erkannt",
        "critical": "Reflexions- oder Helligkeitsunterschied auf Glasfläche erkannt",
    },
    "paint_reflection": {
        "tolerated": "Flächige Lichtabweichung auf Lackfläche ohne eindeutige Strukturänderung erkannt",
        "warning": "Flächige Lichtabweichung auf Lackfläche ohne eindeutige Strukturänderung erkannt",
        "critical": "Flächige Lichtabweichung auf Lackfläche ohne eindeutige Strukturänderung erkannt",
    },
    "color_shift": {
        "critical": "Farb- oder Helligkeitsabweichung mit Strukturbezug erkannt",
        "warning": "Farb- oder Helligkeitsabweichung mit Strukturbezug erkannt",
        "tolerated": "Farb- oder Helligkeitsabweichung erkannt",
    },
    "mask_alignment": {
        "critical": "Fahrzeugposition, Skalierung, Proportion oder Struktur deutlich abweichend",
        "warning": "Fahrzeugposition, Skalierung, Proportion oder Struktur deutlich abweichend",
        "tolerated": "Fahrzeugposition, Skalierung, Proportion oder Struktur leicht abweichend",
    },
    "front_rear_structure": {
        "critical": "Mögliche Änderung an Front-/Heck-Struktur erkannt",
        "warning": "Mögliche Änderung an Front-/Heck-Struktur erkannt",
        "tolerated": "Mögliche Änderung an Front-/Heck-Struktur erkannt",
    },
}

def add_finding(findings, canonical_key, severity, source):
    current = findings.get(canonical_key)
    if current is None:
        findings[canonical_key] = {"severity": severity, "sources": [source]}
        return
    if FINDING_SEVERITY_RANK.get(severity, 0) > FINDING_SEVERITY_RANK.get(current["severity"], 0):
        current["severity"] = severity
    if source not in current["sources"]:
        current["sources"].append(source)

def split_findings(findings):
    specific_front = {"headlight_light_signature", "grille_front_structure", "emblem_front_structure"}
    if "front_rear_structure" in findings and specific_front.intersection(findings):
        findings.pop("front_rear_structure", None)
    critical, tolerated = [], []
    for key, item in findings.items():
        severity = item["severity"]
        text = FINDING_TEXTS.get(key, {}).get(severity, key)
        if severity == "critical":
            critical.append(text)
        else:
            tolerated.append(text)
    return critical, tolerated

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
    "raw_lpips",
    "lpips",
    "lpips_similarity_percent",
    "lpips_map_mean",
    "lpips_foreground",
    "lpips_foreground_similarity_percent",
    "delta_e_ciede2000",
    "delta_e_similarity_percent",
    "lpips_car_only",
    "car_only_lpips",
    "lpips_car_only_similarity_percent",
    "legacy_debug_weighted_lpips_raw",
    "legacy_debug_weighted_lpips_similarity_percent",
    "legacy_reflection_robust_lpips_raw",
    "legacy_reflection_robust_lpips_similarity_percent",
    "final_similarity_score",
    "lpips_raw",
    "lpips_score",
    "car_only_lpips_raw",
    "car_only_lpips_score",
    "structure_only_score",
    "detail_zones_score",
    "color_reflection_score",
    "color_reflection_debug",
    "product_integrity_score",
    "product_integrity_base_score_before_caps",
    "product_integrity_final_score_after_caps",
    "product_integrity_score_delta_due_to_caps",
    "applied_caps",
    "applied_penalties",
    "hidden_findings_count",
    "product_integrity_decision",
    "critical_findings",
    "tolerated_findings",
    "headlight_score",
    "headlight_diff",
    "light_signature_score",
    "front_light_signature_score",
    "front_wheel_score",
    "rear_wheel_score",
    "wheel_tire_score",
    "tire_score",
    "grille_score",
    "emblem_score",
    "window_line_score",
    "silhouette_score",
    "critical_component_detected",
    "critical_component_names",
    "product_integrity_profile",
    "product_integrity_debug_paths",
    "product_integrity_debug",
    "product_integrity_interpretation",
    "decision_reason",
    "contour_warning_reason",
    "glass_mask_area_ratio",
    "window_contour_area_ratio",
    "detail_zone_area_ratio",
    "structure_masked_area_ratio",
    "mercedes_profile_enabled",
    "used_weight_profile",
    "reflection_weight_mean",
    "reflection_weight_min",
    "reflection_downweight_area_ratio",
    "glass_interior_area_ratio",
    "glass_contour_area_ratio",
    "weighted_lpips_weight_sum",
    "weighted_lpips_active_area_ratio",
    "weighted_lpips_region_contributions",
    "reflection_weight_map_path",
    "mercedes_weight_map_path",
    "critical_component_mask_path",
    "glass_interior_mask_path",
    "window_contour_mask_path",
    "window_candidate_region_path",
    "window_surface_mask_path",
    "window_line_mask_path",
    "overlay_window_contour_ref_path",
    "overlay_window_contour_gen_path",
    "weighted_lpips_map_path",
    "effective_weight_map_path",
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

CSV_FLOAT_COLUMNS = [
    "content_mask_area_ratio",
    "ssim",
    "ssim_percent",
    "lpips",
    "raw_lpips",
    "lpips_similarity_percent",
    "lpips_map_mean",
    "lpips_foreground",
    "lpips_foreground_similarity_percent",
    "delta_e_ciede2000",
    "delta_e_similarity_percent",
    "lpips_car_only",
    "car_only_lpips",
    "lpips_car_only_similarity_percent",
    "legacy_debug_weighted_lpips_raw",
    "legacy_debug_weighted_lpips_similarity_percent",
    "legacy_reflection_robust_lpips_raw",
    "legacy_reflection_robust_lpips_similarity_percent",
    "final_similarity_score",
    "headlight_score",
    "headlight_diff",
    "light_signature_score",
    "front_light_signature_score",
    "front_wheel_score",
    "rear_wheel_score",
    "wheel_tire_score",
    "tire_score",
    "grille_score",
    "emblem_score",
    "window_line_score",
    "silhouette_score",
    "reflection_weight_mean",
    "reflection_weight_min",
    "reflection_downweight_area_ratio",
    "glass_mask_area_ratio",
    "window_contour_area_ratio",
    "detail_zone_area_ratio",
    "structure_masked_area_ratio",
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


def merge_profile_defaults(default_profile, loaded_profile):
    merged = json.loads(json.dumps(default_profile))
    for key, value in loaded_profile.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def load_mercedes_weight_profile(config_path=None, profile_name=None):
    """Lade ein Mercedes-Gewichtungsprofil. Nutze ein robustes Default-Profil als Fallback."""
    profile = json.loads(json.dumps(DEFAULT_MERCEDES_WEIGHT_PROFILE))
    source = "default"

    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "mercedes_weight_profiles.json"
    config_path = Path(config_path)

    try:
        if config_path.exists():
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            profiles = payload.get("profiles", {})
            selected_name = profile_name or payload.get("default_profile") or DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME
            loaded_profile = profiles.get(selected_name)
            if loaded_profile is None:
                print(
                    f"[WARN] Mercedes Weight Profile '{selected_name}' nicht gefunden. "
                    f"Nutze Default-Profil '{DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME}'."
                )
            else:
                profile = merge_profile_defaults(profile, loaded_profile)
                profile["name"] = loaded_profile.get("name", selected_name)
                source = str(config_path)
        else:
            print(f"[WARN] Mercedes Weight Profile Datei fehlt: {config_path}. Nutze Default-Profil.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Mercedes Weight Profile konnte nicht geladen werden ({exc}). Nutze Default-Profil.")

    profile["enabled"] = bool(profile.get("enabled", True))
    profile["source"] = source
    return profile


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


def normalize_pair(ref_img, gen_img, mode="letterbox", pad_color=LETTERBOX_PAD_COLOR):
    if mode != "letterbox":
        raise ValueError("mode muss 'letterbox' sein")

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm = ref_img.copy()
    gen_pil = np_to_pil_uint8(gen_img)

    scale = min(ref_w / gen_w, ref_h / gen_h)
    scaled_w = max(1, int(round(gen_w * scale)))
    scaled_h = max(1, int(round(gen_h * scale)))

    resized = gen_pil.resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (ref_w, ref_h), color=pad_color)

    offset_x = (ref_w - scaled_w) // 2
    offset_y = (ref_h - scaled_h) // 2
    canvas.paste(resized, (offset_x, offset_y))

    gen_norm = np.asarray(canvas, dtype=np.float32) / 255.0

    content_mask = np.zeros((ref_h, ref_w), dtype=bool)
    content_mask[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = True
    return ref_norm, gen_norm, content_mask


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


def remove_objects_smaller_than(mask, min_size):
    min_size = max(1, int(min_size))
    if "max_size" in inspect.signature(remove_small_objects).parameters:
        return remove_small_objects(mask, max_size=max(0, min_size - 1))
    return remove_small_objects(mask, min_size=min_size)


def fill_holes_smaller_than(mask, area_threshold):
    area_threshold = max(1, int(area_threshold))
    if "max_size" in inspect.signature(remove_small_holes).parameters:
        return remove_small_holes(mask, max_size=max(0, area_threshold - 1))
    return remove_small_holes(mask, area_threshold=area_threshold)


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
        mask_img = Image.fromarray(metric_mask.astype(np.uint8) * 255)
        mask_img = mask_img.resize((expected_shape[1], expected_shape[0]), Image.Resampling.NEAREST)
        metric_mask = np.asarray(mask_img) > 127

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
    # LPIPS ruft intern ältere torchvision-Backbones mit ``pretrained=True`` auf.
    # Das ist nur eine Deprecation-Warnung aus torchvision und kein Laufzeitfehler;
    # die Initialisierung bleibt bewusst unverändert, damit dieselben trainierten
    # LPIPS-Gewichte genutzt werden.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*parameter 'pretrained' is deprecated.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum or `None` for 'weights'.*", category=UserWarning)
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


def resize_float_map_to_shape(value_map, target_shape, resample=Image.Resampling.BILINEAR):
    value_map = np.asarray(value_map, dtype=np.float32)
    target_rows, target_cols = target_shape
    if value_map.shape == (target_rows, target_cols):
        return value_map.astype(np.float32)

    # Erhalte absolute Gewichte. Eine Min/Max-Normalisierung würde echte 0-Ausschlüsse
    # und die fachliche Priorisierung beim Resizing verfälschen.
    max_value = float(np.max(value_map))
    if max_value <= 1.0:
        scale = 255.0
        arr = value_map * scale
    else:
        scale = 255.0 / max_value
        arr = value_map * scale
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    resized = np.asarray(img.resize((target_cols, target_rows), resample=resample), dtype=np.float32)
    if max_value <= 1.0:
        return (resized / 255.0).astype(np.float32)
    return (resized / scale).astype(np.float32)


def compute_edge_strength(img):
    gray = color.rgb2gray(img)
    gy, gx = np.gradient(gray.astype(np.float32))
    edge = np.hypot(gx, gy)
    return edge / (float(np.percentile(edge, 98)) + 1e-8)


def gaussian_zone(x, y, zone):
    cx = float(zone.get("cx", 0.5))
    cy = float(zone.get("cy", 0.5))
    sx = max(float(zone.get("sx", 0.1)), 1e-6)
    sy = max(float(zone.get("sy", 0.1)), 1e-6)
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def symmetric_gaussian_zone(x, y, zone):
    x_offset = float(zone.get("x_offset", 0.25))
    cy = float(zone.get("cy", 0.5))
    sx = max(float(zone.get("sx", 0.1)), 1e-6)
    sy = max(float(zone.get("sy", 0.1)), 1e-6)
    return np.exp(-(((np.abs(x - 0.50) - x_offset) / sx) ** 2 + ((y - cy) / sy) ** 2))


def build_critical_component_zones(mask, shape):
    """Erzeuge robuste Produktzonen aus Fahrzeugkontur und Bounding-Box."""
    scope = np.asarray(mask, dtype=bool) if mask is not None and np.any(mask) else np.ones(shape, dtype=bool)
    bbox = compute_mask_roi_bbox(scope, shape)
    # Front- und Heckdetails bewusst breiter fassen, danach strikt an der
    # finalen Fahrzeugmaske schneiden. So bleiben Grill/Stern/Scheinwerfer,
    # Rücklicht/Heckkontur und Räder geschützt, ohne Hintergrund aufzunehmen.
    headlight_zone = bbox_mask_from_fraction(shape, bbox, (0.00, 0.16, 0.46, 0.62))
    light_signature_zone = bbox_mask_from_fraction(shape, bbox, (0.00, 0.12, 0.44, 0.58))
    grille_zone = bbox_mask_from_fraction(shape, bbox, (0.00, 0.30, 0.40, 0.78))
    emblem_zone = bbox_mask_from_fraction(shape, bbox, (0.10, 0.34, 0.46, 0.70))
    front_wheel_zone = find_wheel_zone_from_mask(scope, bbox, "front")
    rear_wheel_zone = find_wheel_zone_from_mask(scope, bbox, "rear")
    tire_zone = front_wheel_zone | rear_wheel_zone
    rear_light_zone = bbox_mask_from_fraction(shape, bbox, (0.70, 0.20, 1.00, 0.66))
    rear_contour_zone = bbox_mask_from_fraction(shape, bbox, (0.72, 0.18, 1.00, 0.88))
    window_line_zone = bbox_mask_from_fraction(shape, bbox, (0.06, 0.06, 0.94, 0.56))
    roof_pillar_zone = bbox_mask_from_fraction(shape, bbox, (0.08, 0.02, 0.92, 0.34))
    side_body_zone = bbox_mask_from_fraction(shape, bbox, (0.06, 0.38, 0.96, 0.78))
    front_apron_zone = bbox_mask_from_fraction(shape, bbox, (0.00, 0.56, 0.40, 0.92))
    silhouette_zone = dilation(scope, disk(2)) & ~erosion(scope, disk(3))
    zones = {
        "headlight_zone": headlight_zone,
        "light_signature_zone": light_signature_zone,
        "grille_zone": grille_zone,
        "emblem_zone": emblem_zone,
        "front_wheel_zone": front_wheel_zone,
        "rear_wheel_zone": rear_wheel_zone,
        "tire_zone": tire_zone,
        "rear_light_zone": rear_light_zone,
        "rear_contour_zone": rear_contour_zone,
        "window_line_zone": window_line_zone,
        "roof_pillar_zone": roof_pillar_zone,
        "side_body_zone": side_body_zone,
        "front_apron_zone": front_apron_zone,
        "silhouette_zone": silhouette_zone,
        # Kompatible Kurznamen für die Komponenten-Auswertung:
        "headlight": headlight_zone,
        "front_light_signature": light_signature_zone,
        "grille": grille_zone,
        "emblem": emblem_zone,
        "front_wheel": front_wheel_zone,
        "rear_wheel": rear_wheel_zone,
        "wheel_tire": tire_zone,
        "rear_light": rear_light_zone,
        "rear_contour": rear_contour_zone,
        "window_line": window_line_zone,
        "roof_pillar": roof_pillar_zone,
        "side_body": side_body_zone,
        "front_apron": front_apron_zone,
        "silhouette": silhouette_zone,
    }
    for name, zone in list(zones.items()):
        min_ratio = 0.0005 if any(part in name for part in ("wheel", "grille", "emblem")) else 0.001
        zones[name], _ = validate_zone_against_vehicle(zone, scope, name, min_area_ratio=min_ratio)
    return zones


def build_mercedes_importance_map(ref, gen, car_mask=None, profile=None):
    """Gewichte produktkritische Fahrzeugstrukturen höher als glatte Flächen."""
    profile = profile or DEFAULT_MERCEDES_WEIGHT_PROFILE
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / max(1, w - 1)
    y = yy / max(1, h - 1)

    edge = np.clip(np.maximum(compute_edge_strength(ref), compute_edge_strength(gen)), 0.0, 1.0)
    weight = np.full((h, w), float(profile.get("base_vehicle_weight", 0.82)), dtype=np.float32)

    # Silhouette, Fensterlinie, Türfugen, Linienführung und Bauteilkanten bleiben wichtig.
    weight += float(profile.get("edge_weight", 0.38)) * edge.astype(np.float32)

    # Heuristische Mercedes-/Fahrzeugzonen: Stern/Grill/Frontmitte, Scheinwerfer, Felgen.
    zones = profile.get("zones", {})
    center_front = gaussian_zone(x, y, zones.get("star_grill_front_center", {}))
    lights = symmetric_gaussian_zone(x, y, zones.get("headlights_light_signature", {}))
    wheels = symmetric_gaussian_zone(x, y, zones.get("wheels_tires", {}))
    side_line = gaussian_zone(x, y, zones.get("side_body_character_line", {}))
    weight += (
        float(zones.get("star_grill_front_center", {}).get("weight", 0.55)) * center_front
        + float(zones.get("headlights_light_signature", {}).get("weight", 0.45)) * lights
        + float(zones.get("wheels_tires", {}).get("weight", 0.42)) * wheels
        + float(zones.get("side_body_character_line", {}).get("weight", 0.22)) * side_line
    )

    vehicle = prepare_metric_mask(car_mask, ref, gen) if car_mask is not None and np.any(car_mask) else np.ones((h, w), dtype=bool)
    component_zones = build_critical_component_zones(vehicle, (h, w))
    critical_component_mask = (
        component_zones["headlight_zone"]
        | component_zones["light_signature_zone"]
        | component_zones["grille_zone"]
        | component_zones["emblem_zone"]
        | component_zones["front_wheel_zone"]
        | component_zones["rear_wheel_zone"]
        | component_zones["tire_zone"]
        | component_zones["rear_light_zone"]
        | component_zones["rear_contour_zone"]
        | component_zones["window_line_zone"]
        | component_zones["silhouette_zone"]
    )
    front_detail_mask = (
        component_zones["headlight_zone"]
        | component_zones["light_signature_zone"]
        | component_zones["grille_zone"]
        | component_zones["emblem_zone"]
    )
    wheel_mask = component_zones["front_wheel_zone"] | component_zones["rear_wheel_zone"] | component_zones["tire_zone"]
    line_mask = component_zones["window_line_zone"] | component_zones["silhouette_zone"]
    weight = np.where(line_mask, np.maximum(weight, float(profile.get("critical_line_weight_floor", 1.35))), weight)
    weight = np.where(wheel_mask, np.maximum(weight, float(profile.get("critical_wheel_weight_floor", 1.80))), weight)
    weight = np.where(front_detail_mask, np.maximum(weight, float(profile.get("critical_front_detail_weight_floor", 2.10))), weight)

    if car_mask is not None and np.any(car_mask):
        weight = np.where(vehicle, weight, 0.0)

    glass_masks = build_glass_region_masks(ref, gen, car_mask=car_mask, profile=profile)
    glass_interior = glass_masks["interior"]
    glass_contour = glass_masks["contour"]
    if np.any(glass_interior):
        # Scheibeninnenflächen fließen nicht in Weighted-LPIPS ein; nur Konturen zählen.
        interior_weight = float(profile.get("glass_interior_weight", 0.0))
        weight = np.where(glass_interior, interior_weight, weight)
    if np.any(glass_contour):
        contour_floor = float(profile.get("glass_contour_weight_floor", 0.92))
        contour_boost = float(profile.get("glass_contour_edge_boost", 0.32))
        weight = np.where(glass_contour, np.maximum(weight, contour_floor + contour_boost * edge), weight)
    # Glas-/Reflexionslogik darf harte Produktbereiche nicht abdunkeln.
    weight = np.where(critical_component_mask, np.maximum(weight, float(profile.get("critical_component_weight_floor", 1.65))), weight)
    weight = np.where(front_detail_mask, np.maximum(weight, float(profile.get("critical_front_detail_weight_floor", 2.10))), weight)
    weight = np.where(wheel_mask, np.maximum(weight, float(profile.get("critical_wheel_weight_floor", 1.80))), weight)

    return np.clip(weight, 0.0, float(profile.get("max_weight", 2.4))).astype(np.float32)


def build_glass_region_masks(ref, gen, car_mask=None, profile=None):
    """Trenne Fahrzeugscheiben heuristisch in tolerierte Innenflächen und kritische Konturen."""
    profile = profile or DEFAULT_MERCEDES_WEIGHT_PROFILE
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]

    has_vehicle_mask = car_mask is not None and np.any(car_mask)
    if has_vehicle_mask:
        vehicle = prepare_metric_mask(car_mask, ref, gen)
    else:
        vehicle = np.ones((h, w), dtype=bool)
    x0, y0, x1, y1 = compute_mask_roi_bbox(vehicle, (h, w))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    bx = (xx - x0) / bw
    by = (yy - y0) / bh

    brightness = np.maximum(color.rgb2gray(ref), color.rgb2gray(gen)).astype(np.float32)
    saturation = np.maximum(color.rgb2hsv(ref)[..., 1], color.rgb2hsv(gen)[..., 1]).astype(np.float32)
    edge = np.clip(np.maximum(compute_edge_strength(ref), compute_edge_strength(gen)), 0.0, 1.0)
    smooth = edge < float(profile.get("glass_smooth_edge_limit", 0.42))

    window_candidate_region = (
        (by >= float(profile.get("window_candidate_y_min", 0.06)))
        & (by <= float(profile.get("window_candidate_y_max", 0.52)))
        & (bx >= float(profile.get("window_candidate_x_min", 0.08)))
        & (bx <= float(profile.get("window_candidate_x_max", 0.94)))
        & vehicle
    )
    # Schließe den vorderen unteren Bereich aus: Dort liegen Motorhaube, Grill und
    # Scheinwerferkanten, aber keine Fensterkontur.
    hood_front_exclusion = (
        (bx <= float(profile.get("window_front_exclusion_x_max", 0.34)))
        & (by >= float(profile.get("window_front_exclusion_y_min", 0.40)))
    )
    window_candidate_region &= ~hood_front_exclusion
    # Begrenze die Kandidaten auf zusammenhängende Kanten im Fensterband statt auf
    # eine grobe Box. Eine kleine Dilatation verbindet Dachlinie, Säulen und Beltline.
    window_edge_seed = window_candidate_region & (edge >= float(profile.get("window_candidate_edge_min", 0.06)))
    if np.any(window_edge_seed):
        connected_window_band = dilation(window_edge_seed, disk(max(1, int(profile.get("window_candidate_connect_px", 2)))))
        window_candidate_region &= connected_window_band | (by <= float(profile.get("window_surface_y_max", 0.48)))

    glass_band = window_candidate_region
    low_saturation = saturation <= float(profile.get("glass_saturation_limit", 0.50))
    dark_glass = brightness <= float(profile.get("glass_dark_limit", 0.34))
    bright_reflection = brightness >= float(profile.get("glass_bright_start", 0.58))
    smooth_glass = smooth if has_vehicle_mask else np.zeros((h, w), dtype=bool)
    glass_like = low_saturation & (dark_glass | bright_reflection | smooth_glass)

    candidate = glass_band & glass_like
    min_area = max(8, int(float(profile.get("glass_min_area_ratio", 0.002)) * max(np.sum(vehicle), 1)))
    candidate = remove_objects_smaller_than(candidate, min_area + 1)
    candidate = fill_holes_smaller_than(closing(candidate, disk(3)), min_area + 1)

    max_area = max(1, int(0.34 * np.sum(vehicle)))
    if np.sum(candidate) > max_area:
        stronger_glass_like = glass_band & low_saturation & (dark_glass | bright_reflection | (smooth_glass & (by <= 0.50)))
        candidate = fill_holes_smaller_than(closing(stronger_glass_like, disk(2)), min_area + 1)

    if not np.any(candidate):
        empty = np.zeros((h, w), dtype=bool)
        return {
            "candidate": empty,
            "window_candidate_region": window_candidate_region.astype(bool),
            "interior": empty,
            "surface": empty,
            "contour": empty,
            "line": empty,
        }

    radius = max(1, int(round(min(bh, bw) * float(profile.get("glass_contour_width_ratio", 0.018)))))
    surface = candidate.astype(bool)
    line_seed = window_candidate_region & (
        edge > float(profile.get("glass_contour_edge_limit", 0.30))
    )
    contour = surface & ~erosion(surface, disk(radius))
    line = (contour | dilation(line_seed & dilation(surface, disk(1)), disk(1))) & window_candidate_region & vehicle
    # Keine Motorhaube/Frontkanten: Schneide die finale Kontur erneut an der
    # fensterspezifischen Kandidatenregion.
    line &= ~hood_front_exclusion
    contour = line.astype(bool)
    interior = surface & ~contour

    return {
        "candidate": surface.astype(bool),
        "window_candidate_region": window_candidate_region.astype(bool),
        "interior": interior.astype(bool),
        "surface": surface.astype(bool),
        "contour": contour.astype(bool),
        "line": contour.astype(bool),
    }

def build_reflection_downweight_map(ref, gen, car_mask=None, content_mask=None, profile=None):
    """Reduziere weiche Reflexionsänderungen, erhalte aber Kanten und Geometrie."""
    profile = profile or DEFAULT_MERCEDES_WEIGHT_PROFILE
    metric_mask = prepare_metric_mask(car_mask, ref, gen)
    if metric_mask is None:
        metric_mask = prepare_metric_mask(content_mask, ref, gen)

    ref_gray = color.rgb2gray(ref).astype(np.float32)
    gen_gray = color.rgb2gray(gen).astype(np.float32)
    brightness = np.maximum(ref_gray, gen_gray).astype(np.float32)
    ref_hsv = color.rgb2hsv(ref)
    gen_hsv = color.rgb2hsv(gen)
    saturation = np.maximum(ref_hsv[..., 1], gen_hsv[..., 1]).astype(np.float32)
    edge = np.clip(np.maximum(compute_edge_strength(ref), compute_edge_strength(gen)), 0.0, 1.0)
    edge_delta = np.abs(compute_edge_strength(ref) - compute_edge_strength(gen)).astype(np.float32)

    smooth = 1.0 - edge
    highlight_start = float(profile.get("reflection_highlight_start", 0.62))
    highlight_width = max(float(profile.get("reflection_highlight_width", 0.32)), 1e-6)
    saturation_limit = max(float(profile.get("reflection_low_saturation_limit", 0.55)), 1e-6)
    highlights = np.clip((brightness - highlight_start) / highlight_width, 0.0, 1.0)
    low_saturation_gloss = np.clip((saturation_limit - saturation) / saturation_limit, 0.0, 1.0)
    appearance_likelihood = np.clip(
        (
            float(profile.get("reflection_highlight_weight", 0.65)) * highlights
            + float(profile.get("reflection_low_saturation_weight", 0.35)) * low_saturation_gloss
        )
        * smooth,
        0.0,
        1.0,
    )

    lab_delta = np.linalg.norm(color.rgb2lab(ref) - color.rgb2lab(gen), axis=2).astype(np.float32) / 60.0
    value_delta = np.abs(ref_hsv[..., 2] - gen_hsv[..., 2]).astype(np.float32)
    luma_delta = np.abs(ref_gray - gen_gray).astype(np.float32)
    color_delta = np.clip(0.45 * lab_delta + 0.30 * value_delta + 0.25 * luma_delta, 0.0, 1.0)
    blur_sigma = float(profile.get("reflection_blur_sigma", 3.0))
    soft_delta = gaussian(color_delta, sigma=blur_sigma, preserve_range=True).astype(np.float32)
    edge_change_limit = max(float(profile.get("reflection_edge_change_limit", 0.12)), 1e-6)
    absolute_edge_limit = max(float(profile.get("reflection_absolute_edge_limit", 0.35)), 1e-6)
    low_edge_change = np.clip(1.0 - edge_delta / edge_change_limit, 0.0, 1.0)
    low_absolute_edge = np.clip(1.0 - edge / absolute_edge_limit, 0.0, 1.0)
    reflection_difference = np.clip(0.55 * color_delta + 0.45 * soft_delta, 0.0, 1.0)
    difference_start = float(profile.get("reflection_difference_start", 0.10))
    difference_width = max(float(profile.get("reflection_difference_width", 0.22)), 1e-6)
    difference_likelihood = np.clip((reflection_difference - difference_start) / difference_width, 0.0, 1.0)
    difference_likelihood *= low_edge_change * np.clip(0.35 + 0.65 * low_absolute_edge, 0.0, 1.0)

    reflection_likelihood = np.maximum(appearance_likelihood, difference_likelihood)

    min_weight = float(profile.get("reflection_min_weight", 0.38))
    weight = 1.0 - (1.0 - min_weight) * reflection_likelihood
    # Kanten/Bauteilgrenzen ausdrücklich zurückholen.
    edge_floor = float(profile.get("reflection_edge_floor", 0.78))
    edge_protection = np.maximum(edge, np.clip(edge_delta / edge_change_limit, 0.0, 1.0))
    protected_weight = edge_floor + (1.0 - edge_floor) * edge_protection
    weight = np.where(edge_protection > 0.05, np.maximum(weight, protected_weight), weight)

    glass_masks = build_glass_region_masks(ref, gen, car_mask=car_mask, profile=profile)
    glass_interior = glass_masks["interior"]
    glass_contour = glass_masks["contour"]
    if np.any(glass_interior):
        # Scheibeninhalte wie Himmel, Wald, Innenraum und Reflexe sind nicht produkttreuerelevant.
        glass_weight = float(profile.get("glass_interior_weight", 0.0))
        weight = np.where(glass_interior, glass_weight, weight)
    if np.any(glass_contour):
        # A-/B-/C-Säule, Dachlinie und Fensterlinie bleiben produktrelevant.
        contour_floor = float(profile.get("glass_contour_weight_floor", 0.92))
        weight = np.where(glass_contour, np.maximum(weight, contour_floor), weight)

    # Kritische Produktkomponenten haben Vorrang vor Reflexionslogik. Reifen,
    # Felgen, Grill, Stern, Scheinwerfer und Rückleuchten werden nie
    # heruntergewichtet, auch wenn sie dunkel/glänzend wirken.
    if metric_mask is not None:
        critical_zones = build_critical_component_zones(metric_mask, ref.shape[:2])
        wheel_protected = (critical_zones["front_wheel_zone"] | critical_zones["rear_wheel_zone"] | critical_zones["tire_zone"]) & (edge >= 0.02)
        structure_protected = (
            critical_zones["headlight_zone"]
            | critical_zones["light_signature_zone"]
            | critical_zones["grille_zone"]
            | critical_zones["emblem_zone"]
            | critical_zones["rear_light_zone"]
            | critical_zones["rear_contour_zone"]
        ) & (edge >= 0.05)
        reflection_protected_mask = (wheel_protected | structure_protected) & ~glass_interior
        weight = np.where(reflection_protected_mask, 1.0, weight)

    if metric_mask is not None:
        weight = np.where(metric_mask, weight, 0.0)

    return np.clip(weight, 0.0, 1.0).astype(np.float32)


def build_weighted_lpips_scope_mask(car_mask, glass_interior_mask=None, content_mask=None):
    """Nutze nur Fahrzeugpixel und schließe Scheibeninnenflächen vollständig aus."""
    if car_mask is not None and np.any(car_mask):
        scope = np.asarray(car_mask, dtype=bool).copy()
    elif content_mask is not None and np.any(content_mask):
        scope = np.asarray(content_mask, dtype=bool).copy()
    else:
        scope = None

    if scope is not None and glass_interior_mask is not None and np.any(glass_interior_mask):
        scope &= ~np.asarray(glass_interior_mask, dtype=bool)
    return scope


def compute_weighted_lpips_from_map(dist_map, weight_map, mask=None, eps=1e-8, return_debug=False):
    dist_map = np.asarray(dist_map, dtype=np.float32)
    weights = resize_float_map_to_shape(weight_map, dist_map.shape)
    if mask is not None:
        resized_mask = resize_float_map_to_shape(np.asarray(mask, dtype=np.float32), dist_map.shape, resample=Image.Resampling.NEAREST)
        weights = np.where(resized_mask >= 0.5, weights, 0.0)
    weights = np.clip(weights, 0.0, None)
    weight_sum = float(np.sum(weights))
    if weight_sum <= float(eps):
        value = float(np.mean(dist_map))
    else:
        value = float(np.sum(dist_map * weights) / (weight_sum + float(eps)))
    if not return_debug:
        return value
    return {
        "value": value,
        "weight_sum": weight_sum,
        "active_area_ratio": float(np.mean(weights > 0.0)),
        "weighted_map": (dist_map * weights).astype(np.float32),
        "effective_weight_map": weights.astype(np.float32),
    }


def summarize_weighted_lpips_regions(dist_map, weight_map, regions, eps=1e-8):
    summaries = {}
    weight_map = np.asarray(weight_map, dtype=np.float32)
    for name, region_mask in regions.items():
        if region_mask is None:
            continue
        resized_region = resize_float_map_to_shape(
            np.asarray(region_mask, dtype=np.float32),
            weight_map.shape,
            resample=Image.Resampling.NEAREST,
        )
        region_weights = weight_map * (resized_region >= 0.5).astype(np.float32)
        debug = compute_weighted_lpips_from_map(dist_map, region_weights, eps=eps, return_debug=True)
        summaries[name] = {
            "distance": debug["value"],
            "weight_sum": debug["weight_sum"],
            "active_area_ratio": debug["active_area_ratio"],
        }
    return summaries


def save_weight_debug_map(weight_map, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(weight_map, dtype=np.float32), 0.0, None)
    arr = arr / (float(np.max(arr)) + 1e-8)
    Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def mask_debug_map_to_vehicle(values, vehicle_mask):
    """Setze Debugwerte außerhalb des Fahrzeugbereichs auf Schwarz."""
    arr = np.asarray(values, dtype=np.float32)
    metric_mask = prepare_metric_mask(vehicle_mask, arr[..., None] if arr.ndim == 2 else arr, arr[..., None] if arr.ndim == 2 else arr)
    if metric_mask is None:
        return arr
    return np.where(metric_mask, arr, 0.0)


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
        grown_mask = dilation(refined_mask, footprint=grow_disk)
        refined_mask = refined_mask | (merged_mask & grown_mask)

    # Säubere Segmentierungsrauschen, ohne auf eine Bounding Box auszuweichen:
    # Opening entfernt kleine Hintergrundreste; Closing/Hole-Filling schließen Rad- und
    # Karosserielücken kontrolliert.
    refined_mask = opening(refined_mask, footprint=disk(1))
    refined_mask = closing(refined_mask, footprint=disk(2))
    refined_mask = remove_objects_smaller_than(refined_mask, min_object_area)
    refined_mask = fill_holes_smaller_than(refined_mask, max_hole_area)
    refined_mask = closing(refined_mask, footprint=disk(1))

    if trim_px > 0:
        refined_mask = erosion(refined_mask, footprint=disk(int(trim_px)))
        refined_mask = remove_objects_smaller_than(refined_mask, min_object_area)

    return refined_mask.astype(bool)


def compute_shared_vehicle_crop_box(mask, pad_px=20, min_size_px=64, square=False):
    """Berechne eine gemeinsame Crop-Box aus der finalen Fahrzeugmaske.

    Koordinatenkonvention: x0/y0 inklusive, x1/y1 exklusiv.
    """
    mask = np.asarray(mask, dtype=bool)
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

    if square:
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


def compute_mask_bbox(mask, pad_px=20, min_size_px=64, make_square=True):
    return compute_shared_vehicle_crop_box(mask, pad_px=pad_px, min_size_px=min_size_px, square=make_square)


def validate_crop_box_for_array(image_or_mask, crop_box, name="image_or_mask"):
    arr = np.asarray(image_or_mask)
    if arr.ndim < 2:
        raise ValueError(f"{name} muss mindestens 2 Dimensionen haben, erhalten: {arr.shape}")
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    h, w = arr.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"Ungültige gemeinsame Crop-Box für {name}: "
            f"x0={x0}, y0={y0}, x1={x1}, y1={y1}, Bildgröße={w}x{h}. "
            "Konvention: x0/y0 inklusive, x1/y1 exklusiv."
        )


def apply_shared_crop(image_or_mask, crop_box):
    """Wende dieselbe exklusive Crop-Box unverändert auf Bild oder Maske an."""
    validate_crop_box_for_array(image_or_mask, crop_box)
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    return np.asarray(image_or_mask)[y0:y1, x0:x1].copy()


def validate_same_spatial_size(named_arrays, context):
    sizes = {name: tuple(np.asarray(arr).shape[:2]) for name, arr in named_arrays.items() if arr is not None}
    if not sizes:
        return
    expected_name, expected_size = next(iter(sizes.items()))
    mismatches = {name: size for name, size in sizes.items() if size != expected_size}
    if mismatches:
        formatted = ", ".join(f"{name}={size[1]}x{size[0]}" for name, size in sizes.items())
        raise ValueError(
            f"{context}: Größenvalidierung fehlgeschlagen. Erwartet wie {expected_name}="
            f"{expected_size[1]}x{expected_size[0]}, erhalten: {formatted}. "
            "Prüfe, dass ausschließlich die gemeinsame Crop-Box verwendet wird."
        )


def apply_neutralize_crop(img, mask, bbox, neutral_value=0.5):
    img_crop = apply_shared_crop(img, bbox)
    mask_crop = apply_shared_crop(mask, bbox).astype(bool)
    img_crop[~mask_crop] = neutral_value
    return img_crop, mask_crop


def apply_masked_car_crop(img, mask, bbox):
    img_crop = apply_shared_crop(img, bbox)
    mask_crop = apply_shared_crop(mask, bbox).astype(bool)
    masked_crop = np.zeros_like(img_crop)
    masked_crop[mask_crop] = img_crop[mask_crop]
    return masked_crop, mask_crop


def validate_binary_mask(mask, name="mask", min_area_px=1, max_hole_ratio=0.18, raise_on_large_holes=True):
    """Validiere eine Fahrzeug-/Zonenmaske und melde typische Pipeline-Defekte früh."""
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise ValueError(f"{name} muss zweidimensional sein, erhalten: {mask_bool.shape}.")
    area = int(np.sum(mask_bool))
    if area < int(min_area_px):
        raise ValueError(f"{name} ist leer oder zu klein ({area} px).")
    x0, y0, x1, y1 = compute_mask_roi_bbox(mask_bool, mask_bool.shape)
    roi = mask_bool[y0:y1, x0:x1]
    bbox_area = max(int(roi.size), 1)
    filled_roi = fill_holes_smaller_than(roi, bbox_area)
    holes = filled_roi & ~roi
    hole_ratio = float(np.sum(holes) / max(float(area), 1.0))
    warning = None
    if hole_ratio > float(max_hole_ratio):
        warning = (
            f"{name} enthält große Innenbereiche ({hole_ratio:.2%} der Maskenfläche). "
            "Das kann bei Fahrzeugen durch Fenster, Radhäuser oder offene Felgen plausibel sein; "
            "prüfe bei Bedarf die gespeicherten Masken-Overlays."
        )
        if raise_on_large_holes:
            raise ValueError(warning)
    return {
        "area_px": area,
        "bbox": (x0, y0, x1, y1),
        "hole_ratio": hole_ratio,
        "warning": warning,
    }


def save_mask_image(mask, path):
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def save_mask_overlay(image, mask, path, color_rgb=(1.0, 0.1, 0.0), alpha=0.65):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = np.asarray(image, dtype=np.float32).copy()
    overlay = base.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    color_arr = np.asarray(color_rgb, dtype=np.float32)
    overlay[mask_bool] = (1.0 - float(alpha)) * overlay[mask_bool] + float(alpha) * color_arr
    np_to_pil_uint8(np.clip(overlay, 0.0, 1.0)).save(path)


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
            "fallback_reason": "Car-only deaktiviert",
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
    mask_validation = validate_binary_mask(
        mask,
        "final_vehicle_mask",
        min_area_px=max(1, int(min_mask_area) + 1),
        raise_on_large_holes=False,
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
        "mask_validation": mask_validation,
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
    metric_bbox = compute_shared_vehicle_crop_box(mask, pad_px=pad_px, min_size_px=adaptive_min_size, square=roi_square)
    debug["metric_bbox"] = bbox_to_debug_dict(metric_bbox, pad_px, adaptive_min_size, roi_square)
    debug["bbox"] = debug["metric_bbox"]

    # Vorschau- und Metrik-Crops verwenden bewusst dieselbe Box. Keine unabhängigen
    # Ref-/Gen-Bounding-Boxes und kein Nachtrimmen anhand von Alpha oder Pixelinhalt.
    ref_preview_bbox = metric_bbox
    gen_preview_bbox = metric_bbox
    debug["ref_preview_bbox"] = bbox_to_debug_dict(ref_preview_bbox, pad_px, adaptive_min_size, roi_square)
    debug["gen_preview_bbox"] = bbox_to_debug_dict(gen_preview_bbox, pad_px, adaptive_min_size, roi_square)

    if car_mode == "neutralize_crop":
        ref_car, mask_crop = apply_neutralize_crop(ref_norm, mask, metric_bbox, neutral_value=neutral_value)
        gen_car, _ = apply_neutralize_crop(gen_norm, mask, metric_bbox, neutral_value=neutral_value)
        ref_preview, _ = apply_masked_car_crop(ref_norm, mask, metric_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, mask, metric_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    elif car_mode == "weighted_lpips":
        debug["fallback_reason"] = "car_mode=weighted_lpips ist deprecated und nutzt neutralize_crop."
        ref_car, mask_crop = apply_neutralize_crop(ref_norm, mask, metric_bbox, neutral_value=neutral_value)
        gen_car, _ = apply_neutralize_crop(gen_norm, mask, metric_bbox, neutral_value=neutral_value)
        ref_preview, _ = apply_masked_car_crop(ref_norm, mask, metric_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, mask, metric_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    elif car_mode == "roi_crop":
        ref_car = apply_shared_crop(ref_norm, metric_bbox)
        gen_car = apply_shared_crop(gen_norm, metric_bbox)
        mask_crop = apply_shared_crop(mask, metric_bbox).astype(bool)
        ref_preview, _ = apply_masked_car_crop(ref_norm, mask, metric_bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, mask, metric_bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    else:
        raise ValueError("car_mode muss 'neutralize_crop', 'roi_crop' oder 'weighted_lpips' sein")

    validate_same_spatial_size(
        {
            "ref_car_only": ref_preview,
            "gen_car_only": gen_preview,
            "ref_neutral": ref_car,
            "gen_neutral": gen_car,
            "crop_mask": mask_crop,
        },
        "Car-Only-Crop",
    )
    expected_h = int(metric_bbox[3] - metric_bbox[1])
    expected_w = int(metric_bbox[2] - metric_bbox[0])
    actual_h, actual_w = mask_crop.shape[:2]
    if (actual_w, actual_h) != (expected_w, expected_h):
        raise ValueError(
            f"crop_box.json passt nicht zum Crop: Box ergibt {expected_w}x{expected_h}, "
            f"crop_mask ist {actual_w}x{actual_h}."
        )

    stem = Path(ref_path).stem

    if debug_dir:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        save_mask_image(base_mask.astype(bool), debug_path / f"{stem}_raw_vehicle_mask.png")
        save_mask_image(mask, debug_path / f"{stem}_final_vehicle_mask.png")
        save_mask_image(ref_mask.astype(bool), debug_path / f"{stem}_final_vehicle_mask_reference.png")
        save_mask_image(gen_mask.astype(bool), debug_path / f"{stem}_final_vehicle_mask_comparison.png")
        save_mask_image(mask, debug_path / f"{stem}_mask.png")
        save_mask_overlay(ref_norm, mask, debug_path / f"{stem}_final_vehicle_mask_overlay_reference.png")
        save_mask_overlay(gen_norm, mask, debug_path / f"{stem}_final_vehicle_mask_overlay_comparison.png")
        with open(debug_path / f"{stem}_crop_box.json", "w", encoding="utf-8") as fp:
            json.dump(debug["bbox"], fp, indent=2)
        with open(debug_path / f"{stem}_mask_validation.json", "w", encoding="utf-8") as fp:
            json.dump(mask_validation, fp, indent=2, ensure_ascii=False)
        np_to_pil_uint8(ref_car).save(debug_path / f"{stem}_ref_neutral.png")
        np_to_pil_uint8(gen_car).save(debug_path / f"{stem}_gen_neutral.png")
        np_to_pil_uint8(ref_preview).save(debug_path / f"{stem}_aligned_reference_vehicle.png")
        np_to_pil_uint8(gen_preview).save(debug_path / f"{stem}_aligned_comparison_vehicle.png")
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



def load_product_integrity_profile(config_path=None):
    """Lade die Product-Integrity-Konfiguration; nutze Defaults bei fehlender Datei."""
    profile = json.loads(json.dumps(DEFAULT_PRODUCT_INTEGRITY_PROFILE))
    profile["name"] = DEFAULT_PRODUCT_INTEGRITY_PROFILE_NAME
    profile["source"] = "default"
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "configs" / "product_integrity_profile.json"
    config_path = Path(config_path)
    try:
        if config_path.exists():
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            profile = merge_profile_defaults(profile, loaded)
            profile["name"] = loaded.get("name", DEFAULT_PRODUCT_INTEGRITY_PROFILE_NAME)
            profile["source"] = str(config_path)
        else:
            print(f"[WARN] Product-Integrity-Profil fehlt: {config_path}. Nutze Default-Profil.")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Product-Integrity-Profil konnte nicht geladen werden ({exc}). Nutze Default-Profil.")
    return profile


def score_from_error(error, scale):
    return float(np.clip(100.0 * (1.0 - float(error) / max(float(scale), 1e-8)), 0.0, 100.0))


def masked_mean(values, mask=None, default=0.0):
    arr = np.asarray(values, dtype=np.float32)
    if mask is None:
        return float(np.mean(arr))
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return float(default)
    return float(np.mean(arr[valid]))


def bbox_mask_from_fraction(shape, bbox, fractions):
    x0, y0, x1, y1 = bbox
    fx0, fy0, fx1, fy1 = fractions
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    rx0 = int(round(x0 + fx0 * w)); rx1 = int(round(x0 + fx1 * w))
    ry0 = int(round(y0 + fy0 * h)); ry1 = int(round(y0 + fy1 * h))
    mask = np.zeros(shape, dtype=bool)
    mask[max(0, ry0):min(shape[0], ry1), max(0, rx0):min(shape[1], rx1)] = True
    return mask


def validate_zone_against_vehicle(zone, vehicle_mask, name, min_area_ratio=0.001, min_iou_with_vehicle=0.72):
    """Gib nur Zonen frei, die ausreichend groß sind und sichtbar auf dem Fahrzeug liegen."""
    zone = np.asarray(zone, dtype=bool)
    vehicle = np.asarray(vehicle_mask, dtype=bool) if vehicle_mask is not None and np.any(vehicle_mask) else np.ones(zone.shape, dtype=bool)
    clipped = zone & vehicle
    vehicle_area = max(float(np.sum(vehicle)), 1.0)
    original_area = max(float(np.sum(zone)), 1.0)
    area_ratio = float(np.sum(clipped) / vehicle_area)
    iou_with_vehicle = float(np.sum(clipped) / original_area)
    valid = bool(area_ratio >= float(min_area_ratio) and iou_with_vehicle >= float(min_iou_with_vehicle))
    return clipped if valid else np.zeros(zone.shape, dtype=bool), {
        "name": name,
        "valid": valid,
        "area_ratio": area_ratio,
        "iou_with_vehicle": iou_with_vehicle,
    }


def find_wheel_zone_from_mask(vehicle_mask, bbox, side):
    """Leite eine vollständige sichtbare Radzone aus der unteren Fahrzeugkontur ab.

    Die Zone deckt Reifenring, Felge, Speichenstruktur, Radzentrum und den
    direkt angrenzenden Radlauf ab. Sie wird abschließend immer an der finalen
    Fahrzeugmaske geschnitten, damit kein Hintergrund in Detail- oder
    Critical-Component-Scoring eingeht.
    """
    vehicle = np.asarray(vehicle_mask, dtype=bool)
    shape = vehicle.shape
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    # Die bisherige Front-Rad-Heuristik suchte im linken unteren Fahrzeugdrittel.
    # Bei diesem Seitenprofil liegt dort aber Frontschürze/Grill/Kennzeichen.
    # Suche deshalb nur in plausiblen Radfenstern der unteren Fahrzeughälfte:
    # Vorderrad bei ca. 40-60 %, Hinterrad bei ca. 78-98 % der Fahrzeugbreite.
    if side == "front":
        search_fraction = (0.40, 0.50, 0.62, 1.00)
        fallback_cx = x0 + 0.50 * bw
    else:
        search_fraction = (0.78, 0.50, 0.98, 1.00)
        fallback_cx = x0 + 0.88 * bw
    search = bbox_mask_from_fraction(shape, bbox, search_fraction)
    candidates = vehicle & search
    if not np.any(candidates):
        return np.zeros(shape, dtype=bool)

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    lower_cut = y0 + int(0.58 * bh)
    lower = candidates & (yy >= lower_cut)
    ys, xs = np.where(lower if np.any(lower) else candidates)
    if len(xs):
        # Gewichte die Mitte robuster als den Median, aber klammere sie hart in
        # das zulässige Radfenster. So kann das Vorderrad nicht mehr in den
        # Grill-/Frontschürzenbereich (10-35 %) zurückfallen.
        cx = int(round(np.mean(xs)))
    else:
        cx = int(round(fallback_cx))
    min_cx = int(round(x0 + search_fraction[0] * bw))
    max_cx = int(round(x0 + search_fraction[2] * bw))
    cx = int(np.clip(cx, min_cx, max_cx))
    cy = int(round(np.percentile(ys, 58))) if len(ys) else int(round(y0 + 0.76 * bh))

    # Verwende bewusst großzügige Rad-Ellipsen: Segmentierungen schneiden dunkle
    # Reifen und offene Felgen häufig aus. Die anschließende Masken-Clip-Operation
    # verhindert Hintergrundanteile, behält aber die gesamte sichtbare Radfläche.
    rx = max(5, int(round(0.115 * bw)))
    ry = max(5, int(round(0.225 * bh)))
    wheel_disc = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
    rim_core = (((xx - cx) / max(rx * 0.68, 1.0)) ** 2 + ((yy - cy) / max(ry * 0.68, 1.0)) ** 2) <= 1.0
    tire_ring = wheel_disc & ~((((xx - cx) / max(rx * 0.50, 1.0)) ** 2 + ((yy - cy) / max(ry * 0.50, 1.0)) ** 2) <= 1.0)
    arch = ((((xx - cx) / max(rx * 1.12, 1.0)) ** 2 + ((yy - (cy - 0.18 * ry)) / max(ry * 1.05, 1.0)) ** 2) <= 1.0) & (yy <= cy + 0.35 * ry)
    visible_vehicle = dilation(vehicle, disk(3))
    zone = (wheel_disc | rim_core | tire_ring | arch) & visible_vehicle & vehicle

    # Wenn die Maske im Suchfenster zu wenig Radfläche enthält, nutze einen
    # positionsbasierten Fallback, aber bleibe weiterhin strikt in der finalen
    # Fahrzeugmaske und im plausiblen Radfenster.
    min_area = max(12, int(0.0025 * np.sum(vehicle)))
    if int(np.sum(zone)) < min_area:
        fallback_disc = (((xx - int(round(fallback_cx))) / rx) ** 2 + ((yy - cy) / ry) ** 2) <= 1.0
        zone = fallback_disc & visible_vehicle & vehicle & search
    return zone & search


def build_detail_zone_masks(car_mask, fallback_shape, ref=None, gen=None, glass_masks=None):
    """Erzeuge detailnahe Fahrzeugzonen aus Maske, Kanten und Fahrzeug-Bounding-Box."""
    scope = np.asarray(car_mask, dtype=bool) if car_mask is not None and np.any(car_mask) else np.ones(fallback_shape, dtype=bool)
    bbox = compute_mask_roi_bbox(scope, fallback_shape)
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    h, w = fallback_shape
    yy, xx = np.mgrid[0:h, 0:w]
    bx = (xx - x0) / bw
    by = (yy - y0) / bh

    if ref is not None and gen is not None:
        ref_edge = compute_edge_strength(ref)
        gen_edge = compute_edge_strength(gen)
        stable_edge = np.clip(np.minimum(ref_edge, gen_edge), 0.0, 1.0)
        edge_delta = np.abs(ref_edge - gen_edge)
        scoped_edge_values = stable_edge[scope]
        edge_threshold = max(0.04, float(np.percentile(scoped_edge_values, 66)) if scoped_edge_values.size else 0.04)
        detail_edges = (stable_edge >= edge_threshold) & (edge_delta <= 0.16) & scope
        changed_detail_edges = (edge_delta > 0.08) & scope
    else:
        detail_edges = scope.copy()
        changed_detail_edges = np.zeros(fallback_shape, dtype=bool)

    glass_interior = np.zeros(fallback_shape, dtype=bool)
    glass_contour = np.zeros(fallback_shape, dtype=bool)
    if glass_masks:
        glass_interior = np.asarray(glass_masks.get("interior", glass_interior), dtype=bool)
        glass_contour = np.asarray(glass_masks.get("contour", glass_contour), dtype=bool)
    product_edges = detail_edges & ~glass_interior

    silhouette = (dilation(scope, disk(1)) & ~erosion(scope, disk(2))) & scope
    front_wheel = find_wheel_zone_from_mask(scope, bbox, "front")
    rear_wheel = find_wheel_zone_from_mask(scope, bbox, "rear")
    wheel_structure_seed = dilation(product_edges | changed_detail_edges | silhouette, disk(3))
    wheels_tires = (front_wheel | rear_wheel) & wheel_structure_seed

    front_rear_band = (
        bbox_mask_from_fraction(fallback_shape, bbox, (0.00, 0.22, 0.34, 0.84))
        | bbox_mask_from_fraction(fallback_shape, bbox, (0.66, 0.22, 1.00, 0.86))
    ) & scope
    # Nicht nur Kantenpixel verwenden: Front-/Heckkontur, Grill/Stern,
    # Scheinwerfer, Rücklichtbereich und seitliche Hecklinie sollen sicher in
    # der Detailmaske liegen. Durch den Schnitt mit ``scope`` bleibt der
    # Hintergrund ausgeschlossen.
    front_rear_core = front_rear_band & (((bx <= 0.02) | (bx >= 0.98)) & (by >= 0.40) & (by <= 0.70))
    front_rear = front_rear_band & (product_edges | changed_detail_edges | silhouette | front_rear_core)

    window_band = (by >= 0.06) & (by <= 0.58) & (bx >= 0.06) & (bx <= 0.94) & scope
    changed_window_edges = changed_detail_edges & window_band & (glass_contour | (by <= 0.28) | (silhouette & (by <= 0.36)))
    window_line_edges = window_band & product_edges & (by <= 0.26)
    window_line = (glass_contour | changed_window_edges | window_line_edges | (silhouette & (by <= 0.34))) & ~glass_interior
    body_band = (by >= 0.38) & (by <= 0.74) & (bx >= 0.08) & (bx <= 0.94) & scope & ~wheels_tires
    body_lines = body_band & (product_edges | changed_detail_edges)
    center_band = bbox_mask_from_fraction(fallback_shape, bbox, (0.00, 0.28, 0.46, 0.78)) & scope
    center_front_core = center_band & (bx <= 0.03) & (by >= 0.44) & (by <= 0.64)
    center_grill_emblem = center_band & (product_edges | changed_detail_edges | center_front_core)

    raw_zones = {
        "silhouette": silhouette & ~glass_interior & ~wheels_tires,
        "front_rear": front_rear & ~glass_interior & ~wheels_tires,
        "wheels_tires": wheels_tires & ~glass_interior,
        "window_line": window_line & ~glass_interior,
        "body_lines": body_lines & ~glass_interior,
        "center_grill_emblem": center_grill_emblem & ~glass_interior,
    }
    zones = {}
    for name, zone in raw_zones.items():
        zones[name], _ = validate_zone_against_vehicle(zone, scope, name, min_area_ratio=0.0005 if name == "center_grill_emblem" else 0.001)
    return zones

def compute_structure_only_score(ref, gen, mask=None, debug_dir=None, stem="pair"):
    """Vergleiche graue Kantenbilder ausschließlich innerhalb der Fahrzeugmaske."""
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        metric_mask = np.ones(ref.shape[:2], dtype=bool)
    structure_mask = erosion(metric_mask, disk(1)) if np.any(metric_mask) else metric_mask
    if not np.any(structure_mask):
        structure_mask = metric_mask

    ref_gray_raw = color.rgb2gray(ref).astype(np.float32)
    gen_gray_raw = color.rgb2gray(gen).astype(np.float32)
    ref_neutral = np.full(ref_gray_raw.shape, masked_mean(ref_gray_raw, structure_mask, default=0.5), dtype=np.float32)
    gen_neutral = np.full(gen_gray_raw.shape, masked_mean(gen_gray_raw, structure_mask, default=0.5), dtype=np.float32)
    ref_neutral[metric_mask] = ref_gray_raw[metric_mask]
    gen_neutral[metric_mask] = gen_gray_raw[metric_mask]

    ref_gray = gaussian(ref_neutral, sigma=1.0, preserve_range=True)
    gen_gray = gaussian(gen_neutral, sigma=1.0, preserve_range=True)
    ref_edge = np.clip(sobel(ref_gray), 0.0, 1.0) * structure_mask.astype(np.float32)
    gen_edge = np.clip(sobel(gen_gray), 0.0, 1.0) * structure_mask.astype(np.float32)
    diff = np.abs(ref_edge - gen_edge) * structure_mask.astype(np.float32)
    score = score_from_error(masked_mean(diff, structure_mask), 0.18)
    paths = {}
    if debug_dir:
        debug_path = Path(debug_dir); debug_path.mkdir(parents=True, exist_ok=True)
        paths["structure_ref_edges"] = str(debug_path / f"{stem}_structure_ref_edges.png")
        paths["structure_gen_edges"] = str(debug_path / f"{stem}_structure_gen_edges.png")
        paths["structure_diff"] = str(debug_path / f"{stem}_structure_diff.png")
        save_weight_debug_map(ref_edge, Path(paths["structure_ref_edges"]))
        save_weight_debug_map(gen_edge, Path(paths["structure_gen_edges"]))
        save_weight_debug_map(diff, Path(paths["structure_diff"]))
    structure_area_ratio = float(np.sum(structure_mask) / max(float(structure_mask.size), 1.0))
    return {"score": score, "diff_map": diff, "ref_edge": ref_edge, "gen_edge": gen_edge, "debug_paths": paths, "structure_masked_area_ratio": structure_area_ratio}


def compute_detail_zones_score(ref, gen, mask=None, profile=None, structure_debug=None, debug_dir=None, stem="pair"):
    """Prüfe produktkritische Fallback-Zonen stärker als große Lackflächen."""
    profile = profile or DEFAULT_PRODUCT_INTEGRITY_PROFILE
    metric_mask = prepare_metric_mask(mask, ref, gen)
    glass_masks = build_glass_region_masks(ref, gen, car_mask=metric_mask) if metric_mask is not None else None
    zones = build_detail_zone_masks(metric_mask, ref.shape[:2], ref=ref, gen=gen, glass_masks=glass_masks)
    if structure_debug:
        edge_diff = structure_debug["diff_map"].copy()
        persistent_edges = (structure_debug["ref_edge"] > 0.03) & (structure_debug["gen_edge"] > 0.03)
        edge_diff = np.where(persistent_edges, edge_diff * 0.35, edge_diff)
    else:
        edge_diff = np.abs(sobel(color.rgb2gray(ref)) - sobel(color.rgb2gray(gen)))
    weights_cfg = profile.get("detail_zone_weights", {})
    zone_scores = {}
    weighted_sum = 0.0; weight_sum = 0.0
    for name, zone in zones.items():
        if not np.any(zone):
            continue
        zone_score = score_from_error(masked_mean(edge_diff, zone), 0.16)
        zone_scores[name] = zone_score
        weight = float(weights_cfg.get(name, 1.0))
        weighted_sum += zone_score * weight; weight_sum += weight
    score = float(weighted_sum / weight_sum) if weight_sum else 100.0
    paths = {}
    if debug_dir:
        merged = np.zeros(ref.shape[:2], dtype=np.float32)
        for i, zone in enumerate(zones.values(), start=1):
            merged = np.maximum(merged, zone.astype(np.float32) * (i / max(len(zones), 1)))
        paths["detail_zones_mask"] = str(Path(debug_dir) / f"{stem}_detail_zones_mask.png")
        save_weight_debug_map(merged, Path(paths["detail_zones_mask"]))
    merged_zone = np.zeros(ref.shape[:2], dtype=bool)
    for zone in zones.values():
        merged_zone |= np.asarray(zone, dtype=bool)
    metric_area = max(float(np.sum(metric_mask)) if metric_mask is not None else float(ref.shape[0] * ref.shape[1]), 1.0)
    detail_zone_area_ratio = float(np.sum(merged_zone) / metric_area)
    return {"score": score, "zone_scores": zone_scores, "zones": zones, "debug_paths": paths, "detail_zone_area_ratio": detail_zone_area_ratio}


def compute_color_reflection_score(ref, gen, mask=None, edge_diff=None, debug_dir=None, stem="pair", profile=None):
    """Bewerte Farb-/Lichtänderungen robust innerhalb der Fahrzeugmaske.

    Korrigiere zuerst globale LAB-Licht-/Farbstimmung auf stabilen Lackflächen.
    Bewerte anschließend den CIEDE2000-Restfehler flächengewichtet und robust:
    Lack bleibt maßgeblich, Glasinnenflächen werden stark gedämpft, Glaskonturen
    moderat berücksichtigt und harte Kanten/Produktbauteile bleiben durch die
    separaten Struktur-, Detail- und Komponenten-Scores geschützt.
    """
    profile = profile or DEFAULT_PRODUCT_INTEGRITY_PROFILE
    reflection_cfg = profile.get("reflection_tolerance", {})
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        metric_mask = np.ones(ref.shape[:2], dtype=bool)

    vehicle_scope = metric_mask.astype(bool)
    glass_masks = build_glass_region_masks(ref, gen, car_mask=vehicle_scope)
    glass_interior = glass_masks.get("interior", np.zeros(ref.shape[:2], dtype=bool)) & vehicle_scope
    glass_contour = glass_masks.get("contour", np.zeros(ref.shape[:2], dtype=bool)) & vehicle_scope
    glass_surface = glass_masks.get("surface", np.zeros(ref.shape[:2], dtype=bool)) & vehicle_scope
    paint_mask = vehicle_scope & ~glass_surface
    critical_zones = build_critical_component_zones(vehicle_scope, ref.shape[:2])
    critical_component_mask = np.zeros(ref.shape[:2], dtype=bool)
    for zone in critical_zones.values():
        critical_component_mask |= np.asarray(zone, dtype=bool)
    critical_component_mask &= vehicle_scope

    ref_lab = color.rgb2lab(ref).astype(np.float32)
    gen_lab = color.rgb2lab(gen).astype(np.float32)
    lab_delta = gen_lab - ref_lab
    raw_delta_e = color.deltaE_ciede2000(ref_lab, gen_lab).astype(np.float32)

    # Nutze ruhige Lackflächen für die globale Korrektur. Damit erklären ein
    # anderer Hintergrund, Weißabgleich oder Gesamthelligkeit nicht sofort einen
    # Produktfehler. Wenn keine ruhige Lackfläche vorliegt, fällt die Methode auf
    # die komplette Fahrzeugmaske zurück, niemals auf den Hintergrund.
    if edge_diff is None:
        edge_diff = np.abs(sobel(color.rgb2gray(ref)) - sobel(color.rgb2gray(gen)))
    stable_paint = paint_mask & (np.asarray(edge_diff) <= float(reflection_cfg.get("global_correction_edge_max", 0.10)))
    calibration_mask = stable_paint if np.sum(stable_paint) >= 64 else (paint_mask if np.any(paint_mask) else vehicle_scope)
    median_shift = np.median(lab_delta[calibration_mask], axis=0) if np.any(calibration_mask) else np.zeros(3, dtype=np.float32)
    residual_lab_delta = lab_delta - median_shift.reshape(1, 1, 3)
    corrected_lab = ref_lab + residual_lab_delta
    corrected_delta_e = color.deltaE_ciede2000(ref_lab, corrected_lab).astype(np.float32)

    raw_error = np.clip(raw_delta_e / 100.0, 0.0, 1.0).astype(np.float32)
    corrected_error = np.clip(corrected_delta_e / 100.0, 0.0, 1.0).astype(np.float32)

    # Glatte Reflexionsflächen dürfen sichtbar bleiben, aber den Score nicht
    # dominieren. Kanten werden für Color/Reflection gedämpft, weil sie in den
    # Struktur-/Detail-Scores fachlich strenger bewertet werden.
    low_edge_weight = np.clip(1.0 - (np.asarray(edge_diff, dtype=np.float32) / 0.20), 0.35, 1.0).astype(np.float32)
    area_weight = np.zeros(ref.shape[:2], dtype=np.float32)
    area_weight[paint_mask] = float(reflection_cfg.get("paint_color_weight", 1.0))
    area_weight[glass_contour] = float(reflection_cfg.get("glass_contour_color_weight", 0.45))
    area_weight[glass_interior] = float(reflection_cfg.get("glass_interior_color_weight", 0.12))
    # Kritische Komponenten werden im Color-Score nicht künstlich gehärtet; sie
    # laufen über Detail-/Komponentenregeln. Ein leichter Floor verhindert aber,
    # dass echte Material-/Farbänderungen dort komplett verschwinden.
    area_weight[critical_component_mask & paint_mask] = np.maximum(area_weight[critical_component_mask & paint_mask], 0.85)
    effective_mask = vehicle_scope & (area_weight > 0)

    weighted_error_map = corrected_error * low_edge_weight * area_weight * vehicle_scope.astype(np.float32)
    raw_weighted_error_map = raw_error * low_edge_weight * area_weight * vehicle_scope.astype(np.float32)
    weighted_values = weighted_error_map[effective_mask]
    raw_values = raw_error[vehicle_scope]
    corrected_values = corrected_error[vehicle_scope]

    if weighted_values.size:
        denom = max(float(np.sum(area_weight[effective_mask])), 1e-6)
        color_reflection_weighted_mean = float(np.sum(weighted_error_map[effective_mask]) / denom)
        raw_weighted_mean = float(np.sum(raw_weighted_error_map[effective_mask]) / denom)
        quantiles = {q: float(np.quantile(weighted_values, q / 100.0)) for q in (50, 75, 90, 95, 98, 99)}
        raw_quantiles = {q: float(np.quantile(raw_values, q / 100.0)) for q in (90, 95)} if raw_values.size else {90: 0.0, 95: 0.0}
        corrected_quantiles = {q: float(np.quantile(corrected_values, q / 100.0)) for q in (90, 95)} if corrected_values.size else {90: 0.0, 95: 0.0}
    else:
        color_reflection_weighted_mean = raw_weighted_mean = 0.0
        quantiles = {q: 0.0 for q in (50, 75, 90, 95, 98, 99)}
        raw_quantiles = {90: 0.0, 95: 0.0}
        corrected_quantiles = {90: 0.0, 95: 0.0}

    # Ursache des 4%-Falls war die Kombination aus sehr kleiner Skala (0.08) und
    # einem harten max(P95, P98, P99). Nutze stattdessen eine robuste Mischung:
    # Mittelwert dominiert, P90 schützt vor größeren Bereichen, P95 wirkt nur
    # gedämpft gegen echte flächige Abweichungen.
    final_error = max(
        color_reflection_weighted_mean,
        (0.70 * color_reflection_weighted_mean) + (0.30 * quantiles[90]),
        (0.60 * color_reflection_weighted_mean) + (0.25 * quantiles[90]) + (0.15 * quantiles[95]),
    )
    error_scale = max(float(reflection_cfg.get("color_score_error_scale", 0.22)), 1e-8)
    score_before_clipping = 100.0 * (1.0 - float(final_error) / error_scale)
    score = float(np.clip(score_before_clipping, 0.0, 100.0))

    metric_area = max(float(np.sum(vehicle_scope)), 1.0)
    background_leakage_area_px = int(np.sum(effective_mask & ~vehicle_scope))
    def region_error(region):
        region = np.asarray(region, dtype=bool) & effective_mask
        if not np.any(region):
            return 0.0
        return float(np.sum(weighted_error_map[region]) / max(float(np.sum(area_weight[region])), 1e-6))

    debug = {
        "mask": "final_vehicle_mask_only_with_glass_and_reflection_downweighting",
        "active_mask_area_px": int(np.sum(effective_mask)),
        "background_leakage_area_px": background_leakage_area_px,
        "glass_interior_area_px": int(np.sum(glass_interior)),
        "glass_contour_area_px": int(np.sum(glass_contour)),
        "paint_area_px": int(np.sum(paint_mask)),
        "critical_component_area_px": int(np.sum(critical_component_mask)),
        "raw_delta_e_mean": float(np.mean(raw_delta_e[vehicle_scope])) if np.any(vehicle_scope) else 0.0,
        "raw_delta_e_p90": float(raw_quantiles[90] * 100.0),
        "raw_delta_e_p95": float(raw_quantiles[95] * 100.0),
        "corrected_delta_e_mean": float(np.mean(corrected_delta_e[vehicle_scope])) if np.any(vehicle_scope) else 0.0,
        "corrected_delta_e_p90": float(corrected_quantiles[90] * 100.0),
        "corrected_delta_e_p95": float(corrected_quantiles[95] * 100.0),
        "weighted_color_error_mean": color_reflection_weighted_mean,
        "weighted_color_error_p90": quantiles[90],
        "weighted_color_error_p95": quantiles[95],
        "final_color_reflection_error": float(final_error),
        "color_reflection_error_scale": float(error_scale),
        "color_reflection_score_before_clipping": float(score_before_clipping),
        "color_reflection_score_after_clipping": float(score),
        "color_reflection_interpretation": "Globale LAB-Licht-/Farbstimmung wurde innerhalb der Fahrzeugmaske entfernt; der Score basiert auf robusten, flächengewichteten Restfehlern.",
        "window_area_ratio": float(np.sum(glass_surface) / metric_area),
        "paint_area_ratio": float(np.sum(paint_mask) / metric_area),
        "excluded_area_ratio": float(1.0 - (np.sum(vehicle_scope) / max(float(ref.shape[0] * ref.shape[1]), 1.0))),
        "mean_color_difference": color_reflection_weighted_mean,
        "median_weighted_color_difference": quantiles[50],
        "robust_color_difference_quantile": quantiles[90],
        "global_lab_shift_removed": [float(x) for x in median_shift],
        "score_before_weighting": score,
        "color_reflection_error_before_calibration": raw_weighted_mean,
        "color_reflection_error_after_calibration": color_reflection_weighted_mean,
        "color_reflection_weighted_mean": color_reflection_weighted_mean,
        "color_reflection_p50": quantiles[50],
        "color_reflection_p75": quantiles[75],
        "color_reflection_p90": quantiles[90],
        "color_reflection_p95": quantiles[95],
        "color_reflection_p98": quantiles[98],
        "color_reflection_p99": quantiles[99],
        "glass_interior_error": region_error(glass_interior),
        "glass_contour_error": region_error(glass_contour),
        "paint_error": region_error(paint_mask),
        "area_weight_map_used_in_final_error": True,
    }

    paths = {}
    if debug_dir:
        paths["reflection_color_difference"] = str(Path(debug_dir) / f"{stem}_reflection_color_difference.png")
        paths["color_reflection_metric_mask"] = str(Path(debug_dir) / f"{stem}_color_reflection_metric_mask.png")
        paths["color_reflection_area_weights"] = str(Path(debug_dir) / f"{stem}_color_reflection_area_weights.png")
        save_weight_debug_map(weighted_error_map, Path(paths["reflection_color_difference"]))
        save_weight_debug_map(vehicle_scope.astype(np.float32), Path(paths["color_reflection_metric_mask"]))
        save_weight_debug_map(area_weight, Path(paths["color_reflection_area_weights"]))
    return {"score": score, "map": weighted_error_map, "debug_paths": paths, "debug": debug}

def compute_component_product_scores(ref, gen, mask=None, structure_debug=None, lpips_component_map=None, debug_dir=None, stem="pair"):
    """Bewerte harte Produktbauteile lokal, damit Details nicht im Gesamtscore verschwinden."""
    metric_mask = prepare_metric_mask(mask, ref, gen)
    shape = ref.shape[:2]
    scope = metric_mask if metric_mask is not None else np.ones(shape, dtype=bool)
    zones = build_critical_component_zones(metric_mask, shape)

    ref_gray = color.rgb2gray(ref).astype(np.float32)
    gen_gray = color.rgb2gray(gen).astype(np.float32)
    gray_diff = np.abs(ref_gray - gen_gray)
    ref_edge = np.clip(sobel(ref_gray), 0.0, 1.0).astype(np.float32)
    gen_edge = np.clip(sobel(gen_gray), 0.0, 1.0).astype(np.float32)
    edge_delta = np.abs(ref_edge - gen_edge)
    if structure_debug:
        structure_diff = np.asarray(structure_debug["diff_map"], dtype=np.float32)
    else:
        structure_diff = np.abs(sobel(ref_gray) - sobel(gen_gray)).astype(np.float32)
    if lpips_component_map is not None:
        lpips_local = resize_float_map_to_shape(
            np.asarray(lpips_component_map, dtype=np.float32),
            shape,
            resample=Image.Resampling.BILINEAR,
        )
        lpips_local = lpips_local / (float(np.percentile(lpips_local, 98)) + 1e-8)
        lpips_local = np.clip(lpips_local, 0.0, 1.0)
    else:
        lpips_local = np.zeros(shape, dtype=np.float32)

    # Harte Bauteile nutzen bewusst ungedämpfte lokale Signale. Reflection-Downweighting
    # darf Scheinwerfer, Lichtsignatur, Räder/Felgen/Reifen, Grill und Emblem nicht entschärfen.
    component_diff_map = np.clip((0.55 * structure_diff) + (0.35 * edge_delta) + (0.20 * gray_diff) + (0.15 * lpips_local), 0.0, 1.0)
    wheel_component_diff_map = np.clip((0.70 * structure_diff) + (0.35 * edge_delta) + (0.05 * gray_diff) + (0.15 * lpips_local), 0.0, 1.0)
    component_diff_map = component_diff_map * scope.astype(np.float32)
    wheel_component_diff_map = wheel_component_diff_map * scope.astype(np.float32)
    edge_presence = np.maximum(ref_edge, gen_edge)

    component_scores = {}
    component_diffs = {}
    attribution = np.zeros(shape, dtype=np.float32)
    zone_debug = np.zeros(shape, dtype=np.float32)
    zone_order = list(zones.keys())
    for idx, name in enumerate(zone_order, start=1):
        zone = zones[name]
        if not np.any(zone):
            local_diff = 0.0
        else:
            detail_pixels = zone & (edge_presence >= max(0.025, float(np.percentile(edge_presence[zone], 45))))
            eval_zone = detail_pixels if np.sum(detail_pixels) >= max(8, int(np.sum(zone) * 0.08)) else zone
            source_diff_map = wheel_component_diff_map if name in {"front_wheel", "rear_wheel", "wheel_tire"} else component_diff_map
            local_diff = masked_mean(source_diff_map, eval_zone, default=0.0)
        scale = 0.085 if name in {"front_wheel", "rear_wheel", "wheel_tire"} else (0.105 if name in {"headlight", "front_light_signature"} else 0.13)
        component_diffs[name] = float(local_diff)
        component_scores[f"{name}_score"] = score_from_error(local_diff, scale)
        source_diff_map = wheel_component_diff_map if name in {"front_wheel", "rear_wheel", "wheel_tire"} else component_diff_map
        attribution = np.maximum(attribution, np.where(zone, source_diff_map, 0.0))
        zone_debug = np.maximum(zone_debug, zone.astype(np.float32) * (idx / max(len(zone_order), 1)))

    critical_component_mask = (
        zones["headlight"]
        | zones["front_light_signature"]
        | zones["front_wheel"]
        | zones["rear_wheel"]
        | zones["wheel_tire"]
        | zones["grille"]
        | zones["emblem"]
        | zones["rear_light"]
        | zones["rear_contour"]
    )

    paths = {}
    if debug_dir:
        debug_path = Path(debug_dir); debug_path.mkdir(parents=True, exist_ok=True)
        paths["critical_component_zones"] = str(debug_path / f"{stem}_critical_component_zones.png")
        paths["wheel_zone_front"] = str(debug_path / f"{stem}_wheel_zone_front.png")
        paths["wheel_zone_rear"] = str(debug_path / f"{stem}_wheel_zone_rear.png")
        paths["front_wheel_zone"] = str(debug_path / f"{stem}_front_wheel_zone.png")
        paths["rear_wheel_zone"] = str(debug_path / f"{stem}_rear_wheel_zone.png")
        paths["wheel_zone_combined"] = str(debug_path / f"{stem}_wheel_zone_combined.png")
        paths["wheel_zone_diff"] = str(debug_path / f"{stem}_wheel_zone_diff.png")
        paths["front_wheel_score_map"] = str(debug_path / f"{stem}_front_wheel_score_map.png")
        paths["rear_wheel_score_map"] = str(debug_path / f"{stem}_rear_wheel_score_map.png")
        paths["wheel_score_heatmap"] = str(debug_path / f"{stem}_wheel_score_heatmap.png")
        paths["critical_component_score_map"] = str(debug_path / f"{stem}_critical_component_score_map.png")
        paths["grille_zone"] = str(debug_path / f"{stem}_grille_zone.png")
        paths["headlight_zone"] = str(debug_path / f"{stem}_headlight_zone.png")
        paths["window_zone"] = str(debug_path / f"{stem}_window_zone.png")
        paths["headlight_zone_diff"] = str(debug_path / f"{stem}_headlight_zone_diff.png")
        paths["wheel_tire_zone_diff"] = str(debug_path / f"{stem}_wheel_tire_zone_diff.png")
        paths["grille_zone_diff"] = str(debug_path / f"{stem}_grille_zone_diff.png")
        paths["component_attribution_map"] = str(debug_path / f"{stem}_component_attribution_map.png")
        save_weight_debug_map(zone_debug, Path(paths["critical_component_zones"]))
        save_mask_image(zones["front_wheel"], Path(paths["wheel_zone_front"]))
        save_mask_image(zones["rear_wheel"], Path(paths["wheel_zone_rear"]))
        save_mask_image(zones["front_wheel"], Path(paths["front_wheel_zone"]))
        save_mask_image(zones["rear_wheel"], Path(paths["rear_wheel_zone"]))
        save_mask_image(zones["front_wheel"] | zones["rear_wheel"] | zones["wheel_tire"], Path(paths["wheel_zone_combined"]))
        save_mask_image(zones["grille"], Path(paths["grille_zone"]))
        save_mask_image(zones["headlight"] | zones["front_light_signature"], Path(paths["headlight_zone"]))
        save_mask_image(zones["window_line"] | zones.get("roof_pillar", np.zeros(shape, dtype=bool)), Path(paths["window_zone"]))
        save_weight_debug_map(np.where(zones["headlight"] | zones["front_light_signature"], component_diff_map, 0.0), Path(paths["headlight_zone_diff"]))
        wheel_combined = zones["front_wheel"] | zones["rear_wheel"] | zones["wheel_tire"]
        save_weight_debug_map(np.where(zones["wheel_tire"], component_diff_map, 0.0), Path(paths["wheel_tire_zone_diff"]))
        save_weight_debug_map(np.where(wheel_combined, component_diff_map, 0.0), Path(paths["wheel_zone_diff"]))
        save_weight_debug_map(np.where(zones["front_wheel"], wheel_component_diff_map, 0.0), Path(paths["front_wheel_score_map"]))
        save_weight_debug_map(np.where(zones["rear_wheel"], wheel_component_diff_map, 0.0), Path(paths["rear_wheel_score_map"]))
        save_weight_debug_map(np.where(wheel_combined, wheel_component_diff_map, 0.0), Path(paths["wheel_score_heatmap"]))
        critical_score_map = np.maximum(
            np.where(critical_component_mask, component_diff_map, 0.0),
            np.where(wheel_combined, wheel_component_diff_map, 0.0),
        )
        save_weight_debug_map(critical_score_map, Path(paths["critical_component_score_map"]))
        save_weight_debug_map(np.where(zones["grille"], component_diff_map, 0.0), Path(paths["grille_zone_diff"]))
        save_weight_debug_map(attribution, Path(paths["component_attribution_map"]))

    return {
        "scores": component_scores,
        "diffs": component_diffs,
        "zones": zones,
        "debug_paths": paths,
        "component_diff_map": component_diff_map,
        "component_attribution_map": attribution,
        "critical_component_mask": critical_component_mask,
    }


def normalize_lpips_car_only_similarity_percent(lpips_car_only_value, fallback_similarity_pct):
    """Gib immer eine Similarity in Prozent zurück; LPIPS-Distanzen (0..1) werden invertiert."""
    if lpips_car_only_value is None:
        return float(fallback_similarity_pct)

    numeric_value = float(lpips_car_only_value)
    if 0.0 <= numeric_value <= 1.0:
        return convert_lpips_to_similarity_percent(numeric_value)
    return float(np.clip(numeric_value, 0.0, 100.0))


def compute_product_integrity_scores(ref, gen, car_mask=None, car_only_lpips_score=None, profile=None, debug_dir=None, stem="pair", mask_metrics=None, lpips_component_map=None):
    """Führe Structure, Detail-Zones, Color/Reflection und Car-only-LPIPS zur Produktintegrität zusammen."""
    profile = profile or DEFAULT_PRODUCT_INTEGRITY_PROFILE
    scope = car_mask if car_mask is not None and np.any(car_mask) else None
    structure = compute_structure_only_score(ref, gen, mask=scope, debug_dir=debug_dir, stem=stem)
    detail = compute_detail_zones_score(ref, gen, mask=scope, profile=profile, structure_debug=structure, debug_dir=debug_dir, stem=stem)
    color_score = compute_color_reflection_score(ref, gen, mask=scope, edge_diff=structure["diff_map"], debug_dir=debug_dir, stem=stem, profile=profile)
    components = compute_component_product_scores(ref, gen, mask=scope, structure_debug=structure, lpips_component_map=lpips_component_map, debug_dir=debug_dir, stem=stem)
    component_values = components["scores"]
    component_diffs = components["diffs"]
    lpips_car_only_raw = None if car_only_lpips_score is None else float(car_only_lpips_score)
    lpips_car_only_similarity_pct = normalize_lpips_car_only_similarity_percent(
        car_only_lpips_score,
        fallback_similarity_pct=structure["score"],
    )
    component_scores = {
        "structure_similarity_pct": structure["score"],
        "detail_zones_similarity_pct": detail["score"],
        "color_reflection_similarity_pct": color_score["score"],
        "car_only_lpips_score": lpips_car_only_similarity_pct,
    }
    profile_weight_key_by_component = {
        "structure_similarity_pct": "structure_only_score",
        "detail_zones_similarity_pct": "detail_zones_score",
        "color_reflection_similarity_pct": "color_reflection_score",
        "car_only_lpips_score": "car_only_lpips_score",
    }
    enabled = profile.get("enabled_components", {})
    weights = profile.get("weights", {})
    total = 0.0; denom = 0.0
    color_reflection_weight = 0.0
    color_reflection_weighted_score = float(color_score["score"])
    for key, value in component_scores.items():
        profile_key = profile_weight_key_by_component[key]
        if enabled.get(profile_key, True):
            weight = float(weights.get(profile_key, 0.0))
            if profile_key == "color_reflection_score":
                color_reflection_weight = weight
                color_reflection_weighted_score = float(value)
            total += float(value) * weight
            denom += weight
    base_final_product_integrity_score_pct = float(total / denom) if denom else float(np.mean(list(component_scores.values())))
    total_without_color = total - (color_reflection_weighted_score * color_reflection_weight)
    denom_without_color = denom - color_reflection_weight
    final_without_color_score_pct = float(total_without_color / denom_without_color) if denom_without_color > 0 else base_final_product_integrity_score_pct
    raw_color_reflection_influence = final_without_color_score_pct - base_final_product_integrity_score_pct
    score_adjustments = []
    reflection_cfg = profile.get("reflection_tolerance", {})
    stable_for_color_cap = (
        structure["score"] >= float(reflection_cfg.get("stable_structure_for_color_cap", 90.0))
        and detail["score"] >= float(reflection_cfg.get("stable_detail_for_color_cap", 92.0))
    )
    if stable_for_color_cap:
        max_color_penalty = float(reflection_cfg.get("color_score_penalty_cap_when_structure_stable", 1.2))
        capped_base = max(base_final_product_integrity_score_pct, final_without_color_score_pct - max_color_penalty)
        if capped_base != base_final_product_integrity_score_pct:
            score_adjustments.append({
                "type": "color_reflection_penalty_cap",
                "before": float(base_final_product_integrity_score_pct),
                "after": float(capped_base),
                "reason": "Structure und Detail stabil; Color-/Reflection-Einfluss begrenzt.",
            })
        base_final_product_integrity_score_pct = capped_base
    final_product_integrity_score_pct = base_final_product_integrity_score_pct
    thresholds = profile.get("thresholds", {})
    contour_cfg = profile.get("contour_warning", {})
    mask_metrics = mask_metrics or {}
    mask_iou = mask_metrics.get("mask_iou")
    mask_dice = mask_metrics.get("mask_dice")
    hausdorff_norm = mask_metrics.get("hausdorff_norm")
    centroid_norm = mask_metrics.get("centroid_distance_norm")
    mask_area_ratio = mask_metrics.get("mask_area_ratio")
    high_mask_overlap = (
        mask_iou is not None
        and mask_dice is not None
        and float(mask_iou) >= float(contour_cfg.get("high_iou_min", 0.98))
        and float(mask_dice) >= float(contour_cfg.get("high_dice_min", 0.99))
    )
    contour_geometric_relevant = (
        (hausdorff_norm is not None and float(hausdorff_norm) > float(contour_cfg.get("max_hausdorff_norm", 0.018)))
        or (centroid_norm is not None and float(centroid_norm) > float(contour_cfg.get("max_centroid_norm", 0.006)))
        or (mask_area_ratio is not None and abs(float(mask_area_ratio) - 1.0) > float(contour_cfg.get("max_area_ratio_delta", 0.025)))
    )
    findings = {}
    critical = []
    tolerated = []
    contour_warning_reason = "Keine relevante Konturabweichung erkannt."
    component_findings = []
    critical_component_findings = []
    zone_scores = detail["zone_scores"]
    zone_labels = {
        "silhouette": "Abweichung an Fahrzeugkontur erkannt",
        "front_rear": "mögliche Änderung an Scheinwerfer-, Rückleuchten- oder Front/Heck-Struktur erkannt",
        "wheels_tires": "mögliche Veränderung der Felgen- oder Reifenstruktur erkannt",
        "window_line": "Fensterlinie oder Dach-/Säulenstruktur auffällig",
        "body_lines": "Karosserielinie oder Türfuge auffällig",
        "center_grill_emblem": "mögliche Änderung an Kühlergrill oder Emblem erkannt",
    }
    for name, score in zone_scores.items():
        if name == "silhouette":
            local_strong = score < float(contour_cfg.get("critical_zone_score_max", 72.0))
            local_notice = score < float(contour_cfg.get("warning_zone_score_max", 86.0))
            if local_strong and (not high_mask_overlap or contour_geometric_relevant):
                add_finding(findings, "silhouette_contour", "critical", "detail_zone:silhouette")
                contour_warning_reason = "Kontur lokal stark auffällig und Maskengeometrie produktrelevant verändert."
            elif local_notice or (local_strong and high_mask_overlap):
                add_finding(findings, "silhouette_contour", "tolerated", "detail_zone:silhouette")
                contour_warning_reason = "Hohe Maskenüberlappung; lokale Randabweichung wird als nicht produktkritisch toleriert."
            continue
        if score < float(thresholds.get("critical_zone_score_max", 78.0)):
            zone_key = {"front_rear": "front_rear_structure", "wheels_tires": "wheel_tire_structure", "window_line": "window_line", "body_lines": "body_line_door_gap", "center_grill_emblem": "grille_front_structure"}.get(name, name)
            add_finding(findings, zone_key, "critical", f"detail_zone:{name}")
        elif score < float(thresholds.get("warning_zone_score_max", 90.0)):
            zone_key = {"front_rear": "front_rear_structure", "wheels_tires": "wheel_tire_structure", "window_line": "window_line", "body_lines": "body_line_door_gap", "center_grill_emblem": "grille_front_structure"}.get(name, name)
            add_finding(findings, zone_key, "tolerated", f"detail_zone:{name}")
    if structure["score"] < float(thresholds.get("failed_structure_min", 80.0)):
        add_finding(findings, "mask_alignment", "critical", "structure_score")

    headlight_score = float(component_values.get("headlight_score", 100.0))
    headlight_diff = float(component_diffs.get("headlight", 0.0))
    light_signature_score = float(component_values.get("front_light_signature_score", 100.0))
    grille_score = float(component_values.get("grille_score", 100.0))
    emblem_score = float(component_values.get("emblem_score", 100.0))
    front_wheel_score = float(component_values.get("front_wheel_score", 100.0))
    rear_wheel_score = float(component_values.get("rear_wheel_score", 100.0))
    wheel_tire_score = float(component_values.get("wheel_tire_score", 100.0))
    window_line_score = float(component_values.get("window_line_score", 100.0))
    silhouette_score = float(component_values.get("silhouette_score", 100.0))
    critical_component_score = min(
        headlight_score,
        light_signature_score,
        grille_score,
        emblem_score,
        front_wheel_score,
        rear_wheel_score,
        wheel_tire_score,
        float(component_values.get("rear_light_score", 100.0)),
    )
    wheel_score = min(front_wheel_score, rear_wheel_score, wheel_tire_score)

    component_blend_cap = (0.62 * base_final_product_integrity_score_pct) + (0.14 * detail["score"]) + (0.10 * critical_component_score) + (0.14 * wheel_score)
    component_blend_cap_active = bool(component_findings or any(item.get("severity") == "critical" for item in findings.values()))
    if component_blend_cap_active and component_blend_cap < final_product_integrity_score_pct:
        score_adjustments.append({
            "type": "component_blend_cap",
            "before": float(final_product_integrity_score_pct),
            "after": float(component_blend_cap),
            "delta": float(final_product_integrity_score_pct - component_blend_cap),
            "hard": False,
            "visible": bool(component_findings or findings),
            "reason": "Detail-, Rad- und kritische Bauteil-Scores begrenzen den Endscore weich.",
        })
        final_product_integrity_score_pct = component_blend_cap

    light_signature_diff = float(component_diffs.get("front_light_signature", 0.0))
    front_wheel_diff = float(component_diffs.get("front_wheel", 0.0))
    rear_wheel_diff = float(component_diffs.get("rear_wheel", 0.0))
    wheel_tire_diff = float(component_diffs.get("wheel_tire", 0.0))
    grille_diff = float(component_diffs.get("grille", 0.0))
    emblem_diff = float(component_diffs.get("emblem", 0.0))

    headlight_diff_gate = float(thresholds.get("critical_headlight_diff_min", 0.14))
    light_diff_gate = float(thresholds.get("critical_light_signature_diff_min", 0.12))
    wheel_diff_gate = float(thresholds.get("warning_wheel_diff_min", 0.12))
    front_diff_gate = float(thresholds.get("warning_front_detail_diff_min", 0.17))

    headlight_failed = headlight_diff > headlight_diff_gate or (
        headlight_score < float(thresholds.get("critical_headlight_score_max", 72.0))
        and headlight_diff > min(light_diff_gate, 0.08)
    )
    light_signature_failed = light_signature_score < float(thresholds.get("critical_light_signature_score_max", 74.0)) and light_signature_diff > light_diff_gate
    critical_wheel_score_max = float(thresholds.get("critical_wheel_score_max", 90.0))
    warning_wheel_score_max = float(thresholds.get("warning_wheel_score_max", 94.0))
    wheel_local_diff = max(front_wheel_diff, rear_wheel_diff, wheel_tire_diff)
    front_wheel_bad = front_wheel_score < warning_wheel_score_max and front_wheel_diff > wheel_diff_gate
    rear_wheel_bad = rear_wheel_score < warning_wheel_score_max and rear_wheel_diff > wheel_diff_gate
    wheel_strong = (wheel_score < critical_wheel_score_max and wheel_local_diff > wheel_diff_gate) or (
        wheel_score < warning_wheel_score_max and wheel_local_diff > max(wheel_diff_gate, 0.16)
    )
    wheel_warning = (wheel_score < warning_wheel_score_max and wheel_local_diff > wheel_diff_gate) or front_wheel_bad or rear_wheel_bad
    grille_strong = grille_score < float(thresholds.get("critical_grille_score_max", 70.0)) and grille_diff > front_diff_gate
    grille_warning = grille_score < float(thresholds.get("warning_grille_score_max", 84.0)) and grille_diff > front_diff_gate
    emblem_strong = emblem_score < float(thresholds.get("critical_emblem_score_max", 70.0)) and emblem_diff > front_diff_gate
    emblem_warning = emblem_score < float(thresholds.get("warning_emblem_score_max", 84.0)) and emblem_diff > front_diff_gate

    if headlight_failed:
        add_finding(findings, "headlight_light_signature", "critical", "component:headlight")
        component_findings.append("Scheinwerfer/Lichtsignatur")
        critical_component_findings.append("Scheinwerfer/Lichtsignatur")
    if light_signature_failed and not headlight_failed:
        add_finding(findings, "headlight_light_signature", "critical", "component:light_signature")
        component_findings.append("Lichtsignatur")
        critical_component_findings.append("Lichtsignatur")
    if wheel_strong:
        add_finding(findings, "wheel_tire_structure", "critical", "component:wheel_tire")
        component_findings.append("Felgen/Reifen/Radstruktur")
        critical_component_findings.append("Felgen/Reifen/Radstruktur")
    elif wheel_warning:
        add_finding(findings, "wheel_tire_structure", "tolerated", "component:wheel_tire")
        component_findings.append("Felgen/Reifen/Radstruktur")
    if grille_strong:
        add_finding(findings, "grille_front_structure", "critical", "component:grille")
        component_findings.append("Kühlergrill")
        critical_component_findings.append("Kühlergrill")
    elif grille_warning:
        add_finding(findings, "grille_front_structure", "tolerated", "component:grille")
        component_findings.append("Kühlergrill")
    if emblem_strong:
        add_finding(findings, "emblem_front_structure", "critical", "component:emblem")
        component_findings.append("Mercedes-Stern/Emblem")
        critical_component_findings.append("Mercedes-Stern/Emblem")
    elif emblem_warning:
        add_finding(findings, "emblem_front_structure", "tolerated", "component:emblem")
        component_findings.append("Mercedes-Stern/Emblem")

    finding_zone_map = {
        "wheel_tire_structure": components["zones"]["front_wheel"] | components["zones"]["rear_wheel"] | components["zones"]["wheel_tire"],
        "headlight_light_signature": components["zones"]["headlight"] | components["zones"]["front_light_signature"],
        "grille_front_structure": components["zones"]["grille"],
        "emblem_front_structure": components["zones"]["emblem"],
    }
    overlay_evidence = np.zeros(ref.shape[:2], dtype=bool)
    overlay_threshold = float(thresholds.get("finding_overlay_diff_threshold", 0.08))
    invalid_findings = []
    for key, zone in finding_zone_map.items():
        if key not in findings:
            continue
        local_activity = zone & (components["component_diff_map"] >= overlay_threshold)
        if not np.any(zone) or np.sum(local_activity) < 1:
            findings.pop(key, None)
            invalid_findings.append(key)
            component_findings = [item for item in component_findings if not (key == "wheel_tire_structure" and "Felgen" in item)]
            continue
        overlay_evidence |= local_activity

    reflection_weights = build_reflection_downweight_map(ref, gen, car_mask=scope)
    downweight_threshold = float(reflection_cfg.get("downweight_threshold", DEFAULT_MERCEDES_WEIGHT_PROFILE.get("downweight_threshold", 0.92)))
    scope_mask_for_reflection = scope if scope is not None else np.ones(ref.shape[:2], dtype=bool)
    reflection_candidate = (reflection_weights < downweight_threshold) & scope_mask_for_reflection
    critical_component_mask = np.asarray(components["critical_component_mask"], dtype=bool)
    glass_masks = build_glass_region_masks(ref, gen, car_mask=scope)
    glass_surface_mask = glass_masks.get("surface", np.zeros(ref.shape[:2], dtype=bool))
    paint_reflection_zone = scope_mask_for_reflection & ~critical_component_mask & ~glass_masks.get("line", np.zeros(ref.shape[:2], dtype=bool))
    allowed_reflection_zone = (glass_surface_mask | paint_reflection_zone) & ~critical_component_mask
    accepted_reflection = reflection_candidate & allowed_reflection_zone
    rejected_reflection = reflection_candidate & critical_component_mask
    metric_area = max(float(np.sum(scope)) if scope is not None else float(ref.shape[0] * ref.shape[1]), 1.0)
    reflection_downweight_area_ratio = float(np.sum(accepted_reflection) / metric_area)
    accepted_structure_mean = masked_mean(structure["diff_map"], accepted_reflection, default=1.0)
    has_reflection_evidence = (
        reflection_downweight_area_ratio >= float(reflection_cfg.get("min_downweight_area_ratio", 0.01))
        and color_score["score"] < float(reflection_cfg.get("max_color_reflection_score", 98.0))
        and accepted_structure_mean <= float(reflection_cfg.get("max_structure_diff_mean", 0.12))
        and np.sum(accepted_reflection & critical_component_mask) == 0
        and not component_findings
    )
    if has_reflection_evidence:
        if np.sum(accepted_reflection & glass_surface_mask) >= np.sum(accepted_reflection) * 0.5:
            add_finding(findings, "glass_reflection", "tolerated", "reflection:evidence")
        else:
            add_finding(findings, "paint_reflection", "tolerated", "reflection:evidence")
        if not component_findings:
            if "body_line_door_gap" in findings and accepted_structure_mean <= float(reflection_cfg.get("max_structure_diff_mean", 0.12)):
                findings["body_line_door_gap"]["severity"] = "tolerated"
                findings["body_line_door_gap"]["reason"] = "reflection:tolerated_body_line"
            final_product_integrity_score_pct = max(final_product_integrity_score_pct, min(base_final_product_integrity_score_pct, float(thresholds.get("failed_product_integrity_min", 90.0))))

    def apply_score_cap(cap_value, cap_type, reason, hard=False):
        nonlocal final_product_integrity_score_pct
        cap_value = float(cap_value)
        before = float(final_product_integrity_score_pct)
        after = min(before, cap_value)
        if after < before:
            score_adjustments.append({"type": cap_type, "before": before, "after": after, "delta": float(before - after), "hard": bool(hard), "reason": reason})
        final_product_integrity_score_pct = after

    if headlight_failed or light_signature_failed:
        apply_score_cap(min(headlight_score, light_signature_score) + 18.0, "critical_headlight_light_signature_cap", "Harte Critical-Fail-Regel für Scheinwerfer/Lichtsignatur.", hard=True)
    wheel_finding_active = "wheel_tire_structure" in findings
    if wheel_warning and wheel_finding_active:
        apply_score_cap(wheel_score + 6.0, "tolerated_wheel_warning_cap", "Tolerierter Rad-Hinweis; nur begrenzte weiche Deckelung.", hard=False)
    if wheel_strong and wheel_finding_active:
        apply_score_cap(wheel_score + 4.0, "critical_wheel_cap", "Harte Critical-Fail-Regel für Felgen/Reifen/Radstruktur.", hard=True)
    if wheel_strong and wheel_score < float(thresholds.get("warning_wheel_score_max", 94.0)) and wheel_local_diff > wheel_diff_gate:
        apply_score_cap(wheel_score + (4.0 if wheel_score < float(thresholds.get("critical_wheel_score_max", 90.0)) else 6.0), "critical_wheel_local_diff_cap", "Rad-Score und lokale Raddifferenz überschreiten Critical-Grenze.", hard=True)
    has_critical_finding_before_split = any(item.get("severity") == "critical" for item in findings.values())
    if has_critical_finding_before_split and component_findings and critical_component_score < float(thresholds.get("warning_critical_component_score_max", 94.0)):
        apply_score_cap(critical_component_score + 6.0, "critical_component_cap", "Kritischer Bauteilfund begrenzt Endscore.", hard=True)

    if debug_dir:
        debug_path = Path(debug_dir); debug_path.mkdir(parents=True, exist_ok=True)
        absolute_structure_threshold = float(thresholds.get("absolute_structure_diff_threshold", 0.30))
        absolute_lpips_threshold = float(thresholds.get("absolute_weighted_lpips_threshold", 0.30))
        absolute_component_threshold = float(thresholds.get("absolute_component_diff_threshold", 0.30))
        scope_mask = scope if scope is not None else np.ones(ref.shape[:2], dtype=bool)
        structure_map = np.clip(np.asarray(structure["diff_map"], dtype=np.float32), 0.0, 1.0) * scope_mask.astype(np.float32)
        component_map = np.clip(np.asarray(components["component_diff_map"], dtype=np.float32), 0.0, 1.0) * scope_mask.astype(np.float32)
        lpips_map_local = resize_float_map_to_shape(np.asarray(lpips_component_map, dtype=np.float32), ref.shape[:2]) if lpips_component_map is not None else np.zeros(ref.shape[:2], dtype=np.float32)
        if np.max(lpips_map_local) > 1.0:
            lpips_map_local = lpips_map_local / (float(np.percentile(lpips_map_local, 98)) + 1e-8)
        lpips_map_local = np.clip(lpips_map_local, 0.0, 1.0)
        absolute_mask = (structure_map >= absolute_structure_threshold) | (lpips_map_local >= absolute_lpips_threshold) | (component_map >= absolute_component_threshold)
        relative_map = np.maximum(structure_map, component_map)
        combined_map = np.maximum(relative_map, absolute_mask.astype(np.float32))
        extra_paths = {
            "structure_diff_map": debug_path / f"{stem}_structure_diff_map.png",
            "reflection_diff_map": debug_path / f"{stem}_reflection_diff_map.png",
            "overlay_findings_map": debug_path / f"{stem}_overlay_findings_map.png",
            "invalid_zone_diagnostics": debug_path / f"{stem}_invalid_zone_diagnostics.png",
            "heatmap_relative": debug_path / f"{stem}_heatmap_relative.png",
            "heatmap_absolute_threshold": debug_path / f"{stem}_heatmap_absolute_threshold.png",
            "heatmap_combined": debug_path / f"{stem}_heatmap_combined.png",
            "reflection_candidate_map": debug_path / f"{stem}_reflection_candidate_map.png",
            "reflection_accepted_map": debug_path / f"{stem}_reflection_accepted_map.png",
            "reflection_rejected_due_to_critical_component": debug_path / f"{stem}_reflection_rejected_due_to_critical_component.png",
            "reflection_downweight": debug_path / f"{stem}_reflection_downweight.png",
        }
        save_weight_debug_map(structure_map, extra_paths["structure_diff_map"])
        save_weight_debug_map(color_score["map"] * scope_mask.astype(np.float32), extra_paths["reflection_diff_map"])
        save_weight_debug_map(overlay_evidence.astype(np.float32), extra_paths["overlay_findings_map"])
        invalid_map = np.zeros(ref.shape[:2], dtype=np.float32)
        for zone in finding_zone_map.values():
            if np.any(zone):
                invalid_map = np.maximum(invalid_map, zone.astype(np.float32) * 0.25)
        if invalid_findings:
            invalid_map = np.maximum(invalid_map, np.where(scope_mask, 1.0, 0.0).astype(np.float32) * 0.1)
        save_weight_debug_map(invalid_map, extra_paths["invalid_zone_diagnostics"])
        save_weight_debug_map(relative_map, extra_paths["heatmap_relative"])
        save_weight_debug_map(absolute_mask.astype(np.float32), extra_paths["heatmap_absolute_threshold"])
        save_weight_debug_map(combined_map, extra_paths["heatmap_combined"])
        save_weight_debug_map(reflection_candidate.astype(np.float32), extra_paths["reflection_candidate_map"])
        save_weight_debug_map(accepted_reflection.astype(np.float32), extra_paths["reflection_accepted_map"])
        save_weight_debug_map(rejected_reflection.astype(np.float32), extra_paths["reflection_rejected_due_to_critical_component"])
        save_weight_debug_map(mask_debug_map_to_vehicle(reflection_weights, scope_mask), extra_paths["reflection_downweight"])
    else:
        extra_paths = {}

    critical, tolerated = split_findings(findings)
    stable_pass_candidate = (
        final_product_integrity_score_pct >= float(thresholds.get("soft_pass_product_integrity_min", 88.0))
        and structure["score"] >= float(thresholds.get("passed_structure_min", 90.0))
        and detail["score"] >= float(thresholds.get("passed_detail_min", 88.0))
        and lpips_car_only_similarity_pct >= 80.0
        and not critical
    )
    if (
        headlight_failed
        or light_signature_failed
        or grille_strong
        or emblem_strong
        or wheel_strong
        or (wheel_score < float(thresholds.get("critical_wheel_score_max", 90.0)) and wheel_local_diff > wheel_diff_gate)
    ):
        decision = "failed"
    elif (
        final_product_integrity_score_pct < float(thresholds.get("failed_product_integrity_min", 90.0))
        and not (has_reflection_evidence and not component_findings and final_product_integrity_score_pct >= float(thresholds.get("reflection_tolerated_product_min", 88.0)))
    ) or structure["score"] < float(thresholds.get("failed_structure_min", 80.0)) or detail["score"] < float(thresholds.get("failed_detail_min", 80.0)):
        decision = "failed"
    elif (
        wheel_warning
        or (wheel_score < float(thresholds.get("warning_wheel_score_max", 94.0)) and wheel_local_diff > wheel_diff_gate)
        or (component_findings and critical_component_score < float(thresholds.get("warning_critical_component_score_max", 94.0)))
        or final_product_integrity_score_pct < float(thresholds.get("passed_product_integrity_min", 95.0))
    ):
        decision = "warning"
    elif stable_pass_candidate or (final_product_integrity_score_pct >= float(thresholds.get("passed_product_integrity_min", 95.0)) and lpips_car_only_similarity_pct >= 80.0 and not critical):
        decision = "passed"
    else:
        decision = "warning"
    if critical:
        interpretation = "Die Fahrzeugmaske und Detailzonen zeigen produktrelevante Abweichungen. Mindestens ein kritischer Struktur-, Kontur- oder Detailhinweis muss geprüft werden."
        if headlight_failed or light_signature_failed:
            decision_reason = "Das Bild wurde als failed bewertet, weil im Scheinwerferbereich eine kritische Strukturabweichung erkannt wurde."
            if wheel_warning or wheel_strong:
                decision_reason += " Zusätzlich wurden Abweichungen an Rad-/Felgenstrukturen festgestellt."
            elif grille_strong or emblem_strong:
                decision_reason += " Zusätzlich ist der markenspezifische Frontbereich auffällig."
        elif wheel_strong:
            decision_reason = "Das Bild wurde aufgrund deutlicher Abweichungen an Felgen-, Reifen- oder Radstruktur als kritisch bewertet."
        else:
            decision_reason = "critical_findings vorhanden oder Kernscore unter Fehlergrenze."
    elif tolerated:
        interpretation = "Die Fahrzeugmaske stimmt gut überein. Die Fahrzeugstruktur ist stabil. Erkannte Reflexions-, Helligkeits- oder geringe Randabweichungen werden toleriert; es liegt keine eindeutige Produktabweichung vor."
        decision_reason = "Keine critical_findings; Abweichungen nur als tolerated_findings klassifiziert."
    else:
        interpretation = "Fahrzeugstruktur, Kontur und Detailzonen sind stabil. Es wurden keine produktrelevanten Abweichungen erkannt."
        decision_reason = "Scores über den Pass-Schwellen und keine Findings."
    critical_component_names = sorted(set(critical_component_findings))
    product_integrity_base_score_before_caps = float(base_final_product_integrity_score_pct)
    visible_findings_preview = critical + tolerated
    applied_caps = [item for item in score_adjustments if item.get("after", item.get("before", 0.0)) < item.get("before", 0.0) and ("cap" in item.get("type", "") or item.get("hard"))]
    applied_penalties = [item for item in score_adjustments if item.get("after", item.get("before", 0.0)) < item.get("before", 0.0) and item not in applied_caps]
    hidden_findings_count = sum(1 for item in score_adjustments if item.get("visible") is False)
    score_delta_due_to_caps = float(product_integrity_base_score_before_caps - final_product_integrity_score_pct)
    color_debug = dict(color_score.get("debug", {}))
    color_debug.update({
        "configured_weight": float(color_reflection_weight),
        "score_after_weighting": float(color_reflection_weighted_score * color_reflection_weight),
        "raw_final_score_influence_points": float(raw_color_reflection_influence),
        "capped_final_score_influence_points": float(final_without_color_score_pct - base_final_product_integrity_score_pct),
        "final_score_without_color_reflection": float(final_without_color_score_pct),
    })
    sanity_checks = []
    def add_sanity_check(name, passed, message):
        sanity_checks.append({"name": name, "passed": bool(passed), "message": message})

    all_main_scores = [structure["score"], detail["score"], color_score["score"], lpips_car_only_similarity_pct]
    hard_adjustments = [item for item in score_adjustments if item.get("hard")]
    strongest_adjustment = max(score_adjustments, key=lambda item: item.get("delta", item.get("before", 0.0) - item.get("after", 0.0)), default=None)
    if score_delta_due_to_caps > 0.3 and not visible_findings_preview and not any(item.get("visible", bool(visible_findings_preview)) for item in score_adjustments):
        add_sanity_check("hidden_internal_reduction_without_visible_reason", False, "Product-Integrity wurde durch interne Regel reduziert, aber kein sichtbarer Grund wurde ausgegeben.")
    if lpips_car_only_raw is not None and 0.0 <= lpips_car_only_raw < 0.10:
        add_sanity_check("lpips_distance_below_0_10_similarity_above_90", lpips_car_only_similarity_pct > 90.0, f"LPIPS-Distanz {lpips_car_only_raw:.4f} ergibt Similarity {lpips_car_only_similarity_pct:.2f}%.")
    add_sanity_check("all_components_above_85_not_below_20_without_hard_fail", not (all(score > 85.0 for score in all_main_scores) and final_product_integrity_score_pct < 20.0 and not hard_adjustments), "Alle Teilwerte >85%; Endscore darf ohne harte Critical-Fail-Regel nicht <20% fallen.")
    add_sanity_check("low_final_score_has_cause", not (final_product_integrity_score_pct < 20.0 and not (critical or score_adjustments)), "Endscore <20% muss durch kritische Findings oder dokumentierte Penalties erklärt sein.")
    if lpips_car_only_raw is not None and 0.0 <= lpips_car_only_raw <= 1.0:
        add_sanity_check("final_score_not_raw_lpips_distance_times_100", abs(final_product_integrity_score_pct - (lpips_car_only_raw * 100.0)) > 0.75, "Endscore darf nicht nahezu LPIPS-Distanz*100 entsprechen; sonst Richtungsfehler-Verdacht.")
    add_sanity_check("color_reflection_drop_has_attribution", not (color_score["score"] + 8.0 < structure["score"] and not color_score.get("debug")), "Starker Color-/Reflection-Abfall muss im Debug attributierbar sein.")
    tolerated_penalty = sum(item.get("delta", 0.0) for item in score_adjustments if item["type"].startswith("tolerated"))
    add_sanity_check("tolerated_findings_do_not_massively_reduce_score", tolerated_penalty <= 8.0, f"Tolerierte Hinweise senken den Score um {tolerated_penalty:.2f} Punkte.")
    low_score_explanation = ""
    if final_product_integrity_score_pct < 20.0:
        if hard_adjustments:
            low_score_explanation = "; ".join(f"{item['type']}: {item['reason']}" for item in hard_adjustments)
        elif strongest_adjustment:
            low_score_explanation = f"Stärkste Penalty {strongest_adjustment['type']}: {strongest_adjustment['reason']}"
        elif critical:
            low_score_explanation = "Critical Findings: " + ", ".join(critical)
        else:
            low_score_explanation = "WARNUNG: Kein harter Fehler und keine Penalty als Ursache dokumentiert."

    product_debug = {
        "score_direction": "Alle Product-Integrity-Teilwerte sind Similarity-Prozente: höher ist besser. LPIPS-Distanzen werden vor Gewichtung zu Similarity invertiert.",
        "structure_score": float(structure["score"]),
        "detail_zones_score": float(detail["score"]),
        "color_reflection_score": float(color_score["score"]),
        "car_only_lpips_raw": lpips_car_only_raw,
        "car_only_lpips_similarity_percent": float(lpips_car_only_similarity_pct),
        "configured_weights": {
            "structure_weight": float(weights.get("structure_only_score", 0.0)),
            "detail_weight": float(weights.get("detail_zones_score", 0.0)),
            "color_reflection_weight": float(weights.get("color_reflection_score", 0.0)),
            "car_only_lpips_weight": float(weights.get("car_only_lpips_score", 0.0)),
        },
        "raw_inputs": {"lpips_car_only_input": lpips_car_only_raw, "lpips_car_only_raw": lpips_car_only_raw, "lpips_car_only_similarity_pct": float(lpips_car_only_similarity_pct)},
        "component_scores_pct": {k: float(v) for k, v in component_scores.items()},
        "weights": {k: float(weights.get(k, 0.0)) for k in weights},
        "base_score_before_caps": product_integrity_base_score_before_caps,
        "base_final_before_caps_pct": product_integrity_base_score_before_caps,
        "final_without_color_reflection_pct": float(final_without_color_score_pct),
        "final_product_integrity_score_pct": float(final_product_integrity_score_pct),
        "final_score_after_caps": float(final_product_integrity_score_pct),
        "product_integrity_base_score_before_caps": product_integrity_base_score_before_caps,
        "product_integrity_final_score_after_caps": float(final_product_integrity_score_pct),
        "product_integrity_score_delta_due_to_caps": score_delta_due_to_caps,
        "score_adjustments": score_adjustments,
        "applied_caps": applied_caps,
        "applied_penalties": applied_penalties,
        "visible_findings": visible_findings_preview,
        "hidden_findings_count": int(hidden_findings_count),
        "critical_findings": critical,
        "tolerated_findings": tolerated,
        "low_score_explanation": low_score_explanation,
        "reflection_attribution": {
            "accepted_area_ratio": float(reflection_downweight_area_ratio),
            "accepted_structure_diff_mean": float(accepted_structure_mean),
            "glass_surface_area_ratio": float(np.sum(glass_surface_mask & scope_mask_for_reflection) / metric_area),
            "rejected_critical_component_area_ratio": float(np.sum(rejected_reflection) / metric_area),
            "debug_paths": {key: str(value) for key, value in extra_paths.items() if key.startswith("reflection")},
        },
        "sanity_checks": sanity_checks,
    }
    debug_paths = {}; debug_paths.update(structure["debug_paths"]); debug_paths.update(detail["debug_paths"]); debug_paths.update(color_score["debug_paths"]); debug_paths.update(components["debug_paths"]); debug_paths.update({k: str(v) for k, v in extra_paths.items()})
    return {
        "structure_only_score": structure["score"],
        "detail_zones_score": detail["score"],
        "color_reflection_score": color_score["score"],
        "lpips_car_only_similarity_pct": lpips_car_only_similarity_pct,
        "final_product_integrity_score_pct": final_product_integrity_score_pct,
        "product_integrity_score": final_product_integrity_score_pct,
        "product_integrity_base_score_before_caps": product_integrity_base_score_before_caps,
        "product_integrity_final_score_after_caps": final_product_integrity_score_pct,
        "product_integrity_score_delta_due_to_caps": score_delta_due_to_caps,
        "applied_caps": applied_caps,
        "applied_penalties": applied_penalties,
        "hidden_findings_count": hidden_findings_count,
        "product_integrity_decision": decision,
        "critical_findings": critical,
        "tolerated_findings": tolerated,
        "product_integrity_debug_paths": debug_paths,
        "detail_zone_scores": zone_scores,
        "component_scores": component_values,
        "headlight_score": headlight_score,
        "headlight_diff": headlight_diff,
        "light_signature_score": light_signature_score,
        "front_light_signature_score": light_signature_score,
        "front_wheel_score": front_wheel_score,
        "rear_wheel_score": rear_wheel_score,
        "wheel_tire_score": wheel_tire_score,
        "wheel_score": wheel_score,
        "critical_component_score": critical_component_score,
        "tire_score": wheel_tire_score,
        "grille_score": grille_score,
        "emblem_score": emblem_score,
        "window_line_score": window_line_score,
        "silhouette_score": silhouette_score,
        "critical_component_detected": bool(critical_component_names),
        "critical_component_names": critical_component_names,
        "product_integrity_interpretation": interpretation,
        "decision_reason": decision_reason,
        "contour_warning_reason": contour_warning_reason,
        "detail_zone_area_ratio": detail["detail_zone_area_ratio"],
        "structure_masked_area_ratio": structure["structure_masked_area_ratio"],
        "color_reflection_debug": color_debug,
        "product_integrity_debug": product_debug,
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

    base_mask = remove_objects_smaller_than(base_mask, min_object_area)
    base_mask = fill_holes_smaller_than(base_mask, max_hole_area)
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
    out_dir="normalized",
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
    mercedes_weight_profile=None,
    product_integrity_profile=None,
):
    ref_img = load_image(ref_path)
    gen_img = load_image(gen_path)
    validate_image_for_metrics(ref_img, image_name="ref_img")
    validate_image_for_metrics(gen_img, image_name="gen_img")

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm, gen_norm, content_mask = normalize_pair(ref_img, gen_img, mode=mode)
    ref_norm, gen_norm, content_mask, metric_scale = downscale_pair_for_metrics(
        ref_norm,
        gen_norm,
        content_mask=content_mask,
        max_long_edge_px=max_metric_long_edge,
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

    car_metrics = compute_car_only_metrics(
        ref_norm,
        gen_norm,
        ref_path,
        gen_path,
        lpips_model,
        segmenter,
        car_mode=car_mode,
        mask_source=mask_source,
        pad_px=pad_px,
        neutral_value=neutral_value,
        min_mask_area=min_mask_area,
        mask_downsample=mask_downsample,
        eps=eps,
        debug_dir=debug_dir,
        car_only_dir=car_only_dir,
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
    validate_same_spatial_size(
        {
            "ref_norm": ref_norm,
            "gen_norm": gen_norm,
            "final_vehicle_mask": car_focus_mask,
            "ref_vehicle_mask": ref_car_mask,
            "gen_vehicle_mask": gen_car_mask,
            "content_mask": valid_content_mask,
        },
        "Vollbild-Masken nach Normalisierung",
    )

    reflection_scope_mask = car_focus_mask if car_focus_mask is not None and np.any(car_focus_mask) else valid_content_mask
    mercedes_weight_profile = mercedes_weight_profile or load_mercedes_weight_profile()
    mercedes_profile_enabled = bool(mercedes_weight_profile.get("enabled", True))
    used_weight_profile = str(mercedes_weight_profile.get("name", DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME))

    mercedes_weight_map = build_mercedes_importance_map(
        ref_norm,
        gen_norm,
        car_mask=reflection_scope_mask,
        profile=mercedes_weight_profile,
    )
    reflection_weight_map = build_reflection_downweight_map(
        ref_norm,
        gen_norm,
        car_mask=reflection_scope_mask,
        content_mask=valid_content_mask,
        profile=mercedes_weight_profile,
    )
    glass_region_masks = build_glass_region_masks(
        ref_norm,
        gen_norm,
        car_mask=reflection_scope_mask,
        profile=mercedes_weight_profile,
    )
    weighted_scope_mask = build_weighted_lpips_scope_mask(
        car_mask=reflection_scope_mask,
        glass_interior_mask=glass_region_masks["interior"],
        content_mask=valid_content_mask,
    )
    validate_same_spatial_size(
        {
            "mercedes_weight": mercedes_weight_map,
            "reflection_weight": reflection_weight_map,
            "glass_interior_mask": glass_region_masks["interior"],
            "weighted_lpips_scope": weighted_scope_mask,
            "ref_norm": ref_norm,
            "gen_norm": gen_norm,
        },
        "Weighted-LPIPS-Eingaben",
    )
    reflection_scope_bool = prepare_metric_mask(reflection_scope_mask, ref_norm, gen_norm)
    if reflection_scope_bool is None:
        reflection_scope_bool = reflection_weight_map > 0
    critical_component_zones = build_critical_component_zones(reflection_scope_bool, ref_norm.shape[:2])
    critical_component_mask = (
        critical_component_zones["headlight_zone"]
        | critical_component_zones["light_signature_zone"]
        | critical_component_zones["grille_zone"]
        | critical_component_zones["emblem_zone"]
        | critical_component_zones["front_wheel_zone"]
        | critical_component_zones["rear_wheel_zone"]
        | critical_component_zones["tire_zone"]
        | critical_component_zones["rear_light_zone"]
        | critical_component_zones["rear_contour_zone"]
        | critical_component_zones["window_line_zone"]
    )
    protected_reflection_weight_map = np.where(critical_component_mask, 1.0, reflection_weight_map)
    protected_mercedes_weight_map = np.where(
        critical_component_mask,
        np.maximum(mercedes_weight_map, float(mercedes_weight_profile.get("critical_component_weight_floor", 1.65))),
        mercedes_weight_map,
    )
    front_detail_mask = (
        critical_component_zones["headlight_zone"]
        | critical_component_zones["light_signature_zone"]
        | critical_component_zones["grille_zone"]
        | critical_component_zones["emblem_zone"]
    )
    wheel_detail_mask = (
        critical_component_zones["front_wheel_zone"]
        | critical_component_zones["rear_wheel_zone"]
        | critical_component_zones["tire_zone"]
    )
    protected_mercedes_weight_map = np.where(
        front_detail_mask,
        np.maximum(protected_mercedes_weight_map, float(mercedes_weight_profile.get("critical_front_detail_weight_floor", 2.10))),
        protected_mercedes_weight_map,
    )
    protected_mercedes_weight_map = np.where(
        wheel_detail_mask,
        np.maximum(protected_mercedes_weight_map, float(mercedes_weight_profile.get("critical_wheel_weight_floor", 1.80))),
        protected_mercedes_weight_map,
    )
    final_weight_map_before_reflection_downweight = protected_mercedes_weight_map.copy()
    final_weight_map_after_reflection_downweight = protected_mercedes_weight_map * protected_reflection_weight_map if mercedes_profile_enabled else protected_mercedes_weight_map
    combined_weight_map = final_weight_map_after_reflection_downweight
    if weighted_scope_mask is not None:
        final_weight_map_before_reflection_downweight = np.where(weighted_scope_mask, final_weight_map_before_reflection_downweight, 0.0)
        final_weight_map_after_reflection_downweight = np.where(weighted_scope_mask, final_weight_map_after_reflection_downweight, 0.0)
        combined_weight_map = np.where(weighted_scope_mask, combined_weight_map, 0.0)
    weighted_debug = compute_weighted_lpips_from_map(
        lpips_map,
        combined_weight_map,
        mask=weighted_scope_mask,
        eps=eps,
        return_debug=True,
    )
    # Legacy/Debug: Der gewichtete LPIPS-Durchschnitt bleibt nur zur Diagnose und
    # für Debug-Karten erhalten. Er ist keine Hauptmetrik und keine
    # Entscheidungsgrundlage mehr; Product-Integrity liefert die fachliche Bewertung.
    legacy_debug_weighted_lpips_raw = weighted_debug["value"]
    legacy_debug_weighted_lpips_similarity_percent = convert_lpips_to_similarity_percent(legacy_debug_weighted_lpips_raw)
    legacy_reflection_robust_lpips_raw = legacy_debug_weighted_lpips_raw
    legacy_reflection_robust_lpips_similarity_percent = legacy_debug_weighted_lpips_similarity_percent
    final_similarity_score = None
    active_weights = combined_weight_map[combined_weight_map > 0]
    reflection_weight_mean = float(np.mean(active_weights)) if active_weights.size else None
    reflection_weight_min = float(np.min(active_weights)) if active_weights.size else None
    downweight_threshold = float(mercedes_weight_profile.get("downweight_threshold", 0.75))
    reflection_downweight_area_ratio = float(np.sum(reflection_scope_bool & ~critical_component_mask & (reflection_weight_map > 0) & (reflection_weight_map < downweight_threshold)) / max(float(np.sum(reflection_scope_bool)), 1.0))
    scope_area = max(float(np.sum(reflection_scope_bool)), 1.0)
    glass_interior_area_ratio = float(np.sum(glass_region_masks["interior"]) / scope_area)
    glass_contour_area_ratio = float(np.sum(glass_region_masks["contour"]) / scope_area)
    glass_mask_area_ratio = float(np.sum(glass_region_masks["candidate"]) / scope_area)
    window_contour_area_ratio = glass_contour_area_ratio
    weighted_lpips_region_contributions = summarize_weighted_lpips_regions(
        lpips_map,
        weighted_debug["effective_weight_map"],
        {
            "vehicle_scope_without_glass_interiors": weighted_scope_mask,
            "glass_interiors_excluded": glass_region_masks["interior"],
            "window_contours": glass_region_masks["contour"],
            "high_priority_product_edges": combined_weight_map >= max(1.0, float(np.percentile(active_weights, 75)) if active_weights.size else 1.0),
            "low_priority_reflection_or_plain_paint": (combined_weight_map > 0.0) & (combined_weight_map < downweight_threshold),
        },
        eps=eps,
    )
    reflection_weight_map_path = None
    mercedes_weight_map_path = None
    critical_component_mask_path = None
    glass_interior_mask_path = None
    window_contour_mask_path = None
    window_candidate_region_path = None
    window_surface_mask_path = None
    window_line_mask_path = None
    overlay_window_contour_ref_path = None
    overlay_window_contour_gen_path = None
    weighted_lpips_map_path = None
    effective_weight_map_path = None
    if debug_dir:
        debug_path = Path(debug_dir)
        stem = Path(ref_path).stem
        reflection_weight_map_path = str(debug_path / f"{stem}_reflection_downweight.png")
        mercedes_weight_map_path = str(debug_path / f"{stem}_mercedes_weight.png")
        critical_component_mask_path = str(debug_path / f"{stem}_critical_component_mask.png")
        glass_interior_mask_path = str(debug_path / f"{stem}_glass_interior_mask.png")
        window_contour_mask_path = str(debug_path / f"{stem}_window_contour_mask.png")
        window_candidate_region_path = str(debug_path / f"{stem}_window_candidate_region.png")
        window_surface_mask_path = str(debug_path / f"{stem}_window_surface_mask.png")
        window_line_mask_path = str(debug_path / f"{stem}_window_line_mask.png")
        overlay_window_contour_ref_path = str(debug_path / f"{stem}_overlay_window_contour_on_ref.png")
        overlay_window_contour_gen_path = str(debug_path / f"{stem}_overlay_window_contour_on_gen.png")
        weighted_lpips_map_path = str(debug_path / f"{stem}_weighted_lpips_map.png")
        effective_weight_map_path = str(debug_path / f"{stem}_weighted_lpips_effective_weight.png")
        final_weight_before_path = str(debug_path / f"{stem}_final_weight_map_before_reflection_downweight.png")
        final_weight_after_path = str(debug_path / f"{stem}_final_weight_map_after_reflection_downweight.png")
        save_weight_debug_map(mask_debug_map_to_vehicle(reflection_weight_map, reflection_scope_bool), Path(reflection_weight_map_path))
        save_weight_debug_map(mask_debug_map_to_vehicle(protected_mercedes_weight_map, reflection_scope_bool), Path(mercedes_weight_map_path))
        save_mask_image(critical_component_mask, Path(critical_component_mask_path))
        save_mask_image(glass_region_masks["interior"], Path(glass_interior_mask_path))
        save_mask_image(glass_region_masks["contour"], Path(window_contour_mask_path))
        save_mask_image(glass_region_masks["window_candidate_region"], Path(window_candidate_region_path))
        save_mask_image(glass_region_masks["surface"], Path(window_surface_mask_path))
        save_mask_image(glass_region_masks["line"], Path(window_line_mask_path))
        save_mask_overlay(ref_norm, glass_region_masks["contour"], Path(overlay_window_contour_ref_path))
        save_mask_overlay(gen_norm, glass_region_masks["contour"], Path(overlay_window_contour_gen_path))
        save_weight_debug_map(weighted_debug["weighted_map"], Path(weighted_lpips_map_path))
        save_weight_debug_map(weighted_debug["effective_weight_map"], Path(effective_weight_map_path))
        save_weight_debug_map(mask_debug_map_to_vehicle(final_weight_map_before_reflection_downweight, reflection_scope_bool), Path(final_weight_before_path))
        save_weight_debug_map(mask_debug_map_to_vehicle(final_weight_map_after_reflection_downweight, reflection_scope_bool), Path(final_weight_after_path))

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

    product_integrity_profile = product_integrity_profile or load_product_integrity_profile()
    product_integrity = compute_product_integrity_scores(
        ref_norm,
        gen_norm,
        car_mask=reflection_scope_mask,
        car_only_lpips_score=lpips_car_only_similarity_percent,
        profile=product_integrity_profile,
        debug_dir=debug_dir,
        stem=basename,
        mask_metrics=geometric,
        lpips_component_map=lpips_map,
    )

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
    print(f"  Legacy weighted LPIPS(debug): {legacy_debug_weighted_lpips_raw:.6f}")
    print(f"  Legacy weighted area(debug) : {weighted_debug['active_area_ratio'] * 100.0:.2f}%")
    print(f"  Glass interior excluded: {glass_interior_area_ratio * 100.0:.2f}%")
    print(f"  Product Integrity Score : {product_integrity['product_integrity_score']:.2f}%")
    print(f"  Product Integrity       : {product_integrity['product_integrity_decision']}")
    print(f"  Structure-only Score    : {product_integrity['structure_only_score']:.2f}%")
    print(f"  Detail-zones Score      : {product_integrity['detail_zones_score']:.2f}%")
    print(f"  Color-reflection Score  : {product_integrity['color_reflection_score']:.2f}%")
    print(f"  Headlight Score         : {product_integrity['headlight_score']:.2f}%")
    print(f"  Front Wheel Score       : {product_integrity['front_wheel_score']:.2f}%")
    print(f"  Rear Wheel Score        : {product_integrity['rear_wheel_score']:.2f}%")
    print(f"  Critical Components     : {', '.join(product_integrity['critical_component_names']) or 'keine'}")
    product_debug = product_integrity.get("product_integrity_debug", {})
    print("  Product Integrity Debug : Similarity-Richtung: höher ist besser; Distance/Similarity/Penalty getrennt")
    print(f"  PI Base before caps     : {product_debug.get('base_final_before_caps_pct', product_integrity['product_integrity_score']):.2f}%")
    print(f"  PI without Color/Refl.  : {product_debug.get('final_without_color_reflection_pct', product_integrity['product_integrity_score']):.2f}%")
    print(f"  PI adjustments          : {json.dumps(product_debug.get('score_adjustments', []), ensure_ascii=False)}")
    print(f"  PI sanity checks        : {json.dumps(product_debug.get('sanity_checks', []), ensure_ascii=False)}")
    if product_debug.get("low_score_explanation"):
        print(f"  PI low-score cause      : {product_debug['low_score_explanation']}")
    if segmenter is not None:
        print(f"  Mask area (%)      : {car_metrics['debug']['mask_area_ratio'] * 100.0:.2f}%")
        print(f"  BBox (Metrik)      : {car_metrics['debug']['metric_bbox']}")
        print(f"  BBox (Preview Ref) : {car_metrics['debug']['ref_preview_bbox']}")
        print(f"  BBox (Preview Gen) : {car_metrics['debug']['gen_preview_bbox']}")
        print(f"  LPIPS car-only     : {car_metrics['lpips_car_only']}")
        print(f"  LPIPS car-only (%) : {format_percent(lpips_car_only_similarity_percent)}")
        if car_metrics.get("car_only_paths", {}).get("ref"):
            print(f"  Car-only Ref saved : {car_metrics['car_only_paths']['ref']}")
            print(f"  Car-only Gen saved : {car_metrics['car_only_paths']['gen']}")
    else:
        print("  Car-only           : deaktiviert (nutze Full-Image-Logik)")
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
        "raw_lpips": lpips_val,
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
        "car_only_lpips": car_metrics["lpips_car_only"],
        "lpips_car_only_similarity_percent": lpips_car_only_similarity_percent,
        "legacy_debug_weighted_lpips_raw": legacy_debug_weighted_lpips_raw,
        "legacy_debug_weighted_lpips_similarity_percent": legacy_debug_weighted_lpips_similarity_percent,
        "legacy_reflection_robust_lpips_raw": legacy_reflection_robust_lpips_raw,
        "legacy_reflection_robust_lpips_similarity_percent": legacy_reflection_robust_lpips_similarity_percent,
        "final_similarity_score": product_integrity["product_integrity_score"],
        "lpips_raw": lpips_val,
        "lpips_score": percent_metrics["lpips_similarity_percent"],
        "car_only_lpips_raw": car_metrics["lpips_car_only"],
        "car_only_lpips_score": lpips_car_only_similarity_percent,
        "structure_only_score": product_integrity["structure_only_score"],
        "detail_zones_score": product_integrity["detail_zones_score"],
        "color_reflection_score": product_integrity["color_reflection_score"],
        "color_reflection_debug": json.dumps(product_integrity.get("color_reflection_debug", {}), sort_keys=True),
        "product_integrity_score": product_integrity["product_integrity_score"],
        "product_integrity_base_score_before_caps": product_integrity.get("product_integrity_base_score_before_caps"),
        "product_integrity_final_score_after_caps": product_integrity.get("product_integrity_final_score_after_caps"),
        "product_integrity_score_delta_due_to_caps": product_integrity.get("product_integrity_score_delta_due_to_caps"),
        "applied_caps": json.dumps(product_integrity.get("applied_caps", []), ensure_ascii=False),
        "applied_penalties": json.dumps(product_integrity.get("applied_penalties", []), ensure_ascii=False),
        "hidden_findings_count": product_integrity.get("hidden_findings_count", 0),
        "product_integrity_decision": product_integrity["product_integrity_decision"],
        "critical_findings": json.dumps(product_integrity["critical_findings"], ensure_ascii=False),
        "tolerated_findings": json.dumps(product_integrity["tolerated_findings"], ensure_ascii=False),
        "headlight_score": product_integrity["headlight_score"],
        "headlight_diff": product_integrity["headlight_diff"],
        "light_signature_score": product_integrity["light_signature_score"],
        "front_light_signature_score": product_integrity["front_light_signature_score"],
        "front_wheel_score": product_integrity["front_wheel_score"],
        "rear_wheel_score": product_integrity["rear_wheel_score"],
        "wheel_tire_score": product_integrity["wheel_tire_score"],
        "tire_score": product_integrity["tire_score"],
        "grille_score": product_integrity["grille_score"],
        "emblem_score": product_integrity["emblem_score"],
        "window_line_score": product_integrity["window_line_score"],
        "silhouette_score": product_integrity["silhouette_score"],
        "critical_component_detected": product_integrity["critical_component_detected"],
        "critical_component_names": json.dumps(product_integrity["critical_component_names"], ensure_ascii=False),
        "product_integrity_profile": product_integrity_profile.get("name", DEFAULT_PRODUCT_INTEGRITY_PROFILE_NAME),
        "product_integrity_debug_paths": json.dumps(product_integrity["product_integrity_debug_paths"], sort_keys=True),
        "product_integrity_debug": json.dumps(product_integrity.get("product_integrity_debug", {}), ensure_ascii=False, sort_keys=True),
        "product_integrity_interpretation": product_integrity["product_integrity_interpretation"],
        "decision_reason": product_integrity["decision_reason"],
        "contour_warning_reason": product_integrity["contour_warning_reason"],
        "glass_mask_area_ratio": glass_mask_area_ratio,
        "window_contour_area_ratio": window_contour_area_ratio,
        "detail_zone_area_ratio": product_integrity["detail_zone_area_ratio"],
        "structure_masked_area_ratio": product_integrity["structure_masked_area_ratio"],
        "mercedes_profile_enabled": mercedes_profile_enabled,
        "used_weight_profile": used_weight_profile,
        "reflection_weight_mean": reflection_weight_mean,
        "reflection_weight_min": reflection_weight_min,
        "reflection_downweight_area_ratio": reflection_downweight_area_ratio,
        "glass_interior_area_ratio": glass_interior_area_ratio,
        "glass_contour_area_ratio": glass_contour_area_ratio,
        "weighted_lpips_weight_sum": weighted_debug["weight_sum"],
        "weighted_lpips_active_area_ratio": weighted_debug["active_area_ratio"],
        "weighted_lpips_region_contributions": json.dumps(weighted_lpips_region_contributions, sort_keys=True),
        "reflection_weight_map_path": reflection_weight_map_path,
        "mercedes_weight_map_path": mercedes_weight_map_path,
        "critical_component_mask_path": critical_component_mask_path,
        "glass_interior_mask_path": glass_interior_mask_path,
        "window_contour_mask_path": window_contour_mask_path,
        "window_candidate_region_path": window_candidate_region_path,
        "window_surface_mask_path": window_surface_mask_path,
        "window_line_mask_path": window_line_mask_path,
        "overlay_window_contour_ref_path": overlay_window_contour_ref_path,
        "overlay_window_contour_gen_path": overlay_window_contour_gen_path,
        "weighted_lpips_map_path": weighted_lpips_map_path,
        "effective_weight_map_path": effective_weight_map_path,
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
    out_dir="normalized",
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
    mercedes_weight_profile=None,
    product_integrity_profile=None,
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
            mercedes_weight_profile=mercedes_weight_profile,
            product_integrity_profile=product_integrity_profile,
        )
        results.append(result)

    if not results:
        raise RuntimeError("Keine auswertbaren Bildpaare gefunden.")

    df = build_result_dataframe(results)
    df.to_csv(output_csv, index=False, float_format="%.6f", na_rep="")

    print("============================================================")
    print(f"[INFO] Ergebnisse gespeichert: {output_csv}")
    print(
        df[
            [
                "filename",
                "ssim",
                "ssim_percent",
                "lpips",
                "lpips_similarity_percent",
                "lpips_foreground",
                "lpips_foreground_similarity_percent",
                "delta_e_ciede2000",
                "delta_e_similarity_percent",
                "lpips_car_only",
                "lpips_car_only_similarity_percent",
                "final_similarity_score",
                "product_integrity_score",
                "product_integrity_decision",
                "mask_iou",
                "mask_dice",
            ]
        ].head()
    )


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
    parser.add_argument("--out", default="normalized", help="Output-Ordner für normalisierte Bilder")
    parser.add_argument("--output-csv", default="image_metrics_results.csv", help="CSV-Datei für Metrikergebnisse")
    parser.add_argument("--lpips-net", default="alex", choices=["alex", "vgg", "squeeze"], help="Backbone für LPIPS")
    parser.add_argument("--lpips-heatmap-dir", default="lpips_heatmaps", help="Ausgabeordner für LPIPS-Heatmaps (setze 'none' zum Deaktivieren)")
    parser.add_argument("--use-gpu", action="store_true", help="Nutze CUDA, falls verfügbar")
    parser.add_argument("--seed", type=int, default=None, help="Setze optionalen Zufalls-Seed für reproduzierbare Läufe")
    parser.add_argument("--deterministic", action="store_true", help="Aktiviere deterministische Backends (langsamer, aber reproduzierbarer)")
    parser.add_argument("--enable-car-only", action="store_true", help="Aktiviere Car-only Metriken (LPIPS/SSIM)")
    parser.add_argument("--car-only", action="store_true", help="Kurzform für --enable-car-only")
    parser.add_argument(
        "--car-mode",
        default="neutralize_crop",
        choices=["neutralize_crop", "roi_crop", "weighted_lpips"],
        help="Auto-fokussierte Car-only-Berechnung (weighted_lpips ist deprecated und wird auf neutralize_crop umgebogen)",
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
        type=int,
        default=1600,
        help="Skaliere normalisierte Bilder vor der Metrik-Berechnung auf diese maximale Kantenlänge (Performance-Schutz).",
    )
    parser.add_argument("--eps", type=float, default=1e-8, help="Deprecated: ohne produktive Wirkung")
    parser.add_argument("--mask-score-threshold", type=float, default=0.5, help="Score-Schwelle für Vehicle-Segmentierung")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Pixel-Schwelle der Segmentierungsmaske [0..1]")
    parser.add_argument("--debug-dir", default=None, help="Optionales Debug-Verzeichnis für Masken/Crops")
    parser.add_argument("--car-only-dir", default="car_only", help="Verzeichnis für gespeicherte Car-only-Crops")
    parser.add_argument("--weight-profile-config", default=None, help="Pfad zu configs/mercedes_weight_profiles.json")
    parser.add_argument("--weight-profile", default=None, help="Name des Mercedes Weight Profiles")
    parser.add_argument("--product-integrity-profile-config", default=None, help="Pfad zu configs/product_integrity_profile.json")
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
        ]
    )

    args.enable_car_only = args.enable_car_only or args.car_only or use_car_specific_option
    return args


def main():
    args = parse_args()
    configure_determinism(seed=args.seed, deterministic=args.deterministic)

    print("============================================================")
    print("[INFO] Starte Bildmetrik-Berechnung")
    print(f"[INFO] Mode            : {args.mode}")
    print(f"[INFO] Normalized out  : {args.out}")
    print(f"[INFO] Output CSV      : {args.output_csv}")
    print(f"[INFO] LPIPS Net       : {args.lpips_net}")
    print("[INFO] LPIPS Setup     : offizielles vortrainiertes Inferenzmodell (lin, kein Training im Tool)")
    print(f"[INFO] LPIPS Heatmaps  : {args.lpips_heatmap_dir}")
    print(f"[INFO] Seed            : {args.seed}")
    print(f"[INFO] Deterministisch : {args.deterministic}")
    print(f"[INFO] ROI min-size px : {args.roi_min_size_px}")
    print(f"[INFO] ROI square      : {args.roi_square}")
    print(f"[INFO] Max metric edge : {args.max_metric_long_edge}")
    print(f"[INFO] Car-only aktiv  : {args.enable_car_only}")
    print("============================================================")
    run_lpips_pipeline_sanity_checks()
    print("[INFO] Sanity-Check     : LPIPS-Pipeline geprüft")
    mercedes_weight_profile = load_mercedes_weight_profile(
        config_path=args.weight_profile_config,
        profile_name=args.weight_profile,
    )
    print(f"[INFO] Weight Profile   : {mercedes_weight_profile.get('name')} ({mercedes_weight_profile.get('source')})")
    product_integrity_profile = load_product_integrity_profile(args.product_integrity_profile_config)
    print(f"[INFO] Integrity Profile: {product_integrity_profile.get('name')} ({product_integrity_profile.get('source')})")

    lpips_model = init_lpips_model(net=args.lpips_net, use_gpu=args.use_gpu)
    verify_lpips_forward(lpips_model, net=args.lpips_net, use_gpu=args.use_gpu)
    segmenter = None
    needs_car_segmenter = args.enable_car_only or (args.lpips_heatmap_dir is not None)
    if needs_car_segmenter:
        if args.enable_car_only:
            print("[INFO] Car-only wird aktiviert. Einfacher Aufruf: python image_metrics.py --car-only")
        try:
            segmenter = build_vehicle_segmenter(
                use_gpu=args.use_gpu,
                score_threshold=args.mask_score_threshold,
                mask_threshold=args.mask_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Fahrzeugsegmentierung nicht verfügbar: {exc}")
            if args.enable_car_only:
                print("[WARN] Car-only wurde deaktiviert, Heatmaps laufen global ohne Fahrzeugkontur weiter.")
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
            mercedes_weight_profile=mercedes_weight_profile,
            product_integrity_profile=product_integrity_profile,
        )
        build_result_dataframe([result]).to_csv(args.output_csv, index=False, float_format="%.6f", na_rep="")
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
        mercedes_weight_profile=mercedes_weight_profile,
        product_integrity_profile=product_integrity_profile,
    )


if __name__ == "__main__":
    main()
