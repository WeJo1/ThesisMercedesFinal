import argparse
import inspect
import json
import random
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
        "structure_only_score": 0.45,
        "detail_zones_score": 0.40,
        "color_reflection_score": 0.10,
        "car_only_lpips_score": 0.05,
    },
    "thresholds": {
        "passed_product_integrity_min": 90.0,
        "passed_structure_min": 90.0,
        "passed_detail_min": 88.0,
        "failed_product_integrity_min": 80.0,
        "failed_structure_min": 80.0,
        "failed_detail_min": 80.0,
        "warning_color_reflection_max": 78.0,
        "critical_zone_score_max": 78.0,
        "warning_zone_score_max": 90.0,
    },
    "reflection_tolerance": {
        "low_edge_difference_bonus": 0.35,
        "tolerated_score_below": 82.0,
    },
    "detail_zone_weights": {
        "silhouette": 1.35,
        "front_rear": 1.25,
        "wheels_tires": 1.35,
        "window_line": 1.20,
        "body_lines": 1.00,
        "center_grill_emblem": 1.20,
    },
}
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
    "weighted_mercedes_lpips",
    "weighted_mercedes_lpips_similarity_percent",
    "reflection_robust_lpips",
    "reflection_robust_lpips_similarity_percent",
    "final_similarity_score",
    "lpips_raw",
    "lpips_score",
    "car_only_lpips_raw",
    "car_only_lpips_score",
    "structure_only_score",
    "detail_zones_score",
    "color_reflection_score",
    "product_integrity_score",
    "product_integrity_decision",
    "critical_findings",
    "tolerated_findings",
    "product_integrity_profile",
    "product_integrity_debug_paths",
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
    "glass_interior_mask_path",
    "window_contour_mask_path",
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
    "weighted_mercedes_lpips",
    "weighted_mercedes_lpips_similarity_percent",
    "reflection_robust_lpips",
    "reflection_robust_lpips_similarity_percent",
    "final_similarity_score",
    "reflection_weight_mean",
    "reflection_weight_min",
    "reflection_downweight_area_ratio",
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

    if car_mask is not None and np.any(car_mask):
        vehicle = np.asarray(car_mask, dtype=bool)
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

    glass_band = (
        (by >= float(profile.get("glass_zone_y_min", 0.18)))
        & (by <= float(profile.get("glass_zone_y_max", 0.58)))
        & (bx >= float(profile.get("glass_zone_x_margin", 0.10)))
        & (bx <= 1.0 - float(profile.get("glass_zone_x_margin", 0.10)))
        & vehicle
    )
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
        stronger_glass_like = glass_band & low_saturation & (dark_glass | bright_reflection | (smooth_glass & (by <= 0.48)))
        candidate = fill_holes_smaller_than(closing(stronger_glass_like, disk(2)), min_area + 1)

    if not np.any(candidate):
        empty = np.zeros((h, w), dtype=bool)
        return {"candidate": empty, "interior": empty, "contour": empty}

    radius = max(1, int(round(min(bh, bw) * float(profile.get("glass_contour_width_ratio", 0.018)))))
    contour = candidate & ~erosion(candidate, disk(radius))
    contour |= dilation(candidate & (edge > float(profile.get("glass_contour_edge_limit", 0.30))), disk(1))
    contour &= vehicle
    interior = candidate & ~contour

    return {"candidate": candidate.astype(bool), "interior": interior.astype(bool), "contour": contour.astype(bool)}

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
        save_mask_image(base_mask.astype(bool), debug_path / f"{stem}_raw_vehicle_mask.png")
        save_mask_image(mask, debug_path / f"{stem}_final_vehicle_mask.png")
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
        edge_threshold = max(0.035, float(np.percentile(scoped_edge_values, 58)) if scoped_edge_values.size else 0.035)
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

    silhouette = (dilation(scope, disk(2)) & ~erosion(scope, disk(3))) & scope
    wheel_band = (by >= 0.62) & (by <= 1.0)
    left_wheel = bbox_mask_from_fraction(fallback_shape, bbox, (0.07, 0.60, 0.36, 1.00)) & wheel_band
    right_wheel = bbox_mask_from_fraction(fallback_shape, bbox, (0.64, 0.60, 0.93, 1.00)) & wheel_band
    wheels_tires = (left_wheel | right_wheel) & (product_edges | changed_detail_edges | silhouette)

    front_rear_band = (
        bbox_mask_from_fraction(fallback_shape, bbox, (0.00, 0.30, 0.22, 0.78))
        | bbox_mask_from_fraction(fallback_shape, bbox, (0.78, 0.30, 1.00, 0.78))
    )
    front_rear = front_rear_band & (product_edges | changed_detail_edges | silhouette)

    window_band = (by >= 0.13) & (by <= 0.58) & (bx >= 0.08) & (bx <= 0.92) & scope
    changed_window_edges = changed_detail_edges & window_band & (glass_contour | (by <= 0.24) | (silhouette & (by <= 0.34)))
    window_line = (glass_contour | changed_window_edges | (window_band & product_edges) | (silhouette & (by <= 0.34))) & ~glass_interior
    body_band = (by >= 0.40) & (by <= 0.78) & (bx >= 0.08) & (bx <= 0.92) & scope
    body_lines = body_band & product_edges
    center_band = bbox_mask_from_fraction(fallback_shape, bbox, (0.36, 0.32, 0.64, 0.70))
    center_grill_emblem = center_band & product_edges

    zones = {
        "silhouette": silhouette,
        "front_rear": front_rear,
        "wheels_tires": wheels_tires,
        "window_line": window_line,
        "body_lines": body_lines,
        "center_grill_emblem": center_grill_emblem,
    }
    return {name: (zone & scope) for name, zone in zones.items()}

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
    return {"score": score, "diff_map": diff, "ref_edge": ref_edge, "gen_edge": gen_edge, "debug_paths": paths}


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
    return {"score": score, "zone_scores": zone_scores, "zones": zones, "debug_paths": paths}


def compute_color_reflection_score(ref, gen, mask=None, edge_diff=None, debug_dir=None, stem="pair"):
    """Bewerte Farb-/Lichtänderungen getrennt und dämpfe strukturstarke Pixel."""
    metric_mask = prepare_metric_mask(mask, ref, gen)
    if metric_mask is None:
        metric_mask = np.ones(ref.shape[:2], dtype=bool)
    color_diff = np.mean(np.abs(ref - gen), axis=2)
    if edge_diff is None:
        edge_diff = np.abs(sobel(color.rgb2gray(ref)) - sobel(color.rgb2gray(gen)))
    low_edge_weight = np.clip(1.0 - (edge_diff / 0.20), 0.25, 1.0)
    reflection_map = color_diff * low_edge_weight
    reflection_map = reflection_map * metric_mask.astype(np.float32)
    score = score_from_error(masked_mean(reflection_map, metric_mask), 0.22)
    paths = {}
    if debug_dir:
        paths["reflection_color_difference"] = str(Path(debug_dir) / f"{stem}_reflection_color_difference.png")
        save_weight_debug_map(reflection_map, Path(paths["reflection_color_difference"]))
    return {"score": score, "map": reflection_map, "debug_paths": paths}


def compute_product_integrity_scores(ref, gen, car_mask=None, car_only_lpips_score=None, profile=None, debug_dir=None, stem="pair"):
    """Führe Structure, Detail-Zones, Color/Reflection und Car-only-LPIPS zur Produktintegrität zusammen."""
    profile = profile or DEFAULT_PRODUCT_INTEGRITY_PROFILE
    scope = car_mask if car_mask is not None and np.any(car_mask) else None
    structure = compute_structure_only_score(ref, gen, mask=scope, debug_dir=debug_dir, stem=stem)
    detail = compute_detail_zones_score(ref, gen, mask=scope, profile=profile, structure_debug=structure, debug_dir=debug_dir, stem=stem)
    color_score = compute_color_reflection_score(ref, gen, mask=scope, edge_diff=structure["diff_map"], debug_dir=debug_dir, stem=stem)
    component_scores = {
        "structure_only_score": structure["score"],
        "detail_zones_score": detail["score"],
        "color_reflection_score": color_score["score"],
        "car_only_lpips_score": car_only_lpips_score if car_only_lpips_score is not None else structure["score"],
    }
    enabled = profile.get("enabled_components", {})
    weights = profile.get("weights", {})
    total = 0.0; denom = 0.0
    for key, value in component_scores.items():
        if enabled.get(key, True):
            weight = float(weights.get(key, 0.0)); total += float(value) * weight; denom += weight
    product_score = float(total / denom) if denom else float(np.mean(list(component_scores.values())))
    thresholds = profile.get("thresholds", {})
    critical = []
    tolerated = []
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
        if score < float(thresholds.get("critical_zone_score_max", 78.0)):
            critical.append(zone_labels.get(name, f"Detailzone {name} auffällig"))
        elif score < float(thresholds.get("warning_zone_score_max", 90.0)):
            tolerated.append(zone_labels.get(name, f"Detailzone {name} leicht auffällig, manuelle Prüfung empfohlen"))
    if structure["score"] < float(thresholds.get("failed_structure_min", 80.0)):
        critical.append("Fahrzeugposition, Skalierung, Proportion oder Struktur deutlich abweichend")
    if color_score["score"] < float(profile.get("reflection_tolerance", {}).get("tolerated_score_below", 82.0)):
        tolerated.append("Reflexionsunterschiede auf Lack-/Glasflächen oder flächige Lichtabweichung erkannt")
    if product_score < float(thresholds.get("failed_product_integrity_min", 80.0)) or structure["score"] < float(thresholds.get("failed_structure_min", 80.0)) or detail["score"] < float(thresholds.get("failed_detail_min", 80.0)):
        decision = "failed"
    elif product_score >= float(thresholds.get("passed_product_integrity_min", 90.0)) and structure["score"] >= float(thresholds.get("passed_structure_min", 90.0)) and detail["score"] >= float(thresholds.get("passed_detail_min", 88.0)) and not critical and not tolerated:
        decision = "passed"
    else:
        decision = "warning"
    debug_paths = {}; debug_paths.update(structure["debug_paths"]); debug_paths.update(detail["debug_paths"]); debug_paths.update(color_score["debug_paths"])
    return {
        "structure_only_score": structure["score"],
        "detail_zones_score": detail["score"],
        "color_reflection_score": color_score["score"],
        "product_integrity_score": product_score,
        "product_integrity_decision": decision,
        "critical_findings": critical,
        "tolerated_findings": tolerated,
        "product_integrity_debug_paths": debug_paths,
        "detail_zone_scores": zone_scores,
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
    combined_weight_map = mercedes_weight_map * reflection_weight_map if mercedes_profile_enabled else mercedes_weight_map
    if weighted_scope_mask is not None:
        combined_weight_map = np.where(weighted_scope_mask, combined_weight_map, 0.0)
    weighted_debug = compute_weighted_lpips_from_map(
        lpips_map,
        combined_weight_map,
        mask=weighted_scope_mask,
        eps=eps,
        return_debug=True,
    )
    weighted_mercedes_lpips = weighted_debug["value"]
    reflection_robust_lpips = weighted_mercedes_lpips
    weighted_mercedes_lpips_similarity_percent = convert_lpips_to_similarity_percent(weighted_mercedes_lpips)
    reflection_robust_lpips_similarity_percent = convert_lpips_to_similarity_percent(reflection_robust_lpips)
    final_similarity_score = reflection_robust_lpips_similarity_percent
    active_weights = combined_weight_map[combined_weight_map > 0]
    reflection_weight_mean = float(np.mean(active_weights)) if active_weights.size else None
    reflection_weight_min = float(np.min(active_weights)) if active_weights.size else None
    downweight_threshold = float(mercedes_weight_profile.get("downweight_threshold", 0.75))
    reflection_scope_bool = prepare_metric_mask(reflection_scope_mask, ref_norm, gen_norm)
    if reflection_scope_bool is None:
        reflection_scope_bool = reflection_weight_map > 0
    reflection_downweight_area_ratio = float(np.sum(reflection_scope_bool & (reflection_weight_map > 0) & (reflection_weight_map < downweight_threshold)) / max(float(np.sum(reflection_scope_bool)), 1.0))
    scope_area = max(float(np.sum(reflection_scope_bool)), 1.0)
    glass_interior_area_ratio = float(np.sum(glass_region_masks["interior"]) / scope_area)
    glass_contour_area_ratio = float(np.sum(glass_region_masks["contour"]) / scope_area)
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
    glass_interior_mask_path = None
    window_contour_mask_path = None
    weighted_lpips_map_path = None
    effective_weight_map_path = None
    if debug_dir:
        debug_path = Path(debug_dir)
        stem = Path(ref_path).stem
        reflection_weight_map_path = str(debug_path / f"{stem}_reflection_downweight.png")
        mercedes_weight_map_path = str(debug_path / f"{stem}_mercedes_weight.png")
        glass_interior_mask_path = str(debug_path / f"{stem}_glass_interior_mask.png")
        window_contour_mask_path = str(debug_path / f"{stem}_window_contour_mask.png")
        weighted_lpips_map_path = str(debug_path / f"{stem}_weighted_lpips_map.png")
        effective_weight_map_path = str(debug_path / f"{stem}_weighted_lpips_effective_weight.png")
        save_weight_debug_map(reflection_weight_map, Path(reflection_weight_map_path))
        save_weight_debug_map(mercedes_weight_map, Path(mercedes_weight_map_path))
        save_mask_image(glass_region_masks["interior"], Path(glass_interior_mask_path))
        save_mask_image(glass_region_masks["contour"], Path(window_contour_mask_path))
        save_weight_debug_map(weighted_debug["weighted_map"], Path(weighted_lpips_map_path))
        save_weight_debug_map(weighted_debug["effective_weight_map"], Path(effective_weight_map_path))

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
    print(f"  Weighted Mercedes LPIPS : {weighted_mercedes_lpips:.6f}")
    print(f"  Weighted Mercedes LPIPS % : {format_percent(weighted_mercedes_lpips_similarity_percent)}")
    print(f"  Weighted raw direction : LPIPS-Distanz niedriger ist besser; UI-% höher ist besser")
    print(f"  Weighted active area   : {weighted_debug['active_area_ratio'] * 100.0:.2f}%")
    print(f"  Glass interior excluded: {glass_interior_area_ratio * 100.0:.2f}%")
    print(f"  Reflection robust LPIPS : {reflection_robust_lpips:.6f}")
    print(f"  Final similarity score  : {format_percent(final_similarity_score)}")
    print(f"  Product Integrity Score : {product_integrity['product_integrity_score']:.2f}%")
    print(f"  Product Integrity       : {product_integrity['product_integrity_decision']}")
    print(f"  Structure-only Score    : {product_integrity['structure_only_score']:.2f}%")
    print(f"  Detail-zones Score      : {product_integrity['detail_zones_score']:.2f}%")
    print(f"  Color-reflection Score  : {product_integrity['color_reflection_score']:.2f}%")
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
        "weighted_mercedes_lpips": weighted_mercedes_lpips,
        "weighted_mercedes_lpips_similarity_percent": weighted_mercedes_lpips_similarity_percent,
        "reflection_robust_lpips": reflection_robust_lpips,
        "reflection_robust_lpips_similarity_percent": reflection_robust_lpips_similarity_percent,
        "final_similarity_score": final_similarity_score,
        "lpips_raw": lpips_val,
        "lpips_score": percent_metrics["lpips_similarity_percent"],
        "car_only_lpips_raw": car_metrics["lpips_car_only"],
        "car_only_lpips_score": lpips_car_only_similarity_percent,
        "structure_only_score": product_integrity["structure_only_score"],
        "detail_zones_score": product_integrity["detail_zones_score"],
        "color_reflection_score": product_integrity["color_reflection_score"],
        "product_integrity_score": product_integrity["product_integrity_score"],
        "product_integrity_decision": product_integrity["product_integrity_decision"],
        "critical_findings": json.dumps(product_integrity["critical_findings"], ensure_ascii=False),
        "tolerated_findings": json.dumps(product_integrity["tolerated_findings"], ensure_ascii=False),
        "product_integrity_profile": product_integrity_profile.get("name", DEFAULT_PRODUCT_INTEGRITY_PROFILE_NAME),
        "product_integrity_debug_paths": json.dumps(product_integrity["product_integrity_debug_paths"], sort_keys=True),
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
        "glass_interior_mask_path": glass_interior_mask_path,
        "window_contour_mask_path": window_contour_mask_path,
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
                "weighted_mercedes_lpips",
                "reflection_robust_lpips",
                "final_similarity_score",
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
