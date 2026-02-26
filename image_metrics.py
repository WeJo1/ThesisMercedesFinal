import argparse
import json
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


def load_image(path):
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr


def np_to_pil_uint8(img):
    clipped = np.clip(img, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="RGB")


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


def init_lpips_model(net="alex", use_gpu=False):
    if lpips is None:
        print("[WARN] Paket 'lpips' nicht gefunden. Nutze Proxy-Distanz (mittlere absolute Abweichung) statt LPIPS.")
        return None

    if torch is None:
        print(
            "[WARN] 'torch' konnte nicht geladen werden "
            f"({TORCH_IMPORT_ERROR}). Nutze Proxy-Distanz (mittlere absolute Abweichung) statt LPIPS."
        )
        return None

    model = lpips.LPIPS(net=net)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def numpy_to_lpips_tensor(img):
    if torch is None:
        raise RuntimeError(f"'torch' ist nicht verfügbar: {TORCH_IMPORT_ERROR}")

    chw = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(chw).float()
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def compute_lpips(ref, gen, lpips_model, use_gpu=False):
    if lpips_model is None:
        return float(np.mean(np.abs(ref - gen)))

    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()

    with torch.no_grad():
        dist = lpips_model(ref_t, gen_t)

    return float(dist.item())


def compute_lpips_on_content(ref, gen, content_mask, lpips_model, use_gpu=False, mask_downsample="bilinear", eps=1e-8):
    if content_mask is None:
        return compute_lpips(ref, gen, lpips_model, use_gpu=use_gpu)

    if not np.any(content_mask):
        return compute_lpips(ref, gen, lpips_model, use_gpu=use_gpu)

    return masked_lpips(
        ref,
        gen,
        content_mask,
        lpips_model,
        use_gpu=use_gpu,
        mask_downsample=mask_downsample,
        eps=eps,
    )


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


def compute_mask_bbox(mask, pad_px=20):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = mask.shape
        return 0, 0, w, h

    y0 = max(0, int(np.min(ys)) - pad_px)
    y1 = min(mask.shape[0], int(np.max(ys)) + 1 + pad_px)
    x0 = max(0, int(np.min(xs)) - pad_px)
    x1 = min(mask.shape[1], int(np.max(xs)) + 1 + pad_px)
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


def downsample_mask_torch(mask_t, target_hw, mode="bilinear"):
    if torch is None:
        raise RuntimeError(f"'torch' ist nicht verfügbar: {TORCH_IMPORT_ERROR}")

    align_corners = False if mode == "bilinear" else None
    if mode == "nearest":
        resized = torch.nn.functional.interpolate(mask_t, size=target_hw, mode=mode)
    else:
        resized = torch.nn.functional.interpolate(mask_t, size=target_hw, mode=mode, align_corners=align_corners)
    return resized


def masked_lpips(ref, gen, mask, lpips_model, use_gpu=False, mask_downsample="bilinear", eps=1e-8):
    if lpips_model is None:
        masked = mask.astype(np.float32)[..., None]
        numerator = float(np.sum(np.abs(ref - gen) * masked))
        denominator = float(np.sum(masked) * ref.shape[2]) + eps
        return numerator / denominator

    ref_t = numpy_to_lpips_tensor(ref)
    gen_t = numpy_to_lpips_tensor(gen)
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]

    if use_gpu and torch.cuda.is_available():
        ref_t = ref_t.cuda()
        gen_t = gen_t.cuda()
        mask_t = mask_t.cuda()

    if not hasattr(lpips_model, "net"):
        raise RuntimeError("LPIPS-Modell unterstützt kein feature-basiertes masked LPIPS.")

    with torch.no_grad():
        feats0 = lpips_model.net.forward(ref_t)
        feats1 = lpips_model.net.forward(gen_t)

        weighted_layers = []
        for i, (f0, f1) in enumerate(zip(feats0, feats1)):
            diff = (lpips.normalize_tensor(f0) - lpips.normalize_tensor(f1)) ** 2
            if hasattr(lpips_model, "lins") and len(lpips_model.lins) > i:
                diff = lpips_model.lins[i].model(diff)
            dist_map = torch.mean(diff, dim=1, keepdim=True)
            mask_small = downsample_mask_torch(mask_t, dist_map.shape[-2:], mode=mask_downsample)
            numerator = torch.sum(dist_map * mask_small)
            denominator = torch.sum(mask_small) + eps
            weighted_layers.append(numerator / denominator)

        dist = torch.sum(torch.stack(weighted_layers))

    return float(dist.item())


def save_mask_image(mask, path):
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)


def compute_car_only_metrics(
    ref_norm,
    gen_norm,
    ref_path,
    gen_path,
    lpips_model,
    segmenter,
    car_mode="neutralize_crop",
    mask_source="ref",
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
):
    if segmenter is None:
        debug = {"mask_area_ratio": 0.0, "bbox": None, "fallback_reason": "Car-only deaktiviert"}
        return {
            "lpips_car_only": None,
            "ssim_car_only": None,
            "car_only_paths": {"ref": None, "gen": None},
            "debug": debug,
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
        }

    bbox = compute_mask_bbox(mask, pad_px=pad_px)
    debug["bbox"] = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3], "pad_px": pad_px}

    if car_mode == "neutralize_crop":
        ref_car, mask_crop = apply_neutralize_crop(ref_norm, mask, bbox, neutral_value=neutral_value)
        gen_car, _ = apply_neutralize_crop(gen_norm, mask, bbox, neutral_value=neutral_value)
        ref_preview, _ = apply_masked_car_crop(ref_norm, mask, bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, mask, bbox)
        lpips_car = compute_lpips(ref_car, gen_car, lpips_model, use_gpu=use_gpu)
        ssim_car = compute_ssim(ref_car, gen_car)
    elif car_mode == "weighted_lpips":
        x0, y0, x1, y1 = bbox
        ref_car = ref_norm[y0:y1, x0:x1, :]
        gen_car = gen_norm[y0:y1, x0:x1, :]
        mask_crop = mask[y0:y1, x0:x1]
        ref_preview, _ = apply_masked_car_crop(ref_norm, mask, bbox)
        gen_preview, _ = apply_masked_car_crop(gen_norm, mask, bbox)
        lpips_car = masked_lpips(
            ref_car,
            gen_car,
            mask_crop,
            lpips_model,
            use_gpu=use_gpu,
            mask_downsample=mask_downsample,
            eps=eps,
        )
        ssim_car = compute_ssim(ref_car, gen_car)
    else:
        raise ValueError("car_mode muss 'neutralize_crop' oder 'weighted_lpips' sein")

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
    }


def compute_delta_e(ref, gen):
    ref_lab = color.rgb2lab(ref)
    gen_lab = color.rgb2lab(gen)
    delta_map = color.deltaE_ciede2000(ref_lab, gen_lab)
    return float(np.mean(delta_map))


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


def compute_geometric_metrics(ref, gen):
    ref_mask = create_foreground_mask(ref)
    gen_mask = create_foreground_mask(gen)

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
    mask_source="ref",
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
):
    ref_img = load_image(ref_path)
    gen_img = load_image(gen_path)

    ref_h, ref_w = ref_img.shape[:2]
    gen_h, gen_w = gen_img.shape[:2]

    ref_norm, gen_norm, content_mask = normalize_pair(ref_img, gen_img, mode=mode)
    norm_h, norm_w = ref_norm.shape[:2]

    basename = Path(ref_path).stem
    ref_norm_path, gen_norm_path = save_normalized_pair(ref_norm, gen_norm, basename, out_dir)

    ssim_val = compute_ssim(ref_norm, gen_norm)
    lpips_val = compute_lpips_on_content(
        ref_norm,
        gen_norm,
        content_mask,
        lpips_model,
        use_gpu=use_gpu,
        mask_downsample=mask_downsample,
        eps=eps,
    )
    delta_e_val = compute_delta_e(ref_norm, gen_norm)
    geometric = compute_geometric_metrics(ref_norm, gen_norm)
    percent_metrics = convert_metrics_to_percent(ssim_val, lpips_val, delta_e_val)

    foreground_mask = compute_foreground_mask_union(ref_norm, gen_norm)
    if content_mask is not None:
        foreground_mask = foreground_mask & content_mask
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
    )
    lpips_car_only_similarity_percent = convert_lpips_to_similarity_percent(car_metrics["lpips_car_only"])

    print("------------------------------------------------------------")
    print(f"Pair: {basename}")
    print(f"  Reference original : {ref_w}x{ref_h}")
    print(f"  Generated original : {gen_w}x{gen_h}")
    print(f"  Normalized sizes   : {norm_w}x{norm_h}")
    print(f"  SSIM               : {ssim_val:.6f}")
    print(f"  SSIM (%)           : {percent_metrics['ssim_percent']:.2f}%")
    print(f"  LPIPS              : {lpips_val:.6f}")
    print(f"  LPIPS Similarity % : {percent_metrics['lpips_similarity_percent']:.2f}%")
    print(f"  LPIPS foreground   : {lpips_foreground:.6f}")
    print(f"  LPIPS foreground % : {format_percent(lpips_foreground_similarity_percent)}")
    if segmenter is not None:
        print(f"  Mask area (%)      : {car_metrics['debug']['mask_area_ratio'] * 100.0:.2f}%")
        print(f"  BBox (car)         : {car_metrics['debug']['bbox']}")
        print(f"  LPIPS car-only     : {car_metrics['lpips_car_only']}")
        print(f"  LPIPS car-only (%) : {format_percent(lpips_car_only_similarity_percent)}")
        if car_metrics.get("car_only_paths", {}).get("ref"):
            print(f"  Car-only Ref saved : {car_metrics['car_only_paths']['ref']}")
            print(f"  Car-only Gen saved : {car_metrics['car_only_paths']['gen']}")
    else:
        print("  Car-only           : deaktiviert (nutze Full-Image-Logik)")
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
        "normalization_mode": mode,
        "ssim": ssim_val,
        "ssim_percent": percent_metrics["ssim_percent"],
        "lpips": lpips_val,
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
    mask_source="ref",
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
        )
        results.append(result)

    if not results:
        raise RuntimeError("Keine auswertbaren Bildpaare gefunden.")

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, float_format="%.6f")

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
    parser.add_argument("--use-gpu", action="store_true", help="Nutze CUDA, falls verfügbar")
    parser.add_argument("--enable-car-only", action="store_true", help="Aktiviere Car-only Metriken (LPIPS/SSIM)")
    parser.add_argument("--car-only", action="store_true", help="Kurzform für --enable-car-only")
    parser.add_argument("--car-mode", default="neutralize_crop", choices=["neutralize_crop", "weighted_lpips"], help="Auto-fokussierte LPIPS-Berechnung")
    parser.add_argument("--mask-source", default="ref", choices=["ref", "gen", "union"], help="Quelle für die Auto-Maske")
    parser.add_argument("--pad-px", type=int, default=20, help="Padding für Car-Crop-BBox")
    parser.add_argument("--neutral-value", type=float, default=0.5, help="Neutralwert für Hintergrundpixel [0..1]")
    parser.add_argument("--min-mask-area", type=int, default=0, help="Minimale Maskenfläche in Pixel")
    parser.add_argument("--mask-downsample", default="bilinear", choices=["bilinear", "nearest"], help="Downsample-Modus der Maske für weighted LPIPS")
    parser.add_argument("--mask-grow-px", type=int, default=10, help="Erweitere die Car-Maske lokal um X Pixel")
    parser.add_argument("--mask-min-object-area", type=int, default=500, help="Entferne sehr kleine Maskeninseln")
    parser.add_argument("--mask-max-hole-area", type=int, default=3000, help="Fülle kleine Löcher in der Car-Maske")
    parser.add_argument("--mask-trim-px", type=int, default=1, help="Schneide Maskenrand um X Pixel ein für sauberere Konturen")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon für weighted LPIPS")
    parser.add_argument("--mask-score-threshold", type=float, default=0.5, help="Score-Schwelle für Vehicle-Segmentierung")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Pixel-Schwelle der Segmentierungsmaske [0..1]")
    parser.add_argument("--debug-dir", default=None, help="Optionales Debug-Verzeichnis für Masken/Crops")
    parser.add_argument("--car-only-dir", default="car_only", help="Verzeichnis für gespeicherte Car-only-Crops")

    parser.add_argument("--ref", help="Pfad zu einem einzelnen Referenzbild")
    parser.add_argument("--gen", help="Pfad zu einem einzelnen Generated-Bild")
    parser.add_argument("--reference-dir", default="reference", help="Ordner mit Referenzbildern")
    parser.add_argument("--generated-dir", default="generated", help="Ordner mit Generated-Bildern")

    args = parser.parse_args()

    use_car_specific_option = any(
        [
            args.car_mode != "neutralize_crop",
            args.mask_source != "ref",
            args.pad_px != 20,
            args.neutral_value != 0.5,
            args.min_mask_area != 0,
            args.mask_downsample != "bilinear",
            args.mask_grow_px != 10,
            args.mask_min_object_area != 500,
            args.mask_max_hole_area != 3000,
            args.mask_trim_px != 1,
            args.eps != 1e-8,
            args.mask_score_threshold != 0.5,
            args.mask_threshold != 0.5,
            args.debug_dir is not None,
        ]
    )

    args.enable_car_only = args.enable_car_only or args.car_only or use_car_specific_option
    return args


def main():
    args = parse_args()

    print("============================================================")
    print("[INFO] Starte Bildmetrik-Berechnung")
    print(f"[INFO] Mode            : {args.mode}")
    print(f"[INFO] Normalized out  : {args.out}")
    print(f"[INFO] Output CSV      : {args.output_csv}")
    print(f"[INFO] Car-only aktiv  : {args.enable_car_only}")
    print("============================================================")

    lpips_model = init_lpips_model(net=args.lpips_net, use_gpu=args.use_gpu)
    segmenter = None
    if args.enable_car_only:
        print("[INFO] Car-only wird aktiviert. Einfacher Aufruf: python image_metrics.py --car-only")
        try:
            segmenter = build_vehicle_segmenter(
                use_gpu=args.use_gpu,
                score_threshold=args.mask_score_threshold,
                mask_threshold=args.mask_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Car-only wurde deaktiviert: {exc}")
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
    )


if __name__ == "__main__":
    main()
