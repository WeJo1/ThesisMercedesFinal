import numpy as np
import pytest

from image_metrics import compute_reflection_tolerant_lpips_scores, compute_structure_integrity_score, normalize_metric_mask
from mercedes_weighting import MercedesWeightMapBuilder


def rect_mask(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def make_vehicle_scene():
    h, w = 80, 120
    ref = np.ones((h, w, 3), dtype=np.float32) * 0.72
    cand = ref.copy()
    vehicle = rect_mask((h, w), 20, 65, 15, 105)
    ref[vehicle > 0] = [0.45, 0.45, 0.47]
    cand[vehicle > 0] = [0.45, 0.45, 0.47]
    # Add stable structural lines.
    ref[38:40, 20:100] = 0.1
    cand[38:40, 20:100] = 0.1
    return ref, cand, vehicle


def test_background_outside_vehicle_has_zero_weight():
    ref, cand, vehicle = make_vehicle_scene()
    result = MercedesWeightMapBuilder().build(ref, cand, vehicle)
    assert np.all(result["weight_map"][vehicle == 0] == 0.0)


def test_max_weight_keeps_structure_high_inside_reflection_zone():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 30, 50, 25, 95)
    side_line = rect_mask(vehicle.shape, 38, 40, 25, 95)
    result = MercedesWeightMapBuilder().build(
        ref,
        cand,
        vehicle,
        optional_part_masks={
            "paint_reflection_zone_mask": reflection,
            "side_character_line_mask": side_line,
        },
    )
    weights = result["weight_map"]
    assert float(weights[34, 30]) == 0.25
    assert float(weights[39, 30]) >= 1.0


def test_brand_critical_emblem_uses_highest_weight():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 30, 50, 25, 95)
    emblem = rect_mask(vehicle.shape, 35, 42, 56, 64)
    result = MercedesWeightMapBuilder().build(
        ref,
        cand,
        vehicle,
        optional_part_masks={
            "paint_reflection_zone_mask": reflection,
            "emblem_mask": emblem,
        },
        model_profile="generic_mercedes",
    )
    assert np.all(result["weight_map"][emblem > 0] == 1.5)
    assert result["diagnostic_info"]["coverage_ratio_within_vehicle"]["brand_critical_zone"] > 0.0


def test_amg_profile_marks_amg_masks_as_brand_critical():
    ref, cand, vehicle = make_vehicle_scene()
    amg_badge = rect_mask(vehicle.shape, 50, 55, 80, 90)
    result = MercedesWeightMapBuilder().build(
        ref,
        cand,
        vehicle,
        optional_part_masks={"amg_badges": amg_badge},
        model_profile="amg",
    )
    assert np.all(result["weight_map"][amg_badge > 0] == 1.5)


def test_reflection_change_region_can_stay_low_when_structure_is_unchanged():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 30, 50, 25, 95)
    cand[reflection > 0] = [0.9, 0.9, 0.95]
    result = MercedesWeightMapBuilder().build(
        ref,
        cand,
        vehicle,
        optional_part_masks={"paint_reflection_zone_mask": reflection},
    )
    low_weight_pixels = result["weight_map"][(reflection > 0) & (result["structure_mask"] == 0)]
    assert low_weight_pixels.size > 0
    assert np.all(low_weight_pixels == 0.25)


def test_unknown_profile_fails_clearly():
    ref, cand, vehicle = make_vehicle_scene()
    try:
        MercedesWeightMapBuilder().build(ref, cand, vehicle, model_profile="unknown")
    except ValueError as exc:
        assert "Unbekanntes Mercedes-Profil" in str(exc)
    else:
        raise AssertionError("unknown profile should fail")


def test_normalize_metric_mask_binarizes_255_mask_safely():
    mask = np.array([[0, 128], [255, 1]], dtype=np.uint8)
    normalized = normalize_metric_mask(mask, (2, 2), "vehicle_mask")
    assert normalized.dtype == np.float32
    assert set(np.unique(normalized).tolist()) <= {0.0, 1.0}
    assert normalized[0, 1] == 1.0


def test_structure_integrity_worsens_when_character_line_moves():
    ref, cand, vehicle = make_vehicle_scene()
    stable = compute_structure_integrity_score(ref, cand, vehicle, vehicle, dilation_radius=2)

    moved = cand.copy()
    moved[38:40, 20:100] = [0.45, 0.45, 0.47]
    moved[44:46, 20:100] = 0.1
    shifted = compute_structure_integrity_score(ref, moved, vehicle, vehicle, dilation_radius=2)

    assert stable["score"] < 0.05
    assert shifted["score"] > stable["score"]


class PixelDifferenceModel:
    """Deterministic LPIPS stand-in: return per-pixel absolute RGB difference."""

    def __call__(self, ref_t, gen_t):
        return (ref_t - gen_t).abs().mean(dim=1, keepdim=True) / 2.0


def rect_mask(shape, y0, y1, x0, x1):
    mask = np.zeros(shape, dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask


def make_vehicle_scene():
    h, w = 96, 144
    ref = np.ones((h, w, 3), dtype=np.float32) * 0.78
    cand = ref.copy()
    vehicle = rect_mask((h, w), 25, 72, 16, 128)
    ref[vehicle > 0] = [0.42, 0.43, 0.45]
    cand[vehicle > 0] = [0.42, 0.43, 0.45]
    # Product and structure features.
    ref[45:47, 25:119] = 0.09  # side character line
    cand[45:47, 25:119] = 0.09
    ref[35:42, 104:119] = [0.93, 0.93, 0.72]  # headlight
    cand[35:42, 104:119] = [0.93, 0.93, 0.72]
    ref[39:45, 66:78] = [0.08, 0.08, 0.08]  # emblem/star proxy
    cand[39:45, 66:78] = [0.08, 0.08, 0.08]
    ref[36:50, 54:90] = [0.12, 0.12, 0.13]  # grille/front panel
    cand[36:50, 54:90] = [0.12, 0.12, 0.13]
    ref[58:70, 31:48] = [0.05, 0.05, 0.055]  # wheel
    cand[58:70, 31:48] = [0.05, 0.05, 0.055]
    return ref, cand, vehicle


def score_pair(ref, cand, vehicle, masks=None, profile="generic_mercedes", thresholds=None):
    builder = MercedesWeightMapBuilder()
    built = builder.build(ref, cand, vehicle, optional_part_masks=masks or {}, model_profile=profile)
    result = compute_reflection_tolerant_lpips_scores(
        ref,
        cand,
        vehicle,
        built["weight_map"],
        built["product_detail_mask"],
        built["structure_mask"],
        PixelDifferenceModel(),
        thresholds=thresholds or {
            "reflection_tolerant_lpips": 0.12,
            "product_detail_lpips": 0.005,
            "structure_integrity_score": 0.12,
            "standard_vehicle_lpips_warning": 0.08,
        },
        structure_dilation_radius=3,
    )
    result["builder"] = built
    return result


def test_background_only_change_stays_very_low_inside_vehicle_mask():
    ref, cand, vehicle = make_vehicle_scene()
    cand[vehicle == 0] = [0.05, 0.7, 0.2]
    result = score_pair(ref, cand, vehicle)
    assert result["standard_vehicle_lpips"] == pytest.approx(0.0)
    assert result["reflection_tolerant_lpips"] == pytest.approx(0.0)
    assert result["product_detail_lpips"] == pytest.approx(0.0)


def test_soft_reflection_on_broad_paint_is_downweighted_and_structure_stays_ok():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 50, 61, 55, 105)
    cand[reflection > 0] = [0.86, 0.9, 1.0]
    result = score_pair(ref, cand, vehicle, {"paint_reflection_door_surface": reflection})
    assert result["standard_vehicle_lpips"] > result["reflection_tolerant_lpips"]
    assert result["reflection_tolerant_lpips"] < 0.08
    assert result["structure_integrity_score"] < 0.12


def test_reflection_crossing_character_line_downweights_paint_but_protects_line():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 39, 55, 46, 112)
    line = rect_mask(vehicle.shape, 45, 47, 25, 119)
    cand[(reflection > 0) & (line == 0)] = [0.88, 0.7, 0.95]
    result = score_pair(ref, cand, vehicle, {"paint_reflection_door_surface": reflection, "side_character_line_mask": line})
    weights = result["builder"]["weight_map"]
    low_reflection_weights = weights[(reflection > 0) & (line == 0) & (result["builder"]["structure_mask"] == 0)]
    assert low_reflection_weights.size > 0
    assert np.all(low_reflection_weights == pytest.approx(0.25))
    assert float(weights[46, 55]) >= 1.0
    assert result["structure_integrity_score"] < 0.14


def test_changed_side_character_line_fails_structure_even_on_low_weight_paint():
    ref, cand, vehicle = make_vehicle_scene()
    reflection = rect_mask(vehicle.shape, 39, 55, 46, 112)
    line = rect_mask(vehicle.shape, 45, 47, 25, 119)
    cand[45:47, 25:119] = [0.42, 0.43, 0.45]
    cand[50:52, 25:119] = 0.09
    result = score_pair(ref, cand, vehicle, {"paint_reflection_door_surface": reflection, "side_character_line_mask": line})
    assert result["structure_integrity_score"] > 0.11
    assert result["final_similarity_status"] == "fail"


@pytest.mark.parametrize(
    "mask_name, mask, mutate",
    [
        ("headlights", rect_mask((96, 144), 35, 42, 104, 119), lambda c, m: c.__setitem__(m > 0, [0.15, 0.15, 0.15])),
        ("emblem", rect_mask((96, 144), 39, 45, 66, 78), lambda c, m: c.__setitem__(m > 0, [0.42, 0.43, 0.45])),
        ("front_grille_geometry", rect_mask((96, 144), 36, 50, 54, 90), lambda c, m: c.__setitem__(m > 0, [0.55, 0.55, 0.58])),
        ("wheel_design", rect_mask((96, 144), 58, 70, 31, 48), lambda c, m: c.__setitem__(m > 0, [0.65, 0.65, 0.66])),
    ],
)

def test_product_detail_changes_increase_product_lpips_and_fail(mask_name, mask, mutate):
    ref, cand, vehicle = make_vehicle_scene()
    mutate(cand, mask)
    result = score_pair(ref, cand, vehicle, {mask_name: mask})
    assert result["product_detail_lpips"] > 0.005
    assert result["final_similarity_status"] == "fail"


def test_changed_silhouette_warns_or_fails_structure_integrity():
    ref, cand, vehicle = make_vehicle_scene()
    silhouette = rect_mask(vehicle.shape, 25, 29, 16, 128)
    cand[25:30, 70:110] = 0.78
    result = score_pair(ref, cand, vehicle, {"roofline_silhouette": silhouette})
    assert result["structure_integrity_score"] > 0.11
    assert result["final_similarity_status"] == "fail"


@pytest.mark.parametrize(
    "profile, mask_name",
    [
        ("amg", "amg_grille_pattern"),
        ("maybach", "maybach_grille_treatment"),
        ("eq_electric", "closed_front_panel"),
    ],
)

def test_model_profile_specific_details_are_detected(profile, mask_name):
    ref, cand, vehicle = make_vehicle_scene()
    mask = rect_mask(vehicle.shape, 36, 50, 54, 90)
    cand[mask > 0] = [0.62, 0.62, 0.68]
    result = score_pair(ref, cand, vehicle, {mask_name: mask}, profile=profile)
    assert result["product_detail_lpips"] > 0.005 or result["structure_integrity_score"] > 0.12
    assert result["final_similarity_status"] == "fail"
