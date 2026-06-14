import numpy as np

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
