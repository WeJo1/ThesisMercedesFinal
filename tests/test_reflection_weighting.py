import csv
import json
from pathlib import Path

import numpy as np
import pytest

import image_metrics as im
from gui_server import MetricsHandler


def test_reflection_areas_are_downweighted_but_edges_stay_high():
    ref = np.full((40, 60, 3), 0.45, dtype=np.float32)
    gen = ref.copy()
    ref[:, 5:30] = 0.93
    gen[:, 5:30] = 0.96
    ref[:, 40:43] = 0.05
    gen[:, 40:43] = 0.05

    weights = im.build_reflection_downweight_map(ref, gen)

    assert float(np.mean(weights[10:30, 10:25])) < 0.80
    assert float(np.mean(weights[10:30, 39:44])) > 0.78


def test_reflection_downweight_detects_soft_vehicle_reflection_change():
    ref = np.full((80, 140, 3), 0.42, dtype=np.float32)
    gen = ref.copy()
    car_mask = np.zeros((80, 140), dtype=bool)
    car_mask[18:62, 15:125] = True
    ref[48:58, 32:108] = [0.46, 0.48, 0.50]
    gen[48:58, 32:108] = [0.66, 0.68, 0.74]
    ref[:, :10] = 1.0
    gen[:, :10] = 0.0

    weights = im.build_reflection_downweight_map(ref, gen, car_mask=car_mask)
    downweighted = car_mask & (weights > 0) & (weights < im.DEFAULT_MERCEDES_WEIGHT_PROFILE["downweight_threshold"])

    assert float(np.sum(downweighted) / np.sum(car_mask)) > 0.05
    assert np.max(weights[~car_mask]) == 0


def test_reflection_color_difference_masks_background():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[:18, :] = 1.0
    gen[68:, :] = 0.0
    gen[36:48, 45:95] = np.clip(gen[36:48, 45:95] + 0.18, 0.0, 1.0)

    result = im.compute_color_reflection_score(ref, gen, mask=mask)

    assert np.max(result["map"][~mask]) == 0
    assert np.any(result["map"][mask] > 0)


def synthetic_window_pair():
    ref = np.full((100, 180, 3), 0.50, dtype=np.float32)
    gen = ref.copy()
    mask = np.zeros((100, 180), dtype=bool)
    mask[20:82, 15:165] = True
    ref[30:55, 35:85] = 0.18
    ref[30:55, 90:145] = 0.22
    gen[30:55, 35:85] = 0.72
    gen[30:55, 90:145] = 0.66
    ref[28:31, 32:148] = 0.03
    gen[28:31, 32:148] = 0.03
    ref[55:58, 35:145] = 0.03
    gen[55:58, 35:145] = 0.03
    return ref, gen, mask


def test_glass_masks_cover_window_interiors_but_keep_contours_narrow():
    ref, gen, mask = synthetic_window_pair()

    glass_masks = im.build_glass_region_masks(ref, gen, car_mask=mask)

    assert float(np.mean(glass_masks["interior"][32:53, 38:82])) > 0.55
    assert float(np.mean(glass_masks["interior"][32:53, 94:142])) > 0.55
    assert np.sum(glass_masks["contour"]) < np.sum(glass_masks["interior"])
    assert np.sum(glass_masks["interior"] & ~mask) == 0


def test_detail_zones_follow_edges_and_skip_glass_interiors():
    ref, gen, mask = synthetic_window_pair()
    glass_masks = im.build_glass_region_masks(ref, gen, car_mask=mask)

    zones = im.build_detail_zone_masks(mask, ref.shape[:2], ref=ref, gen=gen, glass_masks=glass_masks)
    detail_union = np.zeros(mask.shape, dtype=bool)
    for zone in zones.values():
        detail_union |= zone

    assert float(np.sum(detail_union) / np.sum(mask)) < 0.45
    assert np.sum(zones["window_line"] & glass_masks["contour"]) > 0
    assert np.sum(detail_union & glass_masks["interior"]) == 0


def test_window_reflection_is_tolerated_but_window_line_change_is_critical():
    ref, gen, mask = synthetic_window_pair()

    reflection_result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=88.0)

    assert reflection_result["detail_zones_score"] >= 87.9
    assert reflection_result["tolerated_findings"]

    changed = ref.copy()
    changed[28:40, 32:148] = 0.50
    line_result = im.compute_product_integrity_scores(ref, changed, car_mask=mask, car_only_lpips_score=70.0)

    assert any("Fensterlinie" in item or "Dach" in item for item in line_result["critical_findings"])
    assert line_result["product_integrity_decision"] in {"warning", "failed"}


def test_product_critical_regions_keep_high_mercedes_weight():
    ref = np.full((100, 160, 3), 0.4, dtype=np.float32)
    gen = ref.copy()
    ref[50:55, 70:90] = 0.05
    gen[50:55, 70:90] = 0.05

    weights = im.build_mercedes_importance_map(ref, gen)

    grill_star_zone = float(np.mean(weights[45:62, 65:95]))
    plain_paint_zone = float(np.mean(weights[8:15, 120:145]))
    assert grill_star_zone > plain_paint_zone
    assert grill_star_zone > 1.0


def test_glass_interior_is_damped_but_window_contour_stays_relevant():
    ref = np.full((100, 160, 3), 0.45, dtype=np.float32)
    gen = ref.copy()
    car_mask = np.zeros((100, 160), dtype=bool)
    car_mask[15:85, 10:150] = True

    ref[24:56, 36:124] = 0.22
    gen[24:56, 36:124] = 0.78
    ref[24:27, 36:124] = 0.04
    gen[24:27, 36:124] = 0.04
    ref[24:56, 36:39] = 0.04
    gen[24:56, 36:39] = 0.04

    glass_masks = im.build_glass_region_masks(ref, gen, car_mask=car_mask)
    reflection_weights = im.build_reflection_downweight_map(ref, gen, car_mask=car_mask)
    mercedes_weights = im.build_mercedes_importance_map(ref, gen, car_mask=car_mask)

    assert np.any(glass_masks["interior"])
    assert np.any(glass_masks["contour"])
    assert float(np.mean(reflection_weights[glass_masks["interior"]])) == 0.0
    assert float(np.mean(reflection_weights[glass_masks["contour"]])) >= 0.90
    assert float(np.mean(mercedes_weights[glass_masks["contour"]])) > float(np.mean(mercedes_weights[glass_masks["interior"]]))


def test_weighted_lpips_differs_from_raw_when_weighting_is_active():
    dist_map = np.array([[1.0, 1.0], [0.1, 0.1]], dtype=np.float32)
    weight_map = np.array([[0.2, 0.2], [1.0, 1.0]], dtype=np.float32)

    raw = float(np.mean(dist_map))
    weighted = im.compute_weighted_lpips_from_map(dist_map, weight_map)

    assert weighted != raw
    assert weighted < raw


def test_weighted_lpips_excludes_zero_weight_regions_from_numerator_and_denominator():
    dist_map = np.array([[1.0, 1.0], [0.1, 0.1]], dtype=np.float32)
    weight_map = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    weighted = im.compute_weighted_lpips_from_map(dist_map, weight_map)

    assert np.isclose(weighted, 0.1)


def test_weighted_scope_excludes_glass_interior_but_keeps_window_contour():
    car_mask = np.ones((4, 4), dtype=bool)
    glass_interior = np.zeros((4, 4), dtype=bool)
    glass_interior[1:3, 1:3] = True

    scope = im.build_weighted_lpips_scope_mask(car_mask, glass_interior_mask=glass_interior)

    assert np.all(scope[~glass_interior])
    assert not np.any(scope[glass_interior])


def test_csv_contains_reflection_values_and_missing_paths_do_not_break(tmp_path):
    result = {
        "filename": "demo.png",
        "lpips": 0.2,
        "legacy_debug_weighted_lpips_raw": 0.15,
        "legacy_reflection_robust_lpips_raw": 0.12,
        "final_similarity_score": 88.0,
        "raw_lpips": 0.2,
        "car_only_lpips": 0.18,
        "mercedes_profile_enabled": True,
        "used_weight_profile": "test_profile",
    }

    df = im.build_result_dataframe([result])
    csv_path = tmp_path / "result.csv"
    df.to_csv(csv_path, index=False)

    with csv_path.open(newline="", encoding="utf-8") as fp:
        row = next(csv.DictReader(fp))

    assert "legacy_debug_weighted_lpips_raw" in row
    assert "legacy_reflection_robust_lpips_raw" in row
    assert "weighted_mercedes_lpips" not in row
    assert "reflection_robust_lpips" not in row
    assert "final_similarity_score" in row
    assert "raw_lpips" in row
    assert "car_only_lpips" in row
    assert "mercedes_profile_enabled" in row
    assert "used_weight_profile" in row

    handler = MetricsHandler.__new__(MetricsHandler)
    payload = handler.build_preview_payload(row, include_previews=True)
    assert payload["legacy_debug_weighted_lpips_raw"] == row["legacy_debug_weighted_lpips_raw"]
    assert payload["legacy_reflection_robust_lpips_raw"] == row["legacy_reflection_robust_lpips_raw"]
    assert payload["final_similarity_score"] == row["final_similarity_score"]
    assert payload["raw_lpips"] == row["raw_lpips"]
    assert payload["car_only_lpips"] == row["car_only_lpips"]
    assert payload["used_weight_profile"] == row["used_weight_profile"]


def test_missing_weight_profile_file_uses_default(tmp_path):
    profile = im.load_mercedes_weight_profile(config_path=tmp_path / "missing.json")

    assert profile["enabled"] is True
    assert profile["name"] == im.DEFAULT_MERCEDES_WEIGHT_PROFILE_NAME


def synthetic_car_pair():
    ref = np.full((80, 140, 3), 0.18, dtype=np.float32)
    ref[28:58, 20:120] = 0.52
    ref[20:35, 42:96] = 0.24
    ref[48:66, 66:84] = 0.06
    ref[48:66, 92:110] = 0.06
    ref[35:39, 20:36] = 0.9
    ref[35:39, 104:120] = 0.9
    mask = np.zeros((80, 140), dtype=bool)
    mask[20:66, 20:120] = True
    return ref, mask


def test_product_integrity_identical_images_pass():
    ref, mask = synthetic_car_pair()
    result = im.compute_product_integrity_scores(ref, ref.copy(), car_mask=mask, car_only_lpips_score=100.0)

    assert result["structure_only_score"] > 99
    assert result["detail_zones_score"] > 99
    assert result["color_reflection_score"] > 99
    assert result["product_integrity_decision"] == "passed"


def test_product_integrity_tolerates_brightness_reflection_change():
    ref, mask = synthetic_car_pair()
    gen = np.clip(ref * 1.15 + 0.05, 0.0, 1.0)
    gen[32:46, 50:95] = np.clip(gen[32:46, 50:95] + 0.22, 0.0, 1.0)
    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=85.0)

    assert result["structure_only_score"] >= 90
    assert result["detail_zones_score"] >= 88
    assert result["color_reflection_score"] < result["structure_only_score"]
    assert result["product_integrity_decision"] in {"passed", "warning"}
    assert result["tolerated_findings"]


def test_product_integrity_flags_detail_zone_change():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[48:66, 92:110] = 0.52
    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=70.0)

    assert result["detail_zones_score"] < 95
    assert any("Felgen" in item or "Reifen" in item for item in result["critical_findings"] + result["tolerated_findings"])
    assert result["product_integrity_decision"] in {"warning", "failed"}


def test_product_integrity_flags_shifted_vehicle_structure():
    ref, mask = synthetic_car_pair()
    gen = np.roll(ref, shift=8, axis=1)
    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=65.0)

    assert result["structure_only_score"] < 90
    assert result["product_integrity_score"] < 90
    assert result["product_integrity_decision"] in {"warning", "failed"}


def test_product_integrity_ignores_background_with_vehicle_mask():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[:18, :] = 0.95
    gen[68:, :] = 0.02
    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=100.0)

    assert result["structure_only_score"] > 99
    assert result["detail_zones_score"] > 99
    assert result["product_integrity_decision"] == "passed"


def test_structure_debug_maps_mask_background_black(tmp_path):
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[:18, ::2] = 1.0
    gen[68:, 1::2] = 0.0

    result = im.compute_structure_only_score(ref, gen, mask=mask, debug_dir=tmp_path, stem="bg_only")

    assert result["score"] > 99
    assert np.max(result["ref_edge"][~mask]) == 0
    assert np.max(result["gen_edge"][~mask]) == 0
    assert np.max(result["diff_map"][~mask]) == 0


def test_structure_score_resizes_vehicle_mask_for_metric_shape():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[:18, :] = 1.0
    small_mask = mask[::2, ::2]

    result = im.compute_structure_only_score(ref, gen, mask=small_mask)

    assert result["score"] > 99
    assert result["diff_map"].shape == ref.shape[:2]

def test_csv_and_preview_include_product_integrity_values(tmp_path):
    result = {
        "filename": "demo.png",
        "structure_only_score": 97.0,
        "detail_zones_score": 95.0,
        "color_reflection_score": 72.0,
        "product_integrity_score": 94.0,
        "product_integrity_decision": "passed",
        "critical_findings": "[]",
        "tolerated_findings": "[\"Reflexionsunterschiede erkannt\"]",
        "product_integrity_interpretation": "Die Fahrzeugstruktur ist stabil.",
        "decision_reason": "Keine critical_findings; Abweichungen nur als tolerated_findings klassifiziert.",
        "contour_warning_reason": "Keine relevante Konturabweichung erkannt.",
        "glass_mask_area_ratio": 0.12,
        "window_contour_area_ratio": 0.03,
        "detail_zone_area_ratio": 0.41,
        "structure_masked_area_ratio": 0.52,
    }
    df = im.build_result_dataframe([result])
    csv_path = tmp_path / "result.csv"
    df.to_csv(csv_path, index=False)
    with csv_path.open(newline="", encoding="utf-8") as fp:
        row = next(csv.DictReader(fp))
    handler = MetricsHandler.__new__(MetricsHandler)
    payload = handler.build_preview_payload(row, include_previews=False)

    assert row["product_integrity_decision"] == "passed"
    assert payload["structure_only_score"] == row["structure_only_score"]
    assert payload["tolerated_findings"] == row["tolerated_findings"]
    assert row["decision_reason"].startswith("Keine critical_findings")
    assert payload["product_integrity_interpretation"] == row["product_integrity_interpretation"]
    assert payload["glass_mask_area_ratio"] == row["glass_mask_area_ratio"]

def test_high_mask_overlap_downgrades_silhouette_notice():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    profile = im.merge_profile_defaults(im.DEFAULT_PRODUCT_INTEGRITY_PROFILE, {
        "contour_warning": {"critical_zone_score_max": 101.0, "warning_zone_score_max": 101.0}
    })
    result = im.compute_product_integrity_scores(
        ref,
        gen,
        car_mask=mask,
        car_only_lpips_score=98.0,
        profile=profile,
        mask_metrics={
            "mask_iou": 0.9857,
            "mask_dice": 0.9928,
            "hausdorff_norm": 0.004,
            "centroid_distance_norm": 0.001,
            "mask_area_ratio": 1.004,
        },
    )

    assert not any("Kontur" in item for item in result["critical_findings"])
    assert any("Rand" in item or "Kontur" in item for item in result["tolerated_findings"])


def test_relevant_geometry_keeps_silhouette_critical():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    profile = im.merge_profile_defaults(im.DEFAULT_PRODUCT_INTEGRITY_PROFILE, {
        "contour_warning": {"critical_zone_score_max": 101.0, "warning_zone_score_max": 101.0}
    })
    result = im.compute_product_integrity_scores(
        ref,
        gen,
        car_mask=mask,
        car_only_lpips_score=98.0,
        profile=profile,
        mask_metrics={
            "mask_iou": 0.981,
            "mask_dice": 0.991,
            "hausdorff_norm": 0.04,
            "centroid_distance_norm": 0.02,
            "mask_area_ratio": 1.08,
        },
    )

    assert any("Relevante Abweichung an Fahrzeugkontur" in item for item in result["critical_findings"])

def test_component_scores_identical_image_pass_without_critical_findings():
    ref, mask = synthetic_car_pair()

    result = im.compute_product_integrity_scores(ref, ref.copy(), car_mask=mask, car_only_lpips_score=100.0)

    assert result["headlight_score"] > 99
    assert result["front_wheel_score"] > 99
    assert result["rear_wheel_score"] > 99
    assert result["critical_findings"] == []
    assert result["product_integrity_decision"] == "passed"


def test_component_scores_fail_on_headlight_or_light_signature_change():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[33:42, 20:43] = 0.05
    gen[35:39, 20:36] = 0.15

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=92.0)

    assert result["headlight_score"] < 90
    assert any("Scheinwerfer" in item or "Lichtsignatur" in item for item in result["critical_findings"])
    assert result["product_integrity_decision"] == "failed"
    assert "Scheinwerferbereich" in result["decision_reason"]


def test_component_scores_warn_on_wheel_rim_or_tire_change():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[48:66, 66:84] = 0.55
    gen[54:62, 70:80] = 0.95

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=86.0)

    assert min(result["front_wheel_score"], result["wheel_tire_score"]) < 90
    assert any("Felgen" in item or "Reifen" in item or "Radstruktur" in item for item in result["critical_findings"] + result["tolerated_findings"])
    assert result["product_integrity_decision"] in {"warning", "failed"}


def test_component_scores_fail_and_name_headlight_and_wheel_changes():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[33:42, 20:43] = 0.05
    gen[48:66, 66:84] = 0.55

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=80.0)

    findings = " ".join(result["critical_findings"] + result["tolerated_findings"])
    assert "Scheinwerfer" in findings or "Lichtsignatur" in findings
    assert "Felgen" in findings or "Reifen" in findings or "Radstruktur" in findings
    assert result["product_integrity_decision"] == "failed"


def test_component_scores_do_not_fail_on_paint_or_glass_reflection_only():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[32:46, 70:105] = np.clip(gen[32:46, 70:105] + 0.22, 0.0, 1.0)

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=90.0)

    findings = " ".join(result["critical_findings"] + result["tolerated_findings"])
    assert "Scheinwerfer" not in findings
    assert "Felgen" not in findings and "Reifen" not in findings
    assert result["tolerated_findings"]
    assert result["product_integrity_decision"] in {"passed", "warning"}

def test_weighted_lpips_is_legacy_debug_not_main_csv_or_ui():
    html = Path("index.html").read_text(encoding="utf-8")

    assert "Mercedes LPIPS gewichtet" not in html
    assert "weightedMercedesLpips" not in html
    assert "legacy_debug_weighted_lpips_raw" in im.CSV_COLUMN_ORDER
    assert "weighted_mercedes_lpips" not in im.CSV_COLUMN_ORDER
    assert "reflection_robust_lpips" not in im.CSV_COLUMN_ORDER


def test_product_integrity_uses_interpretable_scores_not_weighted_lpips():
    ref, mask = synthetic_car_pair()
    profile = im.merge_profile_defaults(im.DEFAULT_PRODUCT_INTEGRITY_PROFILE, {})
    result = im.compute_product_integrity_scores(ref, ref.copy(), car_mask=mask, car_only_lpips_score=0.0575, profile=profile)

    expected_car_only_similarity = 94.25
    expected = (
        result["structure_only_score"] * profile["weights"]["structure_only_score"]
        + result["detail_zones_score"] * profile["weights"]["detail_zones_score"]
        + result["color_reflection_score"] * profile["weights"]["color_reflection_score"]
        + expected_car_only_similarity * profile["weights"]["car_only_lpips_score"]
    )
    assert profile["weights"].get("legacy_weighted_lpips_weight") == 0.0
    expected = expected / (
        profile["weights"]["structure_only_score"]
        + profile["weights"]["detail_zones_score"]
        + profile["weights"]["color_reflection_score"]
        + profile["weights"]["car_only_lpips_score"]
    )
    assert result["lpips_car_only_similarity_pct"] == pytest.approx(expected_car_only_similarity)
    assert result["product_integrity_score"] == pytest.approx(expected)
    assert result["product_integrity_score"] > 90.0


def test_headlight_hard_fail_overrides_good_legacy_or_car_only_scores():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[33:42, 20:43] = 0.05
    gen[35:39, 20:36] = 0.15

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=99.0)

    assert result["product_integrity_decision"] == "failed"
    assert any("Scheinwerfer" in item or "Lichtsignatur" in item for item in result["critical_findings"])


def test_reflection_only_can_pass_without_weighted_lpips_decision_dependency():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[32:42, 70:105] = np.clip(gen[32:42, 70:105] + 0.08, 0.0, 1.0)

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=96.0)

    assert result["structure_only_score"] >= 90
    assert result["detail_zones_score"] >= 88
    assert result["product_integrity_decision"] in {"passed", "warning"}
    assert not any("Scheinwerfer" in item or "Felgen" in item or "Reifen" in item for item in result["critical_findings"])

def test_mercedes_weight_map_prioritizes_front_details_and_wheels():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()

    weights = im.build_mercedes_importance_map(ref, gen, car_mask=mask)
    zones = im.build_critical_component_zones(mask, ref.shape[:2])
    paint_zone = mask.copy()
    for key in ["headlight_zone", "light_signature_zone", "grille_zone", "emblem_zone", "front_wheel_zone", "rear_wheel_zone", "tire_zone", "window_line_zone"]:
        paint_zone &= ~zones[key]

    assert float(np.mean(weights[zones["headlight_zone"]])) >= 1.60
    assert float(np.mean(weights[zones["grille_zone"]])) >= 1.60
    assert float(np.mean(weights[zones["emblem_zone"]])) >= 1.60
    assert float(np.mean(weights[zones["front_wheel_zone"]])) >= 1.45
    assert float(np.mean(weights[zones["headlight_zone"]])) > float(np.mean(weights[paint_zone]))


def test_reflection_downweight_keeps_critical_component_effective_weight_high():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[33:42, 20:43] = np.clip(gen[33:42, 20:43] + 0.35, 0.0, 1.0)

    mercedes_weights = im.build_mercedes_importance_map(ref, gen, car_mask=mask)
    reflection_weights = im.build_reflection_downweight_map(ref, gen, car_mask=mask)
    zones = im.build_critical_component_zones(mask, ref.shape[:2])
    critical_mask = zones["headlight_zone"] | zones["grille_zone"] | zones["emblem_zone"] | zones["front_wheel_zone"] | zones["rear_wheel_zone"] | zones["tire_zone"]
    protected_reflection = np.where(critical_mask, 1.0, reflection_weights)
    effective = mercedes_weights * protected_reflection

    assert float(np.min(protected_reflection[critical_mask])) == 1.0
    assert float(np.mean(effective[zones["headlight_zone"]])) >= 1.60

def test_window_contour_excludes_hood_front_and_exposes_separate_masks():
    ref, gen, mask = synthetic_window_pair()
    # Simuliere eine starke Motorhauben-/Frontkante außerhalb des Fensterbands.
    ref[58:62, 18:70] = 0.02
    gen[58:62, 18:70] = 0.02

    glass_masks = im.build_glass_region_masks(ref, gen, car_mask=mask)
    hood_front_region = np.zeros(mask.shape, dtype=bool)
    hood_front_region[56:66, 18:75] = True

    assert np.any(glass_masks["window_candidate_region"])
    assert np.any(glass_masks["surface"])
    assert np.any(glass_masks["contour"])
    assert np.any(glass_masks["line"])
    assert np.sum(glass_masks["contour"] & hood_front_region) == 0
    assert np.sum(glass_masks["line"] & hood_front_region) == 0
    assert np.sum(glass_masks["contour"] & ~glass_masks["window_candidate_region"]) == 0

def test_headlight_change_stays_visible_with_stronger_front_bumper_debug(tmp_path):
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[35:39, 20:36] = 0.05
    gen[40:58, 20:62] = 0.98

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=95.0, debug_dir=tmp_path, stem="case")

    assert result["product_integrity_decision"] == "failed"
    assert any("Scheinwerfer" in item or "Lichtsignatur" in item for item in result["critical_findings"])
    assert Path(result["product_integrity_debug_paths"]["headlight_zone_diff"]).exists()
    assert Path(result["product_integrity_debug_paths"]["heatmap_absolute_threshold"]).exists()
    assert Path(result["product_integrity_debug_paths"]["heatmap_combined"]).exists()


def test_wheel_and_tire_findings_are_consolidated_once():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[48:66, 30:48] = 0.85
    gen[48:66, 92:110] = 0.85

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=84.0)
    findings = result["critical_findings"] + result["tolerated_findings"]
    wheel_findings = [item for item in findings if "Felgen" in item or "Reifen" in item or "Radstruktur" in item]

    assert len(wheel_findings) == 1


def test_reflection_finding_requires_area_and_color_thresholds():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[32:46, 70:105] = np.clip(gen[32:46, 70:105] + 0.25, 0.0, 1.0)

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=92.0)

    assert any("Reflexions" in item or "Lichtabweichung" in item for item in result["tolerated_findings"])


def test_no_reflection_finding_without_real_reflection_evidence():
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[32:34, 70:72] = np.clip(gen[32:34, 70:72] + 0.04, 0.0, 1.0)

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=98.0)

    assert not any("Reflexions" in item or "Lichtabweichung" in item for item in result["tolerated_findings"])


def test_headlight_difference_is_rejected_as_reflection(tmp_path):
    ref, mask = synthetic_car_pair()
    gen = ref.copy()
    gen[33:42, 20:43] = 0.05
    gen[35:39, 20:36] = 0.15

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=95.0, debug_dir=tmp_path, stem="headlight")

    assert result["product_integrity_decision"] == "failed"
    assert any("Scheinwerfer" in item or "Lichtsignatur" in item for item in result["critical_findings"])
    assert not any("Reflexions" in item for item in result["tolerated_findings"])
    assert Path(result["product_integrity_debug_paths"]["reflection_rejected_due_to_critical_component"]).exists()
