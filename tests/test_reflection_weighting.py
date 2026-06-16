import csv
import json
from pathlib import Path

import numpy as np

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


def test_product_critical_regions_keep_high_mercedes_weight():
    ref = np.full((100, 160, 3), 0.4, dtype=np.float32)
    gen = ref.copy()
    ref[50:55, 70:90] = 0.05
    gen[50:55, 70:90] = 0.05

    weights = im.build_mercedes_importance_map(ref, gen)

    grill_star_zone = float(np.mean(weights[45:62, 65:95]))
    plain_paint_zone = float(np.mean(weights[20:35, 15:35]))
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
        "weighted_mercedes_lpips": 0.15,
        "reflection_robust_lpips": 0.12,
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

    assert "weighted_mercedes_lpips" in row
    assert "reflection_robust_lpips" in row
    assert "final_similarity_score" in row
    assert "raw_lpips" in row
    assert "car_only_lpips" in row
    assert "mercedes_profile_enabled" in row
    assert "used_weight_profile" in row

    handler = MetricsHandler.__new__(MetricsHandler)
    payload = handler.build_preview_payload(row, include_previews=True)
    assert payload["weighted_mercedes_lpips"] == row["weighted_mercedes_lpips"]
    assert payload["reflection_robust_lpips"] == row["reflection_robust_lpips"]
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
    ref[48:66, 30:48] = 0.06
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
