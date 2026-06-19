import numpy as np

import image_metrics as im


def make_car():
    img = np.full((120, 220, 3), 0.70, dtype=np.float32)
    mask = np.zeros((120, 220), dtype=bool)
    mask[36:86, 24:196] = True
    # roof/window taper
    mask[24:44, 58:160] = True
    img[mask] = 0.48
    img[28:44, 64:154] = 0.16
    img[51:55, 24:70] = 0.92  # headlight/front bright line
    img[50:72, 30:56] = 0.10  # grille/front
    img[72:100, 95:128] = 0.06  # front wheel
    img[72:100, 145:178] = 0.06 # rear wheel
    img[62:66, 75:146] = 0.25 # body line
    yy, xx = np.mgrid[0:120, 0:220]
    for cx in (111, 161):
        wheel = ((xx-cx)/16)**2 + ((yy-86)/15)**2 <= 1
        mask |= wheel
        img[wheel] = 0.05
        rim = ((xx-cx)/8)**2 + ((yy-86)/8)**2 <= 1
        img[rim] = 0.75
    return img, mask


def test_background_above_vehicle_is_excluded_from_product_maps():
    ref, mask = make_car()
    gen = ref.copy()
    gen[:22, :] = 0.0
    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=99.0)
    assert result['structure_only_score'] > 99
    assert result['product_integrity_score'] > 95
    assert not any('Felgen' in item for item in result['critical_findings'])


def test_reflection_change_scores_better_than_real_headlight_change():
    ref, mask = make_car()
    reflection = ref.copy()
    reflection[mask] = np.clip(ref[mask] + np.array([0.16, 0.10, 0.02], dtype=np.float32), 0, 1)
    reflection[28:44, 64:154] = np.clip(ref[28:44, 64:154] + 0.45, 0, 1)
    headlight = ref.copy()
    headlight[48:58, 24:82] = 0.18

    reflection_result = im.compute_product_integrity_scores(ref, reflection, car_mask=mask, car_only_lpips_score=82.0)
    headlight_result = im.compute_product_integrity_scores(ref, headlight, car_mask=mask, car_only_lpips_score=82.0)

    assert reflection_result['color_reflection_score'] < 95
    assert not any('Felgen' in item for item in reflection_result['critical_findings'])
    assert headlight_result['product_integrity_score'] < reflection_result['product_integrity_score']
    assert any('Scheinwerfer' in item for item in headlight_result['critical_findings'])


def test_wheel_finding_requires_valid_zone_and_visible_overlay(tmp_path):
    ref, mask = make_car()
    gen = ref.copy()
    yy, xx = np.mgrid[0:120, 0:220]
    changed_rim = ((xx-111)/11)**2 + ((yy-86)/11)**2 <= 1
    gen[changed_rim] = 0.18

    changed = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=80.0, debug_dir=tmp_path, stem='wheel')
    assert any('Felgen' in item for item in changed['critical_findings'] + changed['tolerated_findings'])
    assert (tmp_path / 'wheel_overlay_findings_map.png').exists()

    invalid_mask = mask.copy()
    invalid_mask[:, :95] = False
    invalid = im.compute_product_integrity_scores(ref, gen, car_mask=invalid_mask, car_only_lpips_score=80.0)
    assert not any('Felgen' in item for item in invalid['critical_findings'])


def test_massive_wheel_and_rim_deviation_cannot_pass(tmp_path):
    ref, mask = make_car()
    gen = ref.copy()
    yy, xx = np.mgrid[0:120, 0:220]
    for cx in (111, 161):
        tire = ((xx-cx)/18)**2 + ((yy-86)/16)**2 <= 1
        rim = ((xx-cx)/11)**2 + ((yy-86)/10)**2 <= 1
        spokes = tire & (((np.abs(xx-cx) < 3) | (np.abs(yy-86) < 3)))
        gen[tire] = 0.82
        gen[rim] = 0.18
        gen[spokes] = 0.95

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=91.0, debug_dir=tmp_path, stem='massive_wheel')

    assert result['product_integrity_decision'] != 'passed'
    assert result['wheel_score'] < 94
    assert (tmp_path / 'massive_wheel_front_wheel_zone.png').exists()
    assert (tmp_path / 'massive_wheel_rear_wheel_zone.png').exists()
    assert (tmp_path / 'massive_wheel_wheel_zone_combined.png').exists()
    assert (tmp_path / 'massive_wheel_wheel_score_heatmap.png').exists()
    assert (tmp_path / 'massive_wheel_critical_component_score_map.png').exists()


def test_product_integrity_debug_sanity_checks_for_lpips_direction():
    ref, mask = make_car()
    result = im.compute_product_integrity_scores(ref, ref.copy(), car_mask=mask, car_only_lpips_score=0.0575)
    debug = result["product_integrity_debug"]
    checks = {item["name"]: item for item in debug["sanity_checks"]}

    assert result["lpips_car_only_similarity_pct"] > 90.0
    assert result["product_integrity_score"] > 90.0
    assert checks["lpips_distance_below_0_10_similarity_above_90"]["passed"]
    assert checks["final_score_not_raw_lpips_distance_times_100"]["passed"]
    assert debug["raw_inputs"]["lpips_car_only_input"] == 0.0575
    assert debug["score_direction"].startswith("Alle Product-Integrity-Teilwerte")


def test_tolerated_findings_are_separate_from_critical_component_failures():
    ref, mask = make_car()
    gen = ref.copy()
    # Eine breite, glatte Farb-/Reflexionsänderung erzeugt tolerierte Hinweise, aber keinen harten Bauteil-Fail.
    gen[mask] = np.clip(ref[mask] + np.array([0.14, 0.08, 0.02], dtype=np.float32), 0, 1)

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=92.0)
    debug = result["product_integrity_debug"]
    checks = {item["name"]: item for item in debug["sanity_checks"]}

    assert not result["critical_findings"]
    assert result["critical_component_names"] == []
    assert checks["tolerated_findings_do_not_massively_reduce_score"]["passed"]
    assert all(not item.get("hard") for item in debug["score_adjustments"] if item["type"].startswith("tolerated"))



def test_color_reflection_uses_weighted_local_glass_change_debug_fields():
    ref, mask = make_car()
    gen = ref.copy()
    gen[28:44, 64:154] = np.clip(ref[28:44, 64:154] + np.array([0.18, 0.12, 0.04], dtype=np.float32), 0, 1)

    result = im.compute_color_reflection_score(ref, gen, mask=mask)
    debug = result["debug"]

    assert debug["area_weight_map_used_in_final_error"] is True
    assert debug["active_mask_area_px"] > 0
    assert debug["glass_interior_area_px"] > 0
    assert debug["final_color_reflection_error"] > 0.0
    assert debug["color_reflection_score_after_clipping"] < 100.0
    assert result["score"] == debug["color_reflection_score_after_clipping"]


def test_product_integrity_example_values_use_similarity_percent_without_hidden_cap():
    ref, mask = make_car()
    result = im.compute_product_integrity_scores(ref, ref.copy(), car_mask=mask, car_only_lpips_score=96.87)
    debug = result["product_integrity_debug"]
    expected = (0.535 * result["structure_only_score"] + 0.400 * result["detail_zones_score"] + 0.015 * result["color_reflection_score"] + 0.050 * 96.87)

    assert debug["configured_weights"] == {
        "structure_weight": 0.535,
        "detail_weight": 0.4,
        "color_reflection_weight": 0.015,
        "car_only_lpips_weight": 0.05,
    }
    assert debug["car_only_lpips_similarity_percent"] == 96.87
    assert result["product_integrity_score"] == np.float64(expected).item()
    assert debug["applied_caps"] == []
    assert debug["hidden_findings_count"] == 0


def test_product_integrity_lower_than_base_requires_visible_reason():
    ref, mask = make_car()
    gen = ref.copy()
    yy, xx = np.mgrid[0:120, 0:220]
    tire = ((xx-111)/18)**2 + ((yy-86)/16)**2 <= 1
    gen[tire] = 0.82

    result = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=96.0)
    debug = result["product_integrity_debug"]

    if debug["product_integrity_score_delta_due_to_caps"] > 0.3:
        assert debug["applied_caps"] or debug["applied_penalties"]
        assert debug["visible_findings"]
