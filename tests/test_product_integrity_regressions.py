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
    img[72:100, 45:78] = 0.06  # front wheel
    img[72:100, 145:178] = 0.06 # rear wheel
    img[62:66, 75:146] = 0.25 # body line
    yy, xx = np.mgrid[0:120, 0:220]
    for cx in (61, 161):
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
    changed_rim = ((xx-61)/11)**2 + ((yy-86)/11)**2 <= 1
    gen[changed_rim] = 0.18

    changed = im.compute_product_integrity_scores(ref, gen, car_mask=mask, car_only_lpips_score=80.0, debug_dir=tmp_path, stem='wheel')
    assert any('Felgen' in item for item in changed['critical_findings'] + changed['tolerated_findings'])
    assert (tmp_path / 'wheel_overlay_findings_map.png').exists()

    invalid_mask = mask.copy()
    invalid_mask[:, :95] = False
    invalid = im.compute_product_integrity_scores(ref, gen, car_mask=invalid_mask, car_only_lpips_score=80.0)
    assert not any('Felgen' in item for item in invalid['critical_findings'])
