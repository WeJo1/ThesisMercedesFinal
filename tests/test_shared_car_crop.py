import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import image_metrics as im


def test_shared_vehicle_crop_box_applies_to_ref_gen_and_mask():
    ref = np.zeros((90, 160, 3), dtype=np.float32)
    gen = np.ones((90, 160, 3), dtype=np.float32)
    mask = np.zeros((90, 160), dtype=bool)
    mask[42:83, 56:151] = True

    crop_box = im.compute_shared_vehicle_crop_box(mask, pad_px=11, min_size_px=1, square=False)
    ref_crop = im.apply_shared_crop(ref, crop_box)
    gen_crop = im.apply_shared_crop(gen, crop_box)
    mask_crop = im.apply_shared_crop(mask, crop_box)

    assert crop_box == (45, 31, 160, 90)
    assert ref_crop.shape[:2] == gen_crop.shape[:2] == mask_crop.shape


def test_crop_box_json_dimensions_match_generated_crop_files(tmp_path):
    crop_box = (567, 428, 1518, 829)
    ref = np.zeros((900, 1600, 3), dtype=np.float32)
    gen = np.ones((900, 1600, 3), dtype=np.float32)
    mask = np.zeros((900, 1600), dtype=bool)
    mask[428:829, 567:1518] = True

    crops = {
        "ref_car_only.png": im.apply_shared_crop(ref, crop_box),
        "gen_car_only.png": im.apply_shared_crop(gen, crop_box),
        "crop_mask.png": im.apply_shared_crop(mask, crop_box),
    }
    (tmp_path / "crop_box.json").write_text(
        json.dumps({"x0": crop_box[0], "y0": crop_box[1], "x1": crop_box[2], "y1": crop_box[3]}),
        encoding="utf-8",
    )
    for name, arr in crops.items():
        if arr.ndim == 2:
            Image.fromarray(arr.astype(np.uint8) * 255, mode="L").save(tmp_path / name)
        else:
            im.np_to_pil_uint8(arr).save(tmp_path / name)

    expected_size = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])
    for name in crops:
        assert Image.open(tmp_path / name).size == expected_size


def test_shared_crop_ignores_alpha_black_background_and_transparency():
    crop_box = (10, 5, 55, 35)
    ref = np.zeros((50, 70, 4), dtype=np.float32)
    gen = np.zeros((50, 70, 4), dtype=np.float32)
    mask = np.zeros((50, 70), dtype=bool)
    mask[5:35, 10:55] = True
    ref[12:20, 18:30, 3] = 1.0
    gen[6:34, 11:54, :3] = 0.0
    gen[6:34, 11:54, 3] = 0.2

    assert im.apply_shared_crop(ref, crop_box).shape[:2] == (30, 45)
    assert im.apply_shared_crop(gen, crop_box).shape[:2] == (30, 45)
    assert im.apply_shared_crop(mask, crop_box).shape == (30, 45)


def test_size_validation_aborts_with_clear_message_for_mismatched_car_crops():
    with pytest.raises(ValueError, match="Car-Only-Crop.*Größenvalidierung fehlgeschlagen"):
        im.validate_same_spatial_size(
            {
                "ref_car_only": np.zeros((401, 951, 3), dtype=np.float32),
                "gen_car_only": np.zeros((402, 943, 3), dtype=np.float32),
                "crop_mask": np.zeros((401, 951), dtype=bool),
            },
            "Car-Only-Crop",
        )
