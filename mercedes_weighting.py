"""Mercedes-Benz internal vehicle weighting masks for reflection-tolerant similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import color
from skimage.feature import canny
from skimage.filters import sobel
from skimage.morphology import binary_dilation, disk, remove_small_objects
from skimage.util import img_as_float32

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "mercedes_weight_profiles.json"


class MercedesWeightMapBuilder:
    """Build max-composed Mercedes region weight maps inside an existing vehicle mask."""

    def __init__(self, configuration: str | Path | dict[str, Any] | None = None):
        self.configuration = self._load_configuration(configuration)
        self.weights = self.configuration["weights"]
        self.heuristics = self.configuration.get("heuristics", {})

    def build(
        self,
        reference_image: np.ndarray,
        candidate_image: np.ndarray,
        vehicle_mask: np.ndarray | str | Path,
        optional_part_masks: dict[str, np.ndarray | str | Path] | None = None,
        model_profile: str = "generic_mercedes",
    ) -> dict[str, Any]:
        """Return weight_map, product_detail_mask, reflection_tolerant_mask, structure_mask and diagnostics."""
        ref = self._validate_image(reference_image, "reference_image")
        cand = self._validate_image(candidate_image, "candidate_image")
        if ref.shape != cand.shape:
            raise ValueError(f"reference_image und candidate_image brauchen dieselbe Form ({ref.shape} vs. {cand.shape}).")

        outer_mask = self.load_vehicle_mask(vehicle_mask, ref.shape[:2])
        part_masks = self._load_optional_part_masks(optional_part_masks or {}, ref.shape[:2], outer_mask)
        profile = self._resolve_profile(model_profile)

        normal_surface_mask = outer_mask > 0.0
        structure_mask = self._build_structure_mask(ref, cand, outer_mask, part_masks)
        raw_product_detail_mask = self._build_raw_product_detail_mask(profile, part_masks, outer_mask)
        brand_critical_mask = self._build_brand_critical_mask(profile, part_masks, outer_mask)
        protected_vehicle_shape_mask = structure_mask | raw_product_detail_mask | brand_critical_mask
        reflection_mask = self._build_reflection_tolerant_mask(ref, cand, outer_mask, protected_vehicle_shape_mask, part_masks)
        glass_mask = self._build_glass_mask(ref, cand, outer_mask, protected_vehicle_shape_mask, part_masks)
        product_detail_mask = (raw_product_detail_mask | structure_mask) & (outer_mask > 0.0)
        normal_surface_mask = normal_surface_mask & ~reflection_mask & ~glass_mask

        weight_map = np.zeros(outer_mask.shape, dtype=np.float32)
        self._apply_max_weight(weight_map, normal_surface_mask, self.weights["normal_vehicle_surface"])
        self._apply_max_weight(weight_map, reflection_mask, self.weights["broad_smooth_paint_reflection_zone"])
        self._apply_max_weight(weight_map, glass_mask, self.weights["glass_or_window_reflection_zone"])
        self._apply_max_weight(weight_map, structure_mask, self.weights["structural_edge_zone"])
        self._apply_max_weight(weight_map, product_detail_mask, self.weights["product_detail_zone"])
        self._apply_max_weight(weight_map, brand_critical_mask, self.weights["brand_critical_zone"])
        weight_map *= outer_mask.astype(np.float32)

        diagnostic_info = self._build_diagnostics(
            model_profile=model_profile,
            resolved_profile=profile,
            vehicle_mask=outer_mask,
            part_masks=part_masks,
            masks={
                "normal_vehicle_surface": normal_surface_mask,
                "broad_smooth_paint_reflection_zone": reflection_mask,
                "glass_or_window_reflection_zone": glass_mask,
                "structural_edge_zone": structure_mask,
                "product_detail_zone": product_detail_mask,
                "brand_critical_zone": brand_critical_mask,
            },
        )

        return {
            "weight_map": weight_map.astype(np.float32),
            "product_detail_mask": product_detail_mask.astype(np.float32),
            "reflection_tolerant_mask": (reflection_mask | glass_mask).astype(np.float32),
            "structure_mask": structure_mask.astype(np.float32),
            "diagnostic_info": diagnostic_info,
        }

    def load_vehicle_mask(self, vehicle_mask: np.ndarray | str | Path, expected_shape: tuple[int, int]) -> np.ndarray:
        """Load and validate a single-channel float vehicle mask in range 0..1."""
        mask = self._read_mask(vehicle_mask)
        if mask.shape != expected_shape:
            raise ValueError(f"vehicle_mask hat Form {mask.shape}, erwartet ist {expected_shape}.")
        return np.clip(mask.astype(np.float32), 0.0, 1.0)

    def _load_configuration(self, configuration: str | Path | dict[str, Any] | None) -> dict[str, Any]:
        if configuration is None:
            path = DEFAULT_CONFIG_PATH
            return json.loads(path.read_text(encoding="utf-8"))
        if isinstance(configuration, dict):
            return configuration
        path = Path(configuration)
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_profile(self, model_profile: str) -> dict[str, Any]:
        profiles = self.configuration.get("profiles", {})
        if model_profile not in profiles:
            raise ValueError(f"Unbekanntes Mercedes-Profil: {model_profile}. Verfügbar: {sorted(profiles)}")

        def merge(name: str) -> dict[str, Any]:
            raw = dict(profiles[name])
            parent_name = raw.pop("extends", None)
            if not parent_name:
                return raw
            parent = merge(parent_name)
            for key, value in raw.items():
                if isinstance(value, list):
                    parent[key] = list(dict.fromkeys(parent.get(key, []) + value))
                else:
                    parent[key] = value
            return parent

        resolved = merge(model_profile)
        resolved["name"] = model_profile
        return resolved

    def _validate_image(self, image: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"{name} muss die Form (H, W, 3) haben.")
        if np.min(arr) < 0.0 or np.max(arr) > 1.0:
            arr = img_as_float32(np.clip(arr, 0.0, 255.0))
        return arr

    def _read_mask(self, mask: np.ndarray | str | Path) -> np.ndarray:
        if isinstance(mask, (str, Path)):
            with Image.open(mask) as img:
                mask = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        else:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 3:
                mask = np.mean(mask, axis=2)
            if mask.max(initial=0.0) > 1.0:
                mask = mask / 255.0
        if mask.ndim != 2:
            raise ValueError("Masken müssen zweidimensional oder als Bildpfad übergeben werden.")
        return np.clip(mask, 0.0, 1.0)

    def _load_optional_part_masks(self, masks: dict[str, Any], shape: tuple[int, int], outer_mask: np.ndarray) -> dict[str, np.ndarray]:
        loaded = {}
        for name, mask in masks.items():
            arr = self._read_mask(mask)
            if arr.shape != shape:
                raise ValueError(f"Part-Maske '{name}' hat Form {arr.shape}, erwartet ist {shape}.")
            loaded[name] = (arr > 0.5) & (outer_mask > 0.0)
        return loaded

    def _build_structure_mask(self, ref: np.ndarray, cand: np.ndarray, outer_mask: np.ndarray, part_masks: dict[str, np.ndarray]) -> np.ndarray:
        gray = np.maximum(color.rgb2gray(ref), color.rgb2gray(cand))
        sigma = float(self.heuristics.get("edge_sigma", 1.2))
        edges = canny(gray, sigma=sigma) & (outer_mask > 0.0)
        for name, mask in part_masks.items():
            if self._is_structure_mask_name(name):
                edges |= mask
        radius = int(self.heuristics.get("structure_dilation_px", 2))
        if radius > 0:
            edges = binary_dilation(edges, footprint=disk(radius))
        return edges & (outer_mask > 0.0)

    def _build_reflection_tolerant_mask(self, ref: np.ndarray, cand: np.ndarray, outer_mask: np.ndarray, protected_vehicle_shape_mask: np.ndarray, part_masks: dict[str, np.ndarray]) -> np.ndarray:
        manual = self._union_masks(part_masks, ["paint_reflection", "reflection", "smooth_paint", "door_surface", "hood_center", "roof_paint", "side_panel", "bumper_paint", "mirror_cap"])
        gray_ref = color.rgb2gray(ref)
        gray_cand = color.rgb2gray(cand)
        texture = np.maximum(sobel(gray_ref), sobel(gray_cand))
        edge_density_proxy = texture
        vehicle_pixels = outer_mask > 0.0
        if np.any(vehicle_pixels):
            texture_limit = np.percentile(texture[vehicle_pixels], float(self.heuristics.get("low_texture_percentile", 35)))
            edge_limit = np.percentile(edge_density_proxy[vehicle_pixels], float(self.heuristics.get("low_edge_percentile", 45)))
        else:
            texture_limit = edge_limit = 0.0
        heuristic = (texture <= texture_limit) & (edge_density_proxy <= edge_limit) & vehicle_pixels
        heuristic = remove_small_objects(heuristic, min_size=int(self.heuristics.get("minimum_reflection_component_area_px", 32)))
        return (manual | heuristic) & vehicle_pixels & ~protected_vehicle_shape_mask

    def _build_glass_mask(self, ref: np.ndarray, cand: np.ndarray, outer_mask: np.ndarray, protected_vehicle_shape_mask: np.ndarray, part_masks: dict[str, np.ndarray]) -> np.ndarray:
        manual = self._union_masks(part_masks, ["glass", "window", "windshield", "side_window", "rear_window", "glossy_black_trim"])
        hsv = color.rgb2hsv((ref + cand) / 2.0)
        h, _ = outer_mask.shape
        upper = np.zeros_like(outer_mask, dtype=bool)
        upper[: int(h * float(self.heuristics.get("glass_upper_vehicle_fraction", 0.58))), :] = True
        heuristic = (
            (hsv[..., 2] <= float(self.heuristics.get("glass_dark_luminance_max", 0.38)))
            & (hsv[..., 1] <= float(self.heuristics.get("glass_saturation_max", 0.35)))
            & upper
            & (outer_mask > 0.0)
        )
        return (manual | heuristic) & (outer_mask > 0.0) & ~protected_vehicle_shape_mask

    def _build_raw_product_detail_mask(self, profile: dict[str, Any], part_masks: dict[str, np.ndarray], outer_mask: np.ndarray) -> np.ndarray:
        detail_names = profile.get("product_detail_masks", [])
        return self._union_exact_or_contains(part_masks, detail_names) & (outer_mask > 0.0)

    def _build_brand_critical_mask(self, profile: dict[str, Any], part_masks: dict[str, np.ndarray], outer_mask: np.ndarray) -> np.ndarray:
        brand_names = profile.get("brand_critical_masks", [])
        return self._union_exact_or_contains(part_masks, brand_names) & (outer_mask > 0.0)

    def _union_masks(self, part_masks: dict[str, np.ndarray], tokens: list[str]) -> np.ndarray:
        if not part_masks:
            return np.zeros((1, 1), dtype=bool)
        shape = next(iter(part_masks.values())).shape
        merged = np.zeros(shape, dtype=bool)
        for name, mask in part_masks.items():
            normalized = name.lower()
            if any(token in normalized for token in tokens):
                merged |= mask
        return merged

    def _union_exact_or_contains(self, part_masks: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
        if not part_masks:
            return np.zeros((1, 1), dtype=bool)
        shape = next(iter(part_masks.values())).shape
        merged = np.zeros(shape, dtype=bool)
        wanted = [name.lower() for name in names]
        for name, mask in part_masks.items():
            normalized = name.lower()
            if any(normalized == item or item in normalized or normalized in item for item in wanted):
                merged |= mask
        return merged

    def _is_structure_mask_name(self, name: str) -> bool:
        tokens = ["silhouette", "roofline", "shoulder", "beltline", "character", "arch", "crease", "panel_gap", "contour", "edge", "line"]
        lowered = name.lower()
        return any(token in lowered for token in tokens)

    def _apply_max_weight(self, weight_map: np.ndarray, mask: np.ndarray, weight: float) -> None:
        if mask.shape != weight_map.shape:
            return
        weight_map[mask.astype(bool)] = np.maximum(weight_map[mask.astype(bool)], float(weight))

    def _coverage(self, mask: np.ndarray, vehicle_mask: np.ndarray) -> float:
        denom = float(np.sum(vehicle_mask > 0.0))
        if denom <= 0.0:
            return 0.0
        return float(np.sum(mask.astype(bool) & (vehicle_mask > 0.0)) / denom)

    def _build_diagnostics(self, model_profile: str, resolved_profile: dict[str, Any], vehicle_mask: np.ndarray, part_masks: dict[str, np.ndarray], masks: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            "selected_model_profile": model_profile,
            "resolved_product_detail_zones": resolved_profile.get("product_detail_masks", []),
            "resolved_brand_critical_zones": resolved_profile.get("brand_critical_masks", []),
            "used_manual_masks": sorted(part_masks.keys()),
            "heuristic_masks": ["broad_smooth_paint_reflection_zone", "glass_or_window_reflection_zone", "structural_edge_zone"],
            "approximated_heuristically": {
                "broad_smooth_paint_reflection_zone": not any("reflection" in name.lower() or "paint" in name.lower() for name in part_masks),
                "glass_or_window_reflection_zone": not any("glass" in name.lower() or "window" in name.lower() for name in part_masks),
                "structural_edge_zone": True,
            },
            "coverage_ratio_within_vehicle": {name: self._coverage(mask, vehicle_mask) for name, mask in masks.items()},
            "weight_rule": "final_pixel_weight_is_maximum_relevant_weight",
        }
