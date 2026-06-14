# Mercedes-Benz reflection-tolerant vehicle similarity pipeline

Use this pipeline when a Mercedes-Benz vehicle is inserted into a different digital background and the evaluation should not fail just because the environment creates different paint or glass reflections.

The pipeline **does not remove reflections from the image**. Instead, it makes the similarity evaluation more robust by downweighting reflection-like areas while protecting Mercedes-relevant structural and product details.

## Conceptual score differences

| Score | What it measures | Use it for | Failure meaning |
| --- | --- | --- | --- |
| `standard_vehicle_lpips` | Strict visual difference inside the vehicle mask. Every vehicle pixel has normal importance. | Detect any visible change, including reflections, color shifts, edits and artifacts. | Something changed on the vehicle area. The change may be harmless reflection drift or a real product edit. |
| `reflection_tolerant_lpips` | Weighted visual difference inside the vehicle mask. Reflection-like paint and glass areas receive lower weights; structural and product-critical areas stay important. | Main score for digitally inserted backgrounds. | A changed area is too large, too strong, or not explainable as a tolerated reflection/background effect. |
| `product_detail_lpips` | Visual difference focused on Mercedes product and brand details such as headlights, grille/front panel, emblems, wheels, DRL/light signatures, badges and profile-specific details. | Guard brand identity and model-specific visible features. | Product-signature details changed and the candidate should usually fail. |
| `structure_integrity_score` | Edge/contour consistency for protected structure such as silhouette, roofline, wheel arches, body lines, panel gaps and character lines. Lower is better. | Safety check so reflection tolerance cannot hide shape edits. | Shape, contour or body-line geometry changed; fail or warn strongly even if the pixels are on low-weight paint. |

## Mask and weight traceability

Make every score reproducible and transparent:

1. Start with a `vehicle_mask`; pixels outside it receive zero weight.
2. Add manual part masks where available, for example `headlights`, `emblem`, `front_grille_geometry`, `wheel_design`, `side_character_line_mask`, `paint_reflection_door_surface`, `amg_grille_pattern`, `maybach_grille_treatment` or `closed_front_panel`.
3. Build heuristic masks only where manual masks are missing. Heuristic masks are acceptable for a prototype, but they must be visible in diagnostic output.
4. Compose the final weight map by max priority per pixel: reflection/glass zones are low weight; structure, product-detail and brand-critical zones override them with high weights.
5. Save heatmaps and JSON reports so every final status can be traced to masks, weights, thresholds and reasons.

Manual masks are recommended for high-stakes evaluation, especially for thesis experiments, benchmark datasets and brand-governance review. Heuristic masks can accelerate prototyping, but do not treat them as a hidden ground truth.

## Expected behavior covered by tests

The regression tests create deterministic synthetic vehicle scenes and use a simple pixel-difference LPIPS stand-in so they run without downloading neural weights. They prove these cases:

1. Background-only changes outside the vehicle mask keep `standard_vehicle_lpips`, `reflection_tolerant_lpips` and `product_detail_lpips` at zero.
2. A soft reflection patch on broad paint increases strict vehicle difference more than the reflection-tolerant score, while structure remains acceptable.
3. A reflection mask crossing a side character line gives low weight to paint but preserves high weight on the protected line.
4. Moving the side character line fails via `structure_integrity_score`.
5. Headlight edits increase `product_detail_lpips` and fail.
6. Mercedes star/emblem edits increase `product_detail_lpips` and fail.
7. Grille/front-panel edits increase `product_detail_lpips` and fail.
8. Wheel design edits increase `product_detail_lpips` and fail.
9. Silhouette/roofline changes fail via `structure_integrity_score`.
10. AMG-specific detail changes are detected with the `amg` profile.
11. Maybach-specific detail changes are detected with the `maybach` profile.
12. EQ/electric front-panel detail changes are detected with the `eq_electric` profile.

Run the tests with:

```bash
pytest -q tests/test_mercedes_weighting.py
```

## CLI usage example

The repository CLI currently exposes the reflection-tolerant path through `image_metrics.py`. Use `--enable-reflection-tolerant-lpips`, write diagnostics, and save a JSON report:

```bash
python3 image_metrics.py \
  --ref reference/alt/cla.png \
  --gen generated/alt/claWald.png \
  --enable-reflection-tolerant-lpips \
  --reflection-diagnostic-dir reflection_tolerant_diagnostics \
  --reflection-json reports/cla_reflection_tolerant.json \
  --lpips-heatmap-dir lpips_heatmaps \
  --output-csv image_metrics_results.csv
```

For a production/thesis wrapper, keep these user-facing option names so runs are easy to audit:

```bash
mercedes-similarity \
  --reference reference.png \
  --candidate candidate.png \
  --vehicle-mask masks/vehicle.png \
  --model-profile eq_electric \
  --optional-mask-dir masks/parts \
  --output-dir reports/run_001 \
  --mode mercedes_reflection_tolerant \
  --save-heatmaps \
  --json-report reports/run_001/report.json
```

Use `--mode standard` when you want strict vehicle similarity only. Use `--mode mercedes_reflection_tolerant` when the main question is whether the same Mercedes product survives a background/reflection change.
