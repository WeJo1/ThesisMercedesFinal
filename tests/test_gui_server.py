import sys
import types

sys.modules.setdefault("cgi", types.SimpleNamespace(FieldStorage=None))

from gui_server import MetricsHandler


def test_preview_payload_exposes_weighted_reflection_lpips_to_ui():
    handler = object.__new__(MetricsHandler)
    payload = handler.build_preview_payload(
        {
            "filename": "pair.png",
            "reflection_tolerant_lpips": "0.1234",
            "raw_lpips_distance": "0.2000",
            "raw_lpips_similarity_percent": "80.0000",
            "weighted_lpips_distance": "0.1234",
            "weighted_lpips_similarity_percent": "87.6600",
            "final_similarity_percent": "87.6600",
            "final_metric": "weighted_lpips_similarity_percent",
            "weighting_enabled": "True",
            "active_weight_profile": "generic_mercedes",
            "weighting_mode": "mercedes_reflection_tolerant_lpips",
            "product_detail_lpips": "0.0100",
            "structure_integrity_score": "0.0200",
            "final_similarity_status": "pass",
            "reflection_tolerant_json": "runs/pair/reflection.json",
        },
        include_previews=False,
    )

    assert payload["reflection_tolerant_enabled"] is True
    assert payload["metrics"]["weighted_lpips_distance"] == "0.1234"
    assert payload["metrics"]["final_metric"] == "weighted_lpips_similarity_percent"
    assert payload["metrics"]["final_similarity_percent"] == "87.6600"
    assert payload["metrics"]["final_similarity_percent"] != payload["metrics"]["raw_lpips_similarity_percent"]
    assert payload["weighting_enabled"] is True
    assert payload["active_weight_profile"] == "generic_mercedes"
    assert payload["reflection_tolerant_lpips"] == "0.1234"
    assert payload["product_detail_lpips"] == "0.0100"
    assert payload["structure_integrity_score"] == "0.0200"
    assert payload["final_similarity_status"] == "pass"
    assert payload["reflection_tolerant_json"] == "runs/pair/reflection.json"
