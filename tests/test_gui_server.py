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
            "product_detail_lpips": "0.0100",
            "structure_integrity_score": "0.0200",
            "final_similarity_status": "pass",
            "reflection_tolerant_json": "runs/pair/reflection.json",
        },
        include_previews=False,
    )

    assert payload["reflection_tolerant_enabled"] is True
    assert payload["reflection_tolerant_lpips"] == "0.1234"
    assert payload["product_detail_lpips"] == "0.0100"
    assert payload["structure_integrity_score"] == "0.0200"
    assert payload["final_similarity_status"] == "pass"
    assert payload["reflection_tolerant_json"] == "runs/pair/reflection.json"
