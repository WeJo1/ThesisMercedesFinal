import csv
import json
import os
import shutil
import subprocess
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cgi

BASE_DIR = Path(__file__).resolve().parent


class MetricsHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/compare":
            self.send_error(HTTPStatus.NOT_FOUND, "Route nicht gefunden")
            return

        try:
            payload = self.parse_form_data()
            result = self.run_image_metrics(payload)
            self.send_json(HTTPStatus.OK, result)
        except Exception as exc:  # noqa: BLE001
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def parse_form_data(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Erwarte multipart/form-data")

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )

        ref_image = form["ref_image"] if "ref_image" in form else None
        gen_image = form["gen_image"] if "gen_image" in form else None

        if ref_image is None or not getattr(ref_image, "filename", ""):
            raise ValueError("Referenzbild fehlt")
        if gen_image is None or not getattr(gen_image, "filename", ""):
            raise ValueError("Generated-Bild fehlt")

        return {
            "ref_image": ref_image,
            "gen_image": gen_image,
            "lpips_net": form.getfirst("lpips_net", "alex"),
            "enable_car_only": form.getfirst("enable_car_only", "false") == "true",
            "car_mode": form.getfirst("car_mode", "neutralize_crop"),
            "mask_source": form.getfirst("mask_source", "ref"),
        }

    def run_image_metrics(self, payload):
        with tempfile.TemporaryDirectory(prefix="metrics_gui_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            ref_path = tmp_path / Path(payload["ref_image"].filename).name
            gen_path = tmp_path / Path(payload["gen_image"].filename).name
            csv_path = tmp_path / "result.csv"
            norm_dir = tmp_path / "normalized"

            with ref_path.open("wb") as ref_file:
                shutil.copyfileobj(payload["ref_image"].file, ref_file)
            with gen_path.open("wb") as gen_file:
                shutil.copyfileobj(payload["gen_image"].file, gen_file)

            command = [
                "python",
                str(BASE_DIR / "image_metrics.py"),
                "--ref",
                str(ref_path),
                "--gen",
                str(gen_path),
                "--output-csv",
                str(csv_path),
                "--out",
                str(norm_dir),
                "--lpips-net",
                payload["lpips_net"],
            ]

            if payload["enable_car_only"]:
                command.extend(
                    [
                        "--enable-car-only",
                        "--car-mode",
                        payload["car_mode"],
                        "--mask-source",
                        payload["mask_source"],
                    ]
                )

            process = subprocess.run(command, cwd=BASE_DIR, capture_output=True, text=True)
            if process.returncode != 0:
                raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "image_metrics.py fehlgeschlagen")

            with csv_path.open("r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                row = next(reader, None)

            if row is None:
                raise RuntimeError("Keine Metriken im CSV gefunden")

            return {
                "filename": row.get("filename"),
                "lpips": row.get("lpips"),
                "lpips_similarity_percent": row.get("lpips_similarity_percent"),
                "ssim": row.get("ssim"),
                "psnr": row.get("psnr"),
                "delta_e_ciede2000": row.get("delta_e_ciede2000"),
                "lpips_car_only": row.get("lpips_car_only"),
                "mask_iou": row.get("mask_iou"),
                "mask_dice": row.get("mask_dice"),
            }

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(os.environ.get("PORT", "4173"))
    server = ThreadingHTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[INFO] GUI Server läuft auf http://0.0.0.0:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
