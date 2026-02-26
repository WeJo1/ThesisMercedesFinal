import base64
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cgi

BASE_DIR = Path(__file__).resolve().parent
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class MetricsHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A003
        message = format % args
        print(f"[HTTP] {self.address_string()} {self.command} {self.path} -> {message}")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/compare":
            self.send_error(HTTPStatus.NOT_FOUND, "Route nicht gefunden")
            return

        try:
            print("[API] Starte Vergleich über /api/compare")
            payload = self.parse_form_data()
            result = self.run_image_metrics(payload)
            print("[API] Vergleich erfolgreich abgeschlossen")
            self.send_json(HTTPStatus.OK, result)
        except Exception as exc:  # noqa: BLE001
            print(f"[API] Vergleich fehlgeschlagen: {exc}")
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
            csv_path = tmp_path / "result.csv"
            norm_dir = tmp_path / "normalized"
            car_only_dir = tmp_path / "car_only"

            ref_path, ref_origin = self.store_upload_as_image(payload["ref_image"], tmp_path, "ref")
            gen_path, gen_origin = self.store_upload_as_image(payload["gen_image"], tmp_path, "gen")

            command = [
                sys.executable,
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
                        "--car-only-dir",
                        str(car_only_dir),
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
                "ref_upload": ref_origin,
                "gen_upload": gen_origin,
                "car_only_ref_preview": self.image_file_to_data_url(row.get("car_only_ref_path")),
                "car_only_gen_preview": self.image_file_to_data_url(row.get("car_only_gen_path")),
                "car_only_mask_preview": self.image_file_to_data_url(row.get("car_only_mask_path")),
            }

    def store_upload_as_image(self, file_field, tmp_path, prefix):
        original_name = Path(file_field.filename).name
        suffix = Path(original_name).suffix.lower()

        if suffix == ".zip":
            extracted_path = self.extract_image_from_zip(file_field, tmp_path, prefix)
            return extracted_path, f"{original_name} -> {extracted_path.name}"

        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(f"Dateityp nicht unterstützt: {original_name}")

        output_path = tmp_path / f"{prefix}_{Path(original_name).name}"
        with output_path.open("wb") as output_file:
            shutil.copyfileobj(file_field.file, output_file)
        return output_path, original_name

    def extract_image_from_zip(self, file_field, tmp_path, prefix):
        zip_bytes = file_field.file.read()
        if not zip_bytes:
            raise ValueError("ZIP-Datei ist leer")

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            candidates = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                inner_name = Path(info.filename)
                if inner_name.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                if inner_name.name.startswith("."):
                    continue
                candidates.append(info)

            if not candidates:
                raise ValueError("ZIP enthält kein unterstütztes Bildformat")

            candidates.sort(key=lambda item: item.filename.lower())
            chosen = candidates[0]
            image_data = archive.read(chosen)

        target_name = f"{prefix}_{Path(chosen.filename).name}"
        target_path = tmp_path / target_name
        with target_path.open("wb") as target_file:
            target_file.write(image_data)
        return target_path

    def image_file_to_data_url(self, file_path):
        if not file_path:
            return None

        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return None

        ext = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }.get(ext, "application/octet-stream")

        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{encoded}"

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
