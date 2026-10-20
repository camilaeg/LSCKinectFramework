from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import uuid

import cv2
import mediapipe as mp
import numpy as np


MAX_IMAGE_BYTES = 2 * 1024 * 1024
DURATION_SECONDS = 3.0

LABELS = {
    "hola",
    "gracias",
    "si",
    "no",
    "adios",
    "sin_sena",
}

DATASET = Path(__file__).resolve().parent / "dataset"


class Recorder:
    def __init__(self):
        self.current = None
        self.message = "Selecciona una seña para grabar."

    def start(self, label):
        self.finish_if_due()

        if self.current is not None:
            return False

        self.current = {
            "label": label,
            "started": time.monotonic(),
            "created_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "frames": [],
        }

        self.message = "Grabando..."
        return True

    def add_frame(self, hands, width, height):
        self.finish_if_due()

        if self.current is None:
            return

        elapsed = (
            time.monotonic() - self.current["started"]
        )

        self.current["frames"].append({
            "t": round(elapsed, 4),
            "width": width,
            "height": height,
            "hands": hands,
        })

    def finish_if_due(self):
        if self.current is None:
            return

        elapsed = (
            time.monotonic() - self.current["started"]
        )

        if elapsed < DURATION_SECONDS:
            return

        recording = self.current
        self.current = None

        frames = recording["frames"]
        label = recording["label"]

        # Rechazar grabaciones demasiado cortas.
        if len(frames) < 10:
            self.message = (
                "Ejemplo descartado: llegaron menos de "
                "10 imágenes. Comprueba que la aplicación "
                "esté mostrando la cámara y vuelve a grabar."
            )
            return

        frames_with_hands = sum(
            bool(frame["hands"]) for frame in frames
        )

        # Las señas necesitan al menos alguna detección.
        # Sin seña también puede incluir ausencia de manos.
        if label != "sin_sena" and frames_with_hands < 5:
            self.message = (
                "Ejemplo descartado: las manos se detectaron "
                "en menos de 5 imágenes. Vuelve a grabar."
            )
            return

        folder = DATASET / label
        folder.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S"
        )

        filename = (
            f"{stamp}_{uuid.uuid4().hex[:10]}.json"
        )

        target = folder / filename
        temporary = target.with_suffix(".tmp")

        document = {
            "schema_version": 1,
            "label": label,
            "created_utc": recording["created_utc"],
            "duration_seconds": DURATION_SECONDS,
            "source": "kinect_v1_color",
            "coordinate_system": {
                "x": "normalized image coordinate",
                "y": "normalized image coordinate",
                "z": (
                    "MediaPipe relative depth; "
                    "not meters"
                ),
            },
            "frame_count": len(frames),
            "frames_with_hands": frames_with_hands,
            "frames": frames,
        }

        temporary.write_text(
            json.dumps(document, ensure_ascii=False),
            encoding="utf-8",
        )

        temporary.replace(target)

        count = len(list(folder.glob("*.json")))

        self.message = (
            f"Guardado: {label} | "
            f"Ejemplos de esta categoría: {count} | "
            f"Imágenes con manos: "
            f"{frames_with_hands}/{len(frames)}"
        )

    def status(self):
        self.finish_if_due()

        if self.current is not None:
            frames = len(self.current["frames"])
            return True, f"Grabando... imágenes recibidas: {frames}"

        return False, self.message


recorder = Recorder()


class DetectorHandler(BaseHTTPRequestHandler):
    detector = None

    def send_data(
        self,
        status,
        body,
        content_type,
        headers=None,
    ):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))

        for name, value in (headers or {}).items():
            self.send_header(name, value)

        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({
                "status": "ok",
                "mediapipe": mp.__version__,
            }).encode("utf-8")

            self.send_data(
                200,
                body,
                "application/json",
            )
            return

        if self.path == "/record/status":
            active, message = recorder.status()

            self.send_data(
                200,
                message.encode("utf-8"),
                "text/plain; charset=utf-8",
                {"X-Recording": "1" if active else "0"},
            )
            return

        self.send_error(404)

    def do_POST(self):
        if self.path not in {
            "/detect",
            "/record/start",
        }:
            self.send_error(404)
            return

        try:
            size = int(
                self.headers.get("Content-Length", "0")
            )
        except ValueError:
            self.send_error(400, "Longitud invalida")
            return

        limit = (
            100
            if self.path == "/record/start"
            else MAX_IMAGE_BYTES
        )

        if not 0 < size <= limit:
            self.send_error(413)
            return

        data = self.rfile.read(size)

        if self.path == "/record/start":
            try:
                label = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                self.send_error(400)
                return

            if label not in LABELS:
                self.send_error(400, "Categoria invalida")
                return

            if not recorder.start(label):
                self.send_error(
                    409,
                    "Ya existe una grabacion activa",
                )
                return

            self.send_data(
                200,
                b"Grabacion iniciada",
                "text/plain; charset=utf-8",
            )
            return

        image = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            self.send_error(400, "Imagen invalida")
            return

        result = self.detector.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )

        landmarks = result.multi_hand_landmarks or []
        hands = []

        # No asignamos izquierda/derecha:
        # el orden de detección puede cambiar entre imágenes.
        for hand in landmarks:
            hands.append([
                {
                    "x": float(point.x),
                    "y": float(point.y),
                    "z": float(point.z),
                }
                for point in hand.landmark
            ])

            mp.solutions.drawing_utils.draw_landmarks(
                image,
                hand,
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        height, width = image.shape[:2]

        recorder.add_frame(hands, width, height)

        ok, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )

        if not ok:
            self.send_error(500)
            return

        self.send_data(
            200,
            encoded.tobytes(),
            "image/jpeg",
            {"X-Hands-Count": str(len(hands))},
        )

    def log_message(self, format, *args):
        pass


def main():
    DATASET.mkdir(parents=True, exist_ok=True)

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as detector:
        DetectorHandler.detector = detector

        server = HTTPServer(
            ("127.0.0.1", 8765),
            DetectorHandler,
        )

        print(
            "Detector listo en 127.0.0.1:8765",
            flush=True,
        )
        print(
            f"Los ejemplos se guardarán en: {DATASET}",
            flush=True,
        )

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nCerrando detector...")
        finally:
            server.server_close()


if __name__ == "__main__":
    main()