"""
ErgoVision prototype for Windows.
Uses only Python's standard library plus OpenCV for webcam face/eye detection.

Run:
    python main.py
Then open http://127.0.0.1:8000
Stop with Ctrl+C.
"""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import json
import threading
import time
import webbrowser
import sys

try:
    import cv2
except ImportError:
    cv2 = None

HOST = "127.0.0.1"
PORT = 8000
ROOT = Path(__file__).resolve().parent


class ErgoState:
    def __init__(self):
        self.lock = threading.Lock()
        self.camera = None
        self.camera_thread = None
        self.camera_running = False
        self.latest_jpeg = None
        self.face_detected = False
        self.eyes_detected = None
        self.blinks = 0
        self.last_eye_state = None
        self.no_eye_frames = 0
        self.last_error = ""
        self.warmth_enabled = False
        self.warmth_level = 25
        self.original_gamma = None
        self.gamma_supported = False
        self.last_frame_time = 0

    def status(self):
        with self.lock:
            return {
                "camera_running": self.camera_running,
                "face_detected": self.face_detected,
                "eyes_detected": self.eyes_detected,
                "blinks": self.blinks,
                "warmth_enabled": self.warmth_enabled,
                "warmth_level": self.warmth_level,
                "opencv_available": cv2 is not None,
                "error": self.last_error,
                "last_frame_time": self.last_frame_time,
            }


STATE = ErgoState()


def apply_windows_warmth(enabled, level=25):
    """Apply a subtle Windows display gamma ramp. Returns (success, message)."""
    if sys.platform != "win32":
        return False, "System warmth is supported only on Windows."
    import ctypes

    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32
    hdc = user32.GetDC(None)
    if not hdc:
        return False, "Windows did not provide a display handle."

    try:
        ramp_type = (ctypes.c_ushort * 256) * 3
        ramp = ramp_type()
        if enabled:
            warmth = max(0, min(100, int(level))) / 100.0
            red_gain = 1.0
            green_gain = 1.0 - 0.035 * warmth
            blue_gain = 1.0 - 0.19 * warmth
            for i in range(256):
                ramp[0][i] = int(min(65535, i * 257 * red_gain))
                ramp[1][i] = int(min(65535, i * 257 * green_gain))
                ramp[2][i] = int(min(65535, i * 257 * blue_gain))
        else:
            for i in range(256):
                value = i * 257
                ramp[0][i] = value
                ramp[1][i] = value
                ramp[2][i] = value

        ok = bool(gdi32.SetDeviceGammaRamp(hdc, ctypes.byref(ramp)))
        if ok:
            return True, (
                "Windows display warmth applied. This is a gamma colour filter, "
                "not physical brightness; some graphics drivers may ignore it."
                if enabled else "Display gamma reset to neutral."
            )
        return False, "This display driver rejected gamma adjustment. Use Windows Night Light instead."
    except Exception as exc:
        return False, f"Could not adjust display warmth: {exc}"
    finally:
        user32.ReleaseDC(None, hdc)


def open_working_camera():
    """Tries multiple camera indices and backends to find an available webcam."""
    if sys.platform == "win32":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    # Try device index 0, 1, 2 across standard backends
    for index in (0, 1, 2):
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                # Verify that frames can actually be captured
                ok, _ = cap.read()
                if ok:
                    return cap
                cap.release()
    return None


def camera_worker():
    if cv2 is None:
        with STATE.lock:
            STATE.last_error = "OpenCV is not installed. Run: python -m pip install opencv-python"
            STATE.camera_running = False
        return

    camera = open_working_camera()
    if camera is None:
        with STATE.lock:
            STATE.last_error = (
                "Camera not found or blocked. Check Windows Settings -> Privacy & Security -> "
                "Camera and ensure 'Let desktop apps access your camera' is turned ON."
            )
            STATE.camera_running = False
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    eye_cascade_path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"

    face_cascade = cv2.CascadeClassifier(str(face_cascade_path))
    eye_cascade = cv2.CascadeClassifier(str(eye_cascade_path))

    with STATE.lock:
        STATE.camera = camera
        STATE.last_error = ""

    try:
        while True:
            with STATE.lock:
                keep_running = STATE.camera_running
            if not keep_running:
                break

            ok, frame = camera.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(70, 70))
            face_found = len(faces) > 0
            eyes_found = False if face_found else None

            if face_found:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                roi_gray = gray[y:y + int(h * 0.62), x:x + w]
                roi_color = frame[y:y + int(h * 0.62), x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(12, 12))
                eyes_found = len(eyes) >= 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (190, 150, 190), 2)
                for ex, ey, ew, eh in eyes[:2]:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (90, 200, 130), 2)
                cv2.putText(frame, "Face detected", (x, max(22, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 130), 2)
            else:
                cv2.putText(frame, "No face detected", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 100, 240), 2)

            with STATE.lock:
                STATE.face_detected = bool(face_found)
                STATE.eyes_detected = eyes_found
                if face_found and eyes_found:
                    if STATE.last_eye_state is False and STATE.no_eye_frames >= 1:
                        STATE.blinks += 1
                    STATE.last_eye_state = True
                    STATE.no_eye_frames = 0
                elif face_found and not eyes_found:
                    STATE.no_eye_frames += 1
                    if STATE.no_eye_frames <= 4:
                        STATE.last_eye_state = False
                STATE.last_frame_time = time.time()

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                with STATE.lock:
                    STATE.latest_jpeg = encoded.tobytes()
            time.sleep(0.04)
    except Exception as exc:
        with STATE.lock:
            STATE.last_error = f"Camera processing error: {exc}"
    finally:
        camera.release()
        with STATE.lock:
            STATE.camera = None
            STATE.camera_running = False
            STATE.latest_jpeg = None
            STATE.face_detected = False
            STATE.eyes_detected = None


def start_camera():
    if cv2 is None:
        return False, "OpenCV is missing. In terminal run: python -m pip install opencv-python"
    with STATE.lock:
        if STATE.camera_running:
            return True, "Camera is already running."
        STATE.camera_running = True
        STATE.last_error = ""
        STATE.blinks = 0
        STATE.last_eye_state = None
        STATE.no_eye_frames = 0
        STATE.latest_jpeg = None
        STATE.camera_thread = threading.Thread(target=camera_worker, daemon=True)
        STATE.camera_thread.start()

    time.sleep(0.6)  # Give time to cycle through indices/backends
    with STATE.lock:
        if not STATE.camera_running and STATE.last_error:
            return False, STATE.last_error
    return True, "Webcam started. Face and eye detection run locally on this computer."


def stop_camera():
    with STATE.lock:
        STATE.camera_running = False
        camera_thread = STATE.camera_thread
    if camera_thread and camera_thread.is_alive():
        camera_thread.join(timeout=2.0)
    return True, "Webcam stopped."


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format_string, *args):
        pass

    def send_json(self, data, code=200):
        payload = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status":
            self.send_json(STATE.status())
        elif path == "/camera_feed":
            self.serve_camera_feed()
        elif path == "/":
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON request."}, 400)
            return

        if path == "/api/camera/start":
            ok, message = start_camera()
            self.send_json({"ok": ok, "message": message, **STATE.status()}, 200 if ok else 500)
        elif path == "/api/camera/stop":
            ok, message = stop_camera()
            self.send_json({"ok": ok, "message": message})
        elif path == "/api/warmth":
            enabled = bool(data.get("enabled", False))
            level = max(0, min(100, int(data.get("level", 25))))
            ok, message = apply_windows_warmth(enabled, level)
            with STATE.lock:
                STATE.warmth_enabled = enabled if ok else False
                STATE.warmth_level = level
            self.send_json({"ok": ok, "message": message, **STATE.status()}, 200 if ok else 400)
        else:
            self.send_json({"error": "Unknown API endpoint."}, 404)

    def serve_camera_feed(self):
        boundary = b"--frame"
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                with STATE.lock:
                    frame = STATE.latest_jpeg
                    active = STATE.camera_running
                if frame:
                    self.wfile.write(boundary + b"\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame + b"\r\n")
                    self.wfile.flush()
                elif not active:
                    break
                time.sleep(0.08)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    address = f"http://{HOST}:{PORT}"
    print("=" * 48)
    print("ErgoVision prototype is ready")
    print(f"Open: {address}")
    print("Press Ctrl+C to stop.")
    if cv2 is None:
        print("NOTE: Webcam needs OpenCV. Install with: python -m pip install opencv-python")
    try:
        webbrowser.open(address)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping ErgoVision...")
    finally:
        stop_camera()
        if STATE.warmth_enabled:
            apply_windows_warmth(False)
        server.server_close()
        print("Stopped.")


if __name__ == "__main__":
    main()
