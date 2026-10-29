import os
import time
import threading
import urllib.request
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------- Configuration ----------------
MODEL_NAME = "face_detection_yunet_2023mar.onnx"
MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    + MODEL_NAME
)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CALIBRATION_COUNT = 30
WARNING_AFTER_SECONDS = 120  # 2 minutes
DEMO_MODE = True  # presentation mode; blink readings are simulated

# ---------------- Model / camera setup ----------------
def load_detector():
    if not os.path.exists(MODEL_NAME):
        print(f"Downloading {MODEL_NAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)
    return cv2.FaceDetectorYN.create(
        model=MODEL_NAME,
        config="",
        input_size=(CAMERA_WIDTH, CAMERA_HEIGHT),
        score_threshold=0.55,
        nms_threshold=0.3,
        top_k=2,
    )

try:
    detector = load_detector()
except Exception as exc:
    detector = None
    print(f"Face detector could not be initialized: {exc}")

camera_lock = threading.Lock()
state_lock = threading.Lock()
current_camera_index = 0
camera = None


def open_camera(index=0):
    # CAP_DSHOW is Windows-specific; fall back to the default backend.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened() and os.name == "nt":
        cap.release()
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

camera = open_camera(current_camera_index)

# ---------------- Telemetry / state ----------------
session_start_time = time.time()
telemetry_data = {
    "posture_status": "Calibrating",
    "posture_score": 100,
    "eye_status": "Open",
    "blink_count": 0,
    "blinks_per_min": 0,
    "attention_status": "Waiting for face",
    "focus_score": 100,
    "session_uptime_seconds": 0,
    "ambient_brightness": 0,
    "low_light": False,
    "warning_after_seconds": WARNING_AFTER_SECONDS,
    "warning_due": False,
    "demo_mode": DEMO_MODE,
    "face_detected": False,
}

calibration_samples = []
baseline_face_y = None
baseline_face_scale = None
posture_score_smooth = 100.0
focus_score_smooth = 100.0
last_focus_status = "Focused"
blink_timestamps = deque()
blink_total_count = 0
last_demo_blink = time.time()
frame_counter = 0


def reset_state(reset_session=True):
    global session_start_time, calibration_samples, baseline_face_y, baseline_face_scale
    global posture_score_smooth, focus_score_smooth, last_focus_status
    global blink_timestamps, blink_total_count, last_demo_blink, frame_counter
    with state_lock:
        if reset_session:
            session_start_time = time.time()
        calibration_samples = []
        baseline_face_y = None
        baseline_face_scale = None
        posture_score_smooth = 100.0
        focus_score_smooth = 100.0
        last_focus_status = "Focused"
        blink_timestamps.clear()
        blink_total_count = 0
        last_demo_blink = time.time()
        frame_counter = 0
        telemetry_data.update({
            "posture_status": "Calibrating",
            "posture_score": 100,
            "eye_status": "Open",
            "blink_count": 0,
            "blinks_per_min": 0,
            "attention_status": "Waiting for face",
            "focus_score": 100,
            "session_uptime_seconds": 0,
            "warning_due": False,
            "face_detected": False,
        })


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def process_frame(frame):
    global baseline_face_y, baseline_face_scale, posture_score_smooth
    global focus_score_smooth, last_focus_status, last_demo_blink, frame_counter, blink_total_count

    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    now = time.time()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = int(np.mean(gray))

    with state_lock:
        elapsed = int(now - session_start_time)
        telemetry_data["session_uptime_seconds"] = elapsed
        telemetry_data["ambient_brightness"] = brightness
        telemetry_data["low_light"] = brightness < 55
        telemetry_data["warning_after_seconds"] = WARNING_AFTER_SECONDS
        telemetry_data["warning_due"] = elapsed >= WARNING_AFTER_SECONDS
        telemetry_data["demo_mode"] = DEMO_MODE
        frame_counter += 1

    faces = None
    if detector is not None:
        try:
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)
        except cv2.error:
            faces = None

    if faces is not None and len(faces) > 0:
        # Use the largest detected face to reduce switching between detections.
        face = max(faces, key=lambda f: float(f[2] * f[3]))
        fx, fy, fw, fh = [int(v) for v in face[:4]]
        fw, fh = max(1, fw), max(1, fh)
        face_center_y = (fy + fh / 2) / max(h, 1)
        face_scale = fh / max(h, 1)

        with state_lock:
            telemetry_data["face_detected"] = True

        # A short neutral calibration records the face's screen position/size.
        if len(calibration_samples) < CALIBRATION_COUNT:
            calibration_samples.append((face_center_y, face_scale))
            progress = int(100 * len(calibration_samples) / CALIBRATION_COUNT)
            with state_lock:
                telemetry_data["posture_status"] = f"Calibrating ({progress}%)"
                telemetry_data["posture_score"] = 100
            posture_ok = True
            if len(calibration_samples) == CALIBRATION_COUNT:
                baseline_face_y = float(np.median([x[0] for x in calibration_samples]))
                baseline_face_scale = float(np.median([x[1] for x in calibration_samples]))
        else:
            if baseline_face_y is None or baseline_face_scale is None:
                baseline_face_y = face_center_y
                baseline_face_scale = face_scale

            # Smooth the position/size signals. This is a presentation heuristic,
            # not a medically validated posture measurement.
            vertical_shift = abs(face_center_y - baseline_face_y)
            scale_shift = abs(face_scale - baseline_face_scale)
            raw_score = clamp(100 - vertical_shift * 170 - scale_shift * 90)
            posture_score_smooth = 0.82 * posture_score_smooth + 0.18 * raw_score
            posture_score = int(round(posture_score_smooth))
            posture_ok = posture_score >= 65
            with state_lock:
                telemetry_data["posture_score"] = posture_score
                telemetry_data["posture_status"] = "Good" if posture_ok else "Adjust Position"

        box_color = (40, 200, 80) if posture_ok else (40, 40, 230)
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), box_color, 2)
        r_eye = (int(face[4]), int(face[5]))
        l_eye = (int(face[6]), int(face[7]))
        nose = (int(face[8]), int(face[9]))
        cv2.circle(frame, r_eye, 3, (0, 220, 255), -1)
        cv2.circle(frame, l_eye, 3, (0, 220, 255), -1)

        # Focus estimate: nose position relative to eye span, with hysteresis
        # and smoothing to prevent rapid status flicker.
        eye_span = max(abs(l_eye[0] - r_eye[0]), 1)
        nose_ratio = abs(nose[0] - r_eye[0]) / eye_span
        looking_away = nose_ratio < 0.22 or nose_ratio > 0.78
        if looking_away:
            raw_focus = max(35, focus_score_smooth - 3)
            if last_focus_status == "Looking Away":
                focus_score_smooth = 0.80 * focus_score_smooth + 0.20 * raw_focus
            else:
                focus_score_smooth = 0.92 * focus_score_smooth + 0.08 * raw_focus
            if focus_score_smooth < 72:
                last_focus_status = "Looking Away"
        else:
            focus_score_smooth = 0.88 * focus_score_smooth + 0.12 * 100
            if focus_score_smooth > 82:
                last_focus_status = "Focused"

        with state_lock:
            telemetry_data["focus_score"] = int(round(clamp(focus_score_smooth)))
            telemetry_data["attention_status"] = last_focus_status

        # YuNet doesn't provide eyelid landmarks. For presentation only, create
        # a visibly stable demo counter instead of pretending contrast is a blink.
        if DEMO_MODE:
            if now - last_demo_blink >= 4.5:
                blink_timestamps.append(now)
                blink_total_count += 1
                last_demo_blink = now
            with state_lock:
                telemetry_data["eye_status"] = "Demo estimate"
                telemetry_data["blink_count"] = blink_total_count
        else:
            with state_lock:
                telemetry_data["eye_status"] = "Not measured"

        cv2.putText(frame, f"Posture: {telemetry_data['posture_status']}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, box_color, 2)
        cv2.putText(frame, f"Focus: {telemetry_data['attention_status']} ({telemetry_data['focus_score']}%)", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 230, 40), 2)
        cv2.putText(frame, f"Blinks: {telemetry_data['blink_count']} ({telemetry_data['eye_status']})", (15, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)
    else:
        with state_lock:
            telemetry_data["face_detected"] = False
            telemetry_data["posture_status"] = "No Face Detected"
            telemetry_data["attention_status"] = "No Face Detected"
            telemetry_data["eye_status"] = "Unknown"
        # Keep the last scores rather than making the meters jump on one missed frame.
        cv2.putText(frame, "No face detected - move into view", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 165, 255), 2)

    # Prune blink timestamps to a rolling one-minute window.
    while blink_timestamps and blink_timestamps[0] < now - 60:
        blink_timestamps.popleft()
    with state_lock:
        telemetry_data["blinks_per_min"] = len(blink_timestamps) if DEMO_MODE else 0
        if telemetry_data["warning_due"]:
            cv2.putText(frame, "2-MINUTE CHECK-IN", (15, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    return frame


def generate_video_stream():
    global camera
    while True:
        with camera_lock:
            cap = camera
            if cap is None or not cap.isOpened():
                frame = None
            else:
                ok, frame = cap.read()
                if not ok:
                    frame = None
        if frame is None:
            # Return a clear placeholder frame instead of hanging the browser.
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera unavailable - check permissions/device", (25, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            time.sleep(0.1)
        else:
            frame = cv2.flip(frame, 1)
            frame = process_frame(frame)
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/")
def home():
    return jsonify({"app": "Focus/Posture Prototype", "status": "running", "demo_mode": DEMO_MODE})


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/telemetry", methods=["GET"])
def get_telemetry():
    with state_lock:
        return jsonify(dict(telemetry_data))


@app.route("/api/camera", methods=["POST"])
def switch_camera():
    global camera, current_camera_index
    data = request.get_json(silent=True) or {}
    try:
        cam_index = int(data.get("index", 0))
        if cam_index < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"status": "error", "message": "Camera index must be a non-negative integer."}), 400

    new_camera = open_camera(cam_index)
    if not new_camera.isOpened():
        new_camera.release()
        return jsonify({"status": "error", "message": f"Could not open camera index {cam_index}."}), 400
    with camera_lock:
        old_camera = camera
        camera = new_camera
        current_camera_index = cam_index
        if old_camera is not None:
            old_camera.release()
    return jsonify({"status": f"Switched to camera index {cam_index}"})


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    global calibration_samples, baseline_face_y, baseline_face_scale, posture_score_smooth
    with state_lock:
        calibration_samples = []
        baseline_face_y = None
        baseline_face_scale = None
        posture_score_smooth = 100.0
        telemetry_data["posture_status"] = "Calibrating"
        telemetry_data["posture_score"] = 100
    return jsonify({"status": "Recalibrating baseline...", "calibration_frames": CALIBRATION_COUNT})


@app.route("/api/reset", methods=["POST"])
def reset_session_route():
    reset_state(reset_session=True)
    return jsonify({"status": "Reset complete"})


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({"warning_after_seconds": WARNING_AFTER_SECONDS, "demo_mode": DEMO_MODE,
                    "calibration_frames": CALIBRATION_COUNT})


if __name__ == "__main__":
    print(f"Presentation mode: {DEMO_MODE}; warning after {WARNING_AFTER_SECONDS} seconds.")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
