ERGOVISION PROTOTYPE — WINDOWS
==============================

WHAT'S INCLUDED
- main.py: Python local server, webcam face/eye detection, and Windows display gamma warmth attempt.
- index.html: HTML + CSS in ONE file, with small inline JavaScript for buttons and alerts.
- requirements.txt: one optional external Python package, OpenCV, for webcam processing.

RUN IT
1. Install Python 3 from python.org if needed. During setup, tick "Add Python to PATH".
2. Extract this ZIP.
3. Open the extracted folder in VS Code.
4. Open Terminal > New Terminal.
5. Install webcam dependency:
       python -m pip install -r requirements.txt
   If "python" is not recognized, try "py -m pip install -r requirements.txt".
6. Start the app:
       python main.py
   Or: py main.py
7. It should open your browser. If not, visit:
       http://127.0.0.1:8000
8. Click Start session, then Start camera. Allow camera access in Windows if prompted.
9. Test alert shows a visible dashboard notification. Export JSON downloads the current session summary.

FEATURES IN THIS PROTOTYPE
- Live webcam preview and local face/eye detection using OpenCV Haar cascades.
- Approximate blink-event estimate. It is not a precise blink counter and can be affected by lighting, glasses, or camera angle.
- Session timer and in-page blink/posture reminders.
- Windows system warmth attempt using a gamma ramp. This is not physical brightness and some drivers block it. It may not work on every display.
- In-app alerts and a JSON session summary export.
- HTML and CSS are together in index.html.

IMPORTANT LIMITATIONS
- Posture reminders are periodic prompts; this version does not estimate body posture.
- Windows gamma adjustment is a best-effort display colour filter, not Windows Night Light. If it fails, use Windows Settings > System > Display > Night light.
- Camera frames are processed locally and not uploaded by this app.
- This prototype has been syntax-checked, but needs to be tested on your actual Windows PC and webcam.
- If you stop the program unexpectedly after warmth was applied, restart it and click the warmth toggle off if possible, or sign out/restart Windows to reset the display state.
