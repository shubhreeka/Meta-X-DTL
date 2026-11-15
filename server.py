"""Flask API: oral lesion classification with Grad-CAM, plus optional Gemini explanations.

POST /diagnose          multipart 'image' -> diagnosis, confidence, Grad-CAM overlay (base64 JPEG)
POST /get_explanation   {"diagnosis": ...} -> plain-language explanation (needs GEMINI_API_KEY)
POST /explain_term      {"term": ...}      -> one-line definition       (needs GEMINI_API_KEY)

Research prototype, not a medical device.
"""
import base64
import io
import json
import os

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, ImageOps

from src.config import IMG_SIZE, RESULTS
from src.enhance import enhance
from src.gradcam import build_full_model, overlay, predict_with_gradcam

app = Flask(__name__)
CORS(app)

def _load_local_env():
    """Read KEY=value lines from a .env file next to this script (the file is git-ignored)."""
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ.setdefault(key.strip(), value)


_load_local_env()


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# "gemini-flash-latest" always points at Google's current Flash model; the pinned model is a fallback
# (e.g. when the alias is briefly overloaded). Older pinned versions get retired over time.
GEMINI_MODELS = ["gemini-flash-latest", "gemini-3.8-flash"]


class GeminiError(Exception):
    """Error safe to show to users (never contains the API key)."""

ENH = json.loads((RESULTS / "enhancement.json").read_text())
MODEL = build_full_model()


def to_b64_jpeg(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


@app.route("/diagnose", methods=["POST"])
def diagnose():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"error": "No image file provided"}), 400
    img = ImageOps.exif_transpose(Image.open(request.files["image"].stream)).convert("RGB")
    rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.uint8)

    enhanced = enhance(rgb, ENH["alpha"], ENH["beta"], ENH["clahe_clip"])
    p_malignant, cam = predict_with_gradcam(MODEL, enhanced)
    malignant = p_malignant >= 0.5
    return jsonify({
        "diagnosis": "Malignant Lesion" if malignant else "Benign Lesion",
        "confidence": f"{(p_malignant if malignant else 1 - p_malignant) * 100:.2f}%",
        "explanation": "VGG16 transfer learning with metaheuristic-optimised classifier; "
                       "heatmap shows the regions that drove the prediction (Grad-CAM).",
        "heatmap_image": to_b64_jpeg(overlay(enhanced, cam)),
        "status": "success",
    })


def ask_gemini(system_prompt, user_query):
    if not GEMINI_API_KEY:
        raise GeminiError("Put GEMINI_API_KEY=<your key> in the .env file in this folder to enable explanations.")
    payload = {"contents": [{"parts": [{"text": user_query}]}],
               "systemInstruction": {"parts": [{"text": system_prompt}]}}
    last = None
    for model in GEMINI_MODELS:
        try:
            # key sent as a header, so it never appears in URLs, logs or error messages
            r = requests.post(GEMINI_URL.format(model=model), headers={"x-goog-api-key": GEMINI_API_KEY},
                              json=payload, timeout=30)
        except requests.Timeout:
            last = "busy"
            continue
        except requests.RequestException:
            last = "unreachable"
            continue
        if r.ok:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        if r.status_code in (429, 500, 503):
            last = "busy"
        elif r.status_code in (400, 401, 403):
            raise GeminiError("Explanation unavailable: the Gemini API key was rejected. Check GEMINI_API_KEY in .env.")
        else:
            last = f"HTTP {r.status_code}"
    messages = {"busy": "Gemini is busy right now. Please try again in a minute.",
                "unreachable": "could not connect to the Gemini API (check your internet connection)."}
    raise GeminiError("Explanation unavailable: " + messages.get(last, f"Gemini API returned {last}."))


@app.route("/get_explanation", methods=["POST"])
def get_explanation():
    diagnosis = (request.get_json() or {}).get("diagnosis")
    if not diagnosis:
        return jsonify({"error": "No diagnosis provided"}), 400
    try:
        text = ask_gemini("Act as a helpful medical assistant. In one short paragraph of plain language, explain the "
                          "given oral lesion diagnosis, its common characteristics and that the person should consult "
                          "a dentist or doctor. This is an AI screening result, not a diagnosis.",
                          f"Explain the diagnosis: {diagnosis}")
        return jsonify({"explanation": text})
    except GeminiError as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain_term", methods=["POST"])
def explain_term():
    term = (request.get_json() or {}).get("term")
    if not term:
        return jsonify({"error": "No term provided"}), 400
    try:
        return jsonify({"definition": ask_gemini("Act as a medical dictionary. Give a one-sentence definition.",
                                                 f"Define {term}")})
    except GeminiError as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=9001, debug=False)
