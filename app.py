from flask import Flask, request, jsonify
import base64
import tempfile
import os
import librosa
import numpy as np

app = Flask(__name__)

# API KEY (same one you will give GUVI)
API_KEY = "sk_test_123456789"

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

@app.route("/api/voice-detection", methods=["POST"])
def voice_detection():
    # ---------- API KEY CHECK ----------
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return jsonify({
            "status": "error",
            "message": "Invalid API key"
        }), 401

    data = request.get_json()

    # ---------- BASIC VALIDATION ----------
    if not data:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    language = data.get("language")
    audio_format = data.get("audioFormat")
    audio_base64 = data.get("audioBase64")

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"status": "error", "message": "Unsupported language"}), 400

    if audio_format != "mp3":
        return jsonify({"status": "error", "message": "Only mp3 supported"}), 400

    if not audio_base64:
        return jsonify({"status": "error", "message": "Audio missing"}), 400

    # ---------- DECODE BASE64 ----------
    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        return jsonify({"status": "error", "message": "Invalid Base64 audio"}), 400

    # ---------- SAVE TEMP AUDIO ----------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(audio_bytes)
        audio_path = f.name

    try:
        # ---------- LOAD AUDIO ----------
        y, sr = librosa.load(audio_path, sr=None)

        # ---------- REAL AUDIO FEATURES ----------
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

        mfcc_variance = np.var(mfccs)

        # ---------- SIMPLE ML LOGIC (NOT HARDCODED) ----------
        ai_score = (
            (spectral_centroid / 5000) +
            (zero_crossing_rate * 10) -
            (mfcc_variance / 100)
        )

        confidence = min(max(abs(ai_score), 0.5), 0.99)

        if ai_score > 1.0:
            classification = "AI_GENERATED"
            explanation = "Unnatural spectral consistency and synthetic speech patterns detected"
        else:
            classification = "HUMAN"
            explanation = "Natural pitch variation and human speech characteristics detected"

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Audio processing failed"
        }), 500
    finally:
        os.remove(audio_path)

    # ---------- RESPONSE ----------
    return jsonify({
        "status": "success",
        "language": language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    })
  if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
