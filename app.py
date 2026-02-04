import os
import base64
import tempfile
from flask import Flask, request, jsonify
import numpy as np
import librosa

app = Flask(__name__)

API_KEY = "sk_test_123456789"
SUPPORTED_LANGUAGES = ["en", "hi"]


@app.route("/detect", methods=["POST"])
def voice_detection():
    api_key = request.headers.get("x-api-key")

    if api_key != API_KEY:
        return jsonify({
            "status": "error",
            "message": "Invalid API key"
        }), 401

    data = request.get_json()

    if not data:
        return jsonify({
            "status": "error",
            "message": "Invalid JSON"
        }), 400

    language = data.get("language")
    audio_format = data.get("audioFormat")
    audio_base64 = data.get("audioBase64")

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({
            "status": "error",
            "message": "Unsupported language"
        }), 400

    if audio_format != "mp3":
        return jsonify({
            "status": "error",
            "message": "Only mp3 supported"
        }), 400

    if not audio_base64:
        return jsonify({
            "status": "error",
            "message": "Audio missing"
        }), 400

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        return jsonify({
            "status": "error",
            "message": "Invalid base64 audio"
        }), 400

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_bytes)
            temp_path = f.name

        y, sr = librosa.load(temp_path, sr=None)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc_variance = np.var(mfccs)

        ai_score = (
            (spectral_centroid / 5000) +
            (zero_crossing_rate * 10) -
            (mfcc_variance / 100)
        )

        confidence = min(max(abs(ai_score), 0.5), 0.99)

        if ai_score > 1.0:
            classification = "AI_GENERATED"
            explanation = "Synthetic speech patterns detected"
        else:
            classification = "HUMAN"
            explanation = "Natural human speech detected"

        return jsonify({
            "status": "success",
            "language": language,
            "classification": classification,
            "confidenceScore": round(confidence, 2),
            "explanation": explanation
        })

    except Exception:
        return jsonify({
            "status": "error",
            "message": "Audio processing failed"
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
