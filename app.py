from __future__ import annotations

import os

from flask import jsonify, request

from database.quams_backend import create_app
from services.document_processor import process_file_bytes


app = create_app()


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200


@app.post("/upload")
def upload_file():
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        return jsonify(process_file_bytes(uploaded.filename or "", uploaded.read()))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("OCR processing failed")
        return jsonify({"error": f"Failed to process file: {exc}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
