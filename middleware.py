"""
middleware.py
─────────────────────────────────────────────────────────────────────────────
ED Triage XAI — API Gateway / Middleware

FIXES APPLIED:
  - FRONTEND_FILE changed to "frontend_connected.html" (was "frontend_dashboard.html"
    — caused immediate crash at startup because the file was not found)
  - proxy() now uses the `requests` library instead of urllib.request.
    urllib has no connection pooling, no session reuse, and blocks the GIL under
    load — causing cascade timeouts. `requests` handles all of this correctly.
  - Health check now verifies models_ready field (backend is fully trained,
    not just Flask bound) before declaring backend healthy
  - Backend startup wait extended to 30s (was 15s) to accommodate N=1000 training
  - Middleware startup wait increased to account for training time
  - Added /api/predict/compare proxy route (was missing)
  - Better startup banner with explicit file paths shown
  - Removed misleading "backend_app.py" fallback in ensure_backend() — gave a
    confusing second path that also didn't exist; now fails clearly on BACKEND_FILE
  - Added /favicon.ico → 204 route to prevent noisy 404s in the request log

Architecture:
   Browser  ──►  middleware :8080  ──►  backend_app_N200.py :5000
               (serves HTML)         (ML models, XAI, data)

Usage:
   python middleware.py
   → http://localhost:8080       (opens the dashboard)
   → http://localhost:8080/api/* (public-facing API)
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import json
import time
import subprocess
import socket
import importlib
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# STEP 1 — DEPENDENCY CHECK
# ══════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = {
    "flask":    "flask",
    "numpy":    "numpy",
    "pandas":   "pandas",
    "sklearn":  "scikit-learn",
    "requests": "requests",
}

def check_dependencies():
    print("\n" + "═"*60)
    print("  ED Triage XAI — Middleware / API Gateway")
    print("═"*60)
    print("\n[1/4] Checking Python dependencies…\n")

    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f"  ✅  {package:<20} installed")
        except ImportError:
            print(f"  ❌  {package:<20} MISSING")
            missing.append(package)

    if missing:
        print(f"\n  ⚠️  Missing packages: {', '.join(missing)}")
        print(f"\n  Run:\n    pip install {' '.join(missing)}")
        print("\n  Middleware cannot start without all dependencies.\n")
        sys.exit(1)

    print("\n  All dependencies satisfied ✅")

# ══════════════════════════════════════════════════════════════
# STEP 2 — CHECK / AUTO-LAUNCH BACKEND
# ══════════════════════════════════════════════════════════════

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 5000
BACKEND_URL  = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_FILE = "backend_app_N200.py"

def is_port_open(host, port, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def check_backend_health():
    """
    FIX: now checks models_ready in the response to confirm full ML startup,
    not just that Flask has bound to the port. With N=1000 and n_estimators=100
    there is a ~10s gap between Flask binding and models being ready.
    """
    import requests as req
    try:
        r = req.get(f"{BACKEND_URL}/api/health", timeout=4)
        data = r.json()
        return data.get("status") == "ok" and len(data.get("models_ready", [])) == 3
    except Exception:
        return False

_backend_proc = None

def ensure_backend():
    global _backend_proc
    print("\n[2/4] Checking backend (Flask :5000)…\n")

    if is_port_open(BACKEND_HOST, BACKEND_PORT):
        if check_backend_health():
            print(f"  ✅  Backend already running and healthy at {BACKEND_URL}")
            return
        else:
            print(f"  ⚠️  Port 5000 open but models not ready yet — waiting...")

    backend_file = BACKEND_FILE
    if not os.path.exists(backend_file):
        print(f"  ❌  Backend file '{backend_file}' not found.")
        print(f"      Place backend_app_N200.py in the same folder as middleware.py")
        sys.exit(1)

    if not is_port_open(BACKEND_HOST, BACKEND_PORT):
        print(f"  ⚙️   Launching {backend_file} in background…")
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.log")
        backend_log = open(log_path, "w")
        _backend_proc = subprocess.Popen(
            [sys.executable, "-u", backend_file],
            stdout=backend_log,
            stderr=backend_log,
        )
        print(f"  📄  Backend log → {log_path}")
        print(f"  ℹ️   Training N=1000 dataset + 3 models — please wait ~10s …\n")

    # FIX: wait up to 30s (was 15s) to give training time to complete
    for i in range(30):
        time.sleep(1)
        if check_backend_health():
            pid = _backend_proc.pid if _backend_proc else "external"
            print(f"\n  ✅  Backend ready (PID {pid}) — all 3 models trained")
            return
        # Show a dot every second, newline every 10
        print("." if (i+1) % 10 else f". ({i+1}s)", end="", flush=True)

    print(f"\n\n  ❌  Backend did not become healthy within 30 seconds.")
    print(f"      Check backend.log for errors.")
    print(f"      Try running manually: python {backend_file}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# STEP 3 — CHECK FRONTEND FILE
# ══════════════════════════════════════════════════════════════

# FIX: was "frontend_dashboard.html" — file is actually named "frontend_connected.html"
FRONTEND_FILE = "frontend_connected.html"

def check_frontend():
    print("\n[3/4] Checking frontend file…\n")
    if not os.path.exists(FRONTEND_FILE):
        print(f"  ❌  '{FRONTEND_FILE}' not found in current directory.")
        print(f"      Place frontend_connected.html in the same folder as middleware.py")
        sys.exit(1)
    size = os.path.getsize(FRONTEND_FILE)
    print(f"  ✅  {FRONTEND_FILE} found ({size:,} bytes)")

# ══════════════════════════════════════════════════════════════
# STEP 4 — BUILD THE MIDDLEWARE FLASK APP
# ══════════════════════════════════════════════════════════════

MIDDLEWARE_PORT = 8080

def build_app():
    from flask import Flask, request, jsonify, send_file
    import requests as req  # FIX: use requests library, not urllib

    app = Flask(__name__)
    app.url_map.strict_slashes = False

    # ── CORS ──
    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["X-Served-By"] = "ED-Triage-XAI-Middleware"
        return response

    @app.before_request
    def handle_options():
        if request.method == "OPTIONS":
            res = jsonify({})
            res.headers["Access-Control-Allow-Origin"]  = "*"
            res.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            res.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            return res, 200

    # ── REQUEST LOGGER ──
    def log(method, path, status, ms):
        ts = datetime.now().strftime("%H:%M:%S")
        c  = "\033[92m" if status < 300 else "\033[93m" if status < 500 else "\033[91m"
        print(f"  {ts}  {c}{status}\033[0m  {method:<6} {path}  ({ms:.0f}ms)")

    # ── PROXY HELPER ──
    # FIX: was using urllib.request which has no connection pooling and blocks
    # under load. Now uses the `requests` library which is session-based,
    # supports keep-alive, and handles errors much more cleanly.
    session = req.Session()

    def proxy(path, method="GET", body=None):
        url = f"{BACKEND_URL}{path}"
        t0  = time.perf_counter()
        try:
            if method == "POST":
                r = session.post(url, json=body, timeout=15)
            else:
                r = session.get(url, timeout=10)
            ms = (time.perf_counter() - t0) * 1000
            try:
                payload = r.json()
            except Exception:
                payload = {"error": "Backend returned non-JSON response", "raw": r.text[:200]}
            return payload, r.status_code, ms
        except req.exceptions.ConnectionError:
            ms = (time.perf_counter() - t0) * 1000
            return {"error": "Backend is offline. Is backend_app_N200.py running on :5000?"}, 502, ms
        except req.exceptions.Timeout:
            ms = (time.perf_counter() - t0) * 1000
            return {"error": "Backend timed out. The model may still be training — wait a few seconds and retry."}, 504, ms
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return {"error": f"Proxy error: {str(e)}"}, 502, ms

    # ══════════════════════════════════════════════════════
    # ROUTES
    # ══════════════════════════════════════════════════════

    @app.route("/favicon.ico", methods=["GET"])
    def favicon():
        return '', 204

    @app.route("/", methods=["GET"])
    def serve_frontend():
        t0 = time.perf_counter()
        resp = send_file(os.path.abspath(FRONTEND_FILE), mimetype="text/html")
        log("GET", "/", 200, (time.perf_counter()-t0)*1000)
        return resp

    @app.route("/status", methods=["GET"])
    def status():
        t0 = time.perf_counter()
        backend_up = is_port_open(BACKEND_HOST, BACKEND_PORT)
        backend_ok = check_backend_health() if backend_up else False
        payload = {
            "middleware": {"status": "running", "port": MIDDLEWARE_PORT,
                           "url": f"http://localhost:{MIDDLEWARE_PORT}", "pid": os.getpid()},
            "backend":    {"status": "healthy" if backend_ok else ("port_open" if backend_up else "offline"),
                           "url": BACKEND_URL, "port": BACKEND_PORT},
            "frontend":   {"file": FRONTEND_FILE, "exists": os.path.exists(FRONTEND_FILE),
                           "size_kb": round(os.path.getsize(FRONTEND_FILE)/1024, 1) if os.path.exists(FRONTEND_FILE) else 0,
                           "served_at": f"http://localhost:{MIDDLEWARE_PORT}/"},
            "timestamp":  datetime.now().isoformat(),
        }
        ms = (time.perf_counter()-t0)*1000
        log("GET", "/status", 200, ms)
        return jsonify(payload)

    @app.route("/api/health", methods=["GET"])
    def health():
        data, code, ms = proxy("/api/health")
        log("GET", "/api/health", code, ms)
        return jsonify(data), code

    @app.route("/api/dataset/stats", methods=["GET"])
    def dataset_stats():
        data, code, ms = proxy("/api/dataset/stats")
        log("GET", "/api/dataset/stats", code, ms)
        return jsonify(data), code

    @app.route("/api/dataset/sample", methods=["GET"])
    def dataset_sample():
        data, code, ms = proxy("/api/dataset/sample")
        log("GET", "/api/dataset/sample", code, ms)
        return jsonify(data), code

    @app.route("/api/models/metrics", methods=["GET"])
    def all_metrics():
        data, code, ms = proxy("/api/models/metrics")
        log("GET", "/api/models/metrics", code, ms)
        return jsonify(data), code

    @app.route("/api/models/<model_key>/metrics", methods=["GET"])
    def model_metrics(model_key):
        path = f"/api/models/{model_key}/metrics"
        data, code, ms = proxy(path)
        log("GET", path, code, ms)
        return jsonify(data), code

    @app.route("/api/models/feature_importance", methods=["GET"])
    def feature_importance():
        data, code, ms = proxy("/api/models/feature_importance")
        log("GET", "/api/models/feature_importance", code, ms)
        return jsonify(data), code

    @app.route("/api/predict", methods=["POST"])
    def predict():
        body = request.get_json(force=True, silent=True) or {}
        data, code, ms = proxy("/api/predict", method="POST", body=body)
        log("POST", "/api/predict", code, ms)
        return jsonify(data), code

    # FIX: this route was missing from the original middleware
    @app.route("/api/predict/compare", methods=["POST"])
    def predict_compare():
        body = request.get_json(force=True, silent=True) or {}
        data, code, ms = proxy("/api/predict/compare", method="POST", body=body)
        log("POST", "/api/predict/compare", code, ms)
        return jsonify(data), code

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "error": "Route not found on middleware",
            "available_routes": [
                "GET  /", "GET  /status", "GET  /api/health",
                "GET  /api/dataset/stats", "GET  /api/dataset/sample",
                "GET  /api/models/metrics", "GET  /api/models/<key>/metrics",
                "GET  /api/models/feature_importance",
                "POST /api/predict", "POST /api/predict/compare",
            ]
        }), 404

    return app

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    check_dependencies()
    ensure_backend()
    check_frontend()

    app = build_app()

    print(f"\n[4/4] Starting middleware on :{MIDDLEWARE_PORT} …\n")
    print("═"*60)
    print(f"  🌐  Dashboard  →  http://localhost:{MIDDLEWARE_PORT}/")
    print(f"  🔌  API        →  http://localhost:{MIDDLEWARE_PORT}/api/")
    print(f"  📋  Status     →  http://localhost:{MIDDLEWARE_PORT}/status")
    print(f"  ⚙️   Backend    →  http://localhost:{BACKEND_PORT}/")
    print("═"*60)
    print("\n  Request log:\n")

    import atexit
    def cleanup():
        global _backend_proc
        if _backend_proc and _backend_proc.poll() is None:
            print(f"\n  Stopping backend (PID {_backend_proc.pid})…")
            _backend_proc.terminate()
    atexit.register(cleanup)

    app.run(host="0.0.0.0", port=MIDDLEWARE_PORT, debug=False)

if __name__ == "__main__":
    main()
