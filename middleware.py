"""
middleware.py
─────────────────────────────────────────────────────────────────────────────
ED Triage XAI — API Gateway / Middleware

Responsibilities:
  1. Dependency check  — verifies all required Python packages are installed
  2. Backend health    — confirms backend_app.py is reachable at localhost:5000
  3. API exposure      — exposes a clean public-facing REST API at port 8080
  4. Request forwarding — proxies every call to the backend, adds audit headers
  5. Frontend serving  — serves frontend_dashboard.html as a static file at /
  6. CORS              — injects Access-Control headers so browser can reach it
  7. Logging           — prints every request with timestamp, method, path, status

Architecture:
   Browser  ──►  middleware :8080  ──►  backend_app.py :5000
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
# Verifies every required package before starting the server.
# If any is missing, prints install instructions and exits.
# ══════════════════════════════════════════════════════════════

REQUIRED_PACKAGES = {
    "flask":       "flask",
    "numpy":       "numpy",
    "pandas":      "pandas",
    "sklearn":     "scikit-learn",
    "requests":    "requests",
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
# STEP 2 — CHECK IF BACKEND IS ALREADY RUNNING
# Pings localhost:5000/api/health. If not running, auto-launches
# backend_app.py in a subprocess.
# ══════════════════════════════════════════════════════════════

BACKEND_HOST = "localhost"
BACKEND_PORT = 5000
BACKEND_URL  = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_FILE = "backend_app_N200.py"   # falls back to backend_app.py if not found

def is_port_open(host, port, timeout=1.5):
    """Check if TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

def check_backend_health():
    """Hit /api/health and confirm {status:'ok'}."""
    import urllib.request
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/api/health", timeout=3) as r:
            data = json.loads(r.read())
            return data.get("status") == "ok"
    except Exception:
        return False

_backend_proc = None  # keep reference so we can terminate on exit

def ensure_backend():
    global _backend_proc
    print("\n[2/4] Checking backend (Flask :5000)…\n")

    if is_port_open(BACKEND_HOST, BACKEND_PORT):
        if check_backend_health():
            print(f"  ✅  Backend already running at {BACKEND_URL}")
            return
        else:
            print(f"  ⚠️  Port 5000 open but /api/health not responding — proceeding anyway")
            return

    # Not running — try to launch it
    backend_file = BACKEND_FILE if os.path.exists(BACKEND_FILE) else "backend_app.py"
    if not os.path.exists(backend_file):
        print(f"  ❌  Backend file '{backend_file}' not found in current directory.")
        print(f"      Place backend_app_N200.py (or backend_app.py) in the same folder.")
        sys.exit(1)

    print(f"  ⚙️  Launching {backend_file} in background…")
    _backend_proc = subprocess.Popen(
        [sys.executable, backend_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait up to 15s for it to come up
    for i in range(15):
        time.sleep(1)
        if is_port_open(BACKEND_HOST, BACKEND_PORT):
            time.sleep(1)   # let Flask finish binding
            if check_backend_health():
                print(f"  ✅  Backend started (PID {_backend_proc.pid}) — healthy")
                return
            else:
                print(f"  ✅  Backend started (PID {_backend_proc.pid}) — port open")
                return
        print(f"  … waiting ({i+1}s)")

    print("  ❌  Backend did not start within 15 seconds.")
    print("      Try running it manually: python backend_app_N200.py")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# STEP 3 — CHECK IF FRONTEND FILE EXISTS
# ══════════════════════════════════════════════════════════════

FRONTEND_FILE = "frontend_dashboard.html"

def check_frontend():
    print("\n[3/4] Checking frontend file…\n")
    if not os.path.exists(FRONTEND_FILE):
        print(f"  ❌  '{FRONTEND_FILE}' not found in current directory.")
        print(f"      Place frontend_dashboard.html in the same folder as middleware.py.")
        sys.exit(1)
    size = os.path.getsize(FRONTEND_FILE)
    print(f"  ✅  {FRONTEND_FILE} found ({size:,} bytes)")

# ══════════════════════════════════════════════════════════════
# STEP 4 — BUILD THE MIDDLEWARE FLASK APP
# Public-facing API at :8080 — proxies to :5000
# Also serves the frontend HTML at GET /
# ══════════════════════════════════════════════════════════════

MIDDLEWARE_PORT = 8080

def build_app():
    """Construct and return the Flask middleware app."""

    # Import here (after dep check passed)
    from flask import Flask, request, jsonify, Response, send_file
    import urllib.request
    import urllib.error

    app = Flask(__name__)
    app.url_map.strict_slashes = False

    # ── CORS — allow browser calls from any origin ──
    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Request-ID"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["X-Served-By"]                  = "ED-Triage-XAI-Middleware"
        response.headers["X-Backend"]                    = BACKEND_URL
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
        colour = "\033[92m" if status < 300 else "\033[93m" if status < 500 else "\033[91m"
        reset  = "\033[0m"
        print(f"  {ts}  {colour}{status}{reset}  {method:<6} {path}  ({ms:.0f}ms)")

    # ── PROXY HELPER ──
    def proxy(path, method="GET", body=None):
        """Forward a request to the backend and return its response."""
        url  = f"{BACKEND_URL}{path}"
        t0   = time.perf_counter()
        try:
            if method == "POST":
                data = json.dumps(body).encode() if body else b""
                req  = urllib.request.Request(url, data=data,
                       headers={"Content-Type": "application/json"}, method="POST")
            else:
                req = urllib.request.Request(url, method="GET")

            with urllib.request.urlopen(req, timeout=10) as r:
                raw     = r.read()
                payload = json.loads(raw)
                ms      = (time.perf_counter() - t0) * 1000
                return payload, r.status, ms

        except urllib.error.HTTPError as e:
            raw  = e.read()
            ms   = (time.perf_counter() - t0) * 1000
            try:    payload = json.loads(raw)
            except: payload = {"error": str(e)}
            return payload, e.code, ms

        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return {"error": f"Backend unreachable: {e}"}, 502, ms

    # ══════════════════════════════════════════════════════════
    # PUBLIC API ROUTES (middleware → proxies to backend :5000)
    # ══════════════════════════════════════════════════════════

    # ── GET / — Serve the frontend HTML ──
    @app.route("/", methods=["GET"])
    def serve_frontend():
        t0 = time.perf_counter()
        resp = send_file(FRONTEND_FILE, mimetype="text/html")
        log("GET", "/", 200, (time.perf_counter()-t0)*1000)
        return resp

    # ── GET /status — Middleware + backend health check ──
    @app.route("/status", methods=["GET"])
    def status():
        t0 = time.perf_counter()
        backend_up  = is_port_open(BACKEND_HOST, BACKEND_PORT)
        backend_ok  = check_backend_health() if backend_up else False
        payload = {
            "middleware": {
                "status":  "running",
                "port":    MIDDLEWARE_PORT,
                "url":     f"http://localhost:{MIDDLEWARE_PORT}",
                "pid":     os.getpid(),
            },
            "backend": {
                "status":  "healthy" if backend_ok else ("port_open" if backend_up else "offline"),
                "url":     BACKEND_URL,
                "port":    BACKEND_PORT,
            },
            "frontend": {
                "file":    FRONTEND_FILE,
                "exists":  os.path.exists(FRONTEND_FILE),
                "size_kb": round(os.path.getsize(FRONTEND_FILE)/1024, 1) if os.path.exists(FRONTEND_FILE) else 0,
                "served_at": f"http://localhost:{MIDDLEWARE_PORT}/",
            },
            "dependencies_ok": True,
            "timestamp": datetime.now().isoformat(),
        }
        ms = (time.perf_counter()-t0)*1000
        log("GET", "/status", 200, ms)
        return jsonify(payload)

    # ── GET /api/health ──
    @app.route("/api/health", methods=["GET"])
    def health():
        data, status_code, ms = proxy("/api/health")
        log("GET", "/api/health", status_code, ms)
        return jsonify(data), status_code

    # ── GET /api/dataset/stats ──
    @app.route("/api/dataset/stats", methods=["GET"])
    def dataset_stats():
        data, code, ms = proxy("/api/dataset/stats")
        log("GET", "/api/dataset/stats", code, ms)
        return jsonify(data), code

    # ── GET /api/dataset/sample ──
    @app.route("/api/dataset/sample", methods=["GET"])
    def dataset_sample():
        data, code, ms = proxy("/api/dataset/sample")
        log("GET", "/api/dataset/sample", code, ms)
        return jsonify(data), code

    # ── GET /api/models/metrics ──
    @app.route("/api/models/metrics", methods=["GET"])
    def all_metrics():
        data, code, ms = proxy("/api/models/metrics")
        log("GET", "/api/models/metrics", code, ms)
        return jsonify(data), code

    # ── GET /api/models/<key>/metrics ──
    @app.route("/api/models/<model_key>/metrics", methods=["GET"])
    def model_metrics(model_key):
        path = f"/api/models/{model_key}/metrics"
        data, code, ms = proxy(path)
        log("GET", path, code, ms)
        return jsonify(data), code

    # ── GET /api/models/feature_importance ──
    @app.route("/api/models/feature_importance", methods=["GET"])
    def feature_importance():
        data, code, ms = proxy("/api/models/feature_importance")
        log("GET", "/api/models/feature_importance", code, ms)
        return jsonify(data), code

    # ── POST /api/predict ──
    @app.route("/api/predict", methods=["POST"])
    def predict():
        body = request.get_json(force=True, silent=True) or {}
        data, code, ms = proxy("/api/predict", method="POST", body=body)
        log("POST", "/api/predict", code, ms)
        return jsonify(data), code

    # ── POST /api/predict/compare ──
    @app.route("/api/predict/compare", methods=["POST"])
    def predict_compare():
        body = request.get_json(force=True, silent=True) or {}
        data, code, ms = proxy("/api/predict/compare", method="POST", body=body)
        log("POST", "/api/predict/compare", code, ms)
        return jsonify(data), code

    # ── 404 handler ──
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "error": "Route not found on middleware",
            "available_routes": [
                "GET  /",
                "GET  /status",
                "GET  /api/health",
                "GET  /api/dataset/stats",
                "GET  /api/dataset/sample",
                "GET  /api/models/metrics",
                "GET  /api/models/<key>/metrics",
                "GET  /api/models/feature_importance",
                "POST /api/predict",
                "POST /api/predict/compare",
            ]
        }), 404

    return app

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def main():
    # Run all pre-flight checks
    check_dependencies()
    ensure_backend()
    check_frontend()

    # Build app
    app = build_app()

    print(f"\n[4/4] Starting middleware on port {MIDDLEWARE_PORT}…\n")
    print("═"*60)
    print(f"  🌐  Dashboard    →  http://localhost:{MIDDLEWARE_PORT}/")
    print(f"  🔌  Public API   →  http://localhost:{MIDDLEWARE_PORT}/api/")
    print(f"  📋  Status       →  http://localhost:{MIDDLEWARE_PORT}/status")
    print(f"  ⚙️   Backend      →  http://localhost:{BACKEND_PORT}/")
    print("═"*60)
    print("\n  Request log:\n")

    # Graceful shutdown — kill backend subprocess on exit
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
