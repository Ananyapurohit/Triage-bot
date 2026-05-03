#!/usr/bin/env python3
"""
launch.py
─────────────────────────────────────────────────────────────────────────────
ED Triage XAI — One-Command Launcher

Starts:
  1. Backend  (backend_app_N200.py)  on port 5000
  2. Middleware (middleware.py)       on port 8080  ← opens in browser

Run:
    python launch.py

Then open:  http://localhost:8080
─────────────────────────────────────────────────────────────────────────────
"""

import sys, os, time, socket, subprocess, webbrowser

BACKEND_FILE    = "backend_app_N200.py"
MIDDLEWARE_FILE = "middleware.py"
BACKEND_PORT    = 5000
MIDDLEWARE_PORT = 8080

def port_open(port, timeout=1.0):
    try:
        with socket.create_connection(("localhost", port), timeout=timeout):
            return True
    except: return False

def wait_for_port(port, label, timeout=20):
    print(f"  ⏳  Waiting for {label} on :{port} ", end="", flush=True)
    for _ in range(timeout):
        if port_open(port):
            print(f" ✅")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(f" ❌ timed out")
    return False

print("\n" + "═"*58)
print("  ED Triage XAI — Full Stack Launcher")
print("═"*58)

# Check files exist
for f in [BACKEND_FILE, MIDDLEWARE_FILE, "frontend_connected.html"]:
    if not os.path.exists(f):
        print(f"\n  ❌  Missing file: {f}")
        print(f"  Put all files in the same folder and retry.\n")
        sys.exit(1)
    print(f"  ✅  Found: {f}")

print()

# Launch backend if not already up
if port_open(BACKEND_PORT):
    print(f"  ✅  Backend already running on :{BACKEND_PORT}")
    backend_proc = None
else:
    print(f"  🚀  Launching backend → python {BACKEND_FILE}")
    backend_proc = subprocess.Popen(
        [sys.executable, BACKEND_FILE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if not wait_for_port(BACKEND_PORT, "Backend"):
        print("  Backend failed to start. Check backend_app_N200.py.")
        sys.exit(1)

# Launch middleware (which also serves frontend)
print(f"\n  🚀  Launching middleware → python {MIDDLEWARE_FILE}")
mw_proc = subprocess.Popen(
    [sys.executable, MIDDLEWARE_FILE],
    stdout=sys.stdout, stderr=sys.stderr
)

if not wait_for_port(MIDDLEWARE_PORT, "Middleware"):
    print("  Middleware failed to start.")
    sys.exit(1)

url = f"http://localhost:{MIDDLEWARE_PORT}"
print(f"\n{'═'*58}")
print(f"  🌐  Dashboard  →  {url}")
print(f"  🔌  API        →  {url}/api/")
print(f"  📋  Status     →  {url}/status")
print(f"{'═'*58}")
print(f"\n  Opening browser…")

time.sleep(1)
webbrowser.open(url)

print("  Press Ctrl+C to stop all servers.\n")

try:
    mw_proc.wait()
except KeyboardInterrupt:
    print("\n  Shutting down…")
    mw_proc.terminate()
    if backend_proc:
        backend_proc.terminate()
    print("  All servers stopped. Goodbye.\n")
