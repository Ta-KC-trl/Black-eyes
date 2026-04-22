"""
start.py — Launch the Black Eyes API + website with one command.
Run: python start.py
"""

import subprocess, sys, os, time, webbrowser
from pathlib import Path

BASE    = Path(__file__).resolve().parent
PYTHON  = sys.executable
DOCS    = BASE / "docs"
API_PORT  = 5000
WEB_PORT  = 8080

print("━" * 50)
print("  BLACK EYES — Starting services")
print("━" * 50)

# Start Flask API
api = subprocess.Popen(
    [PYTHON, str(BASE / "api.py")],
    cwd=str(BASE),
)
print(f"  ✓ API started     → http://localhost:{API_PORT}")

# Start static file server
web = subprocess.Popen(
    [PYTHON, "-m", "http.server", str(WEB_PORT), "--directory", str(DOCS)],
    cwd=str(BASE),
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
print(f"  ✓ Website started → http://localhost:{WEB_PORT}")

# Open browser after a short delay
time.sleep(2)
webbrowser.open(f"http://localhost:{WEB_PORT}")
print(f"\n  Browser opened. Press Ctrl+C to stop both servers.\n")

try:
    api.wait()
except KeyboardInterrupt:
    print("\n  Shutting down...")
    api.terminate()
    web.terminate()
    print("  Done.")
