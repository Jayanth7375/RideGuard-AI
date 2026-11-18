"""
RideGuard AI - Standalone Web Application Launcher
Launches the FastAPI backend and serves the enterprise single-page web app.
"""

import sys
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 65, flush=True)
    print("🛡️  RIDEGUARD AI — STANDALONE ENTERPRISE WEB APPLICATION", flush=True)
    print("=" * 65, flush=True)
    print("🚀 Initializing FastAPI Backend Engine & Serving Web Dashboard...", flush=True)
    print("🌐 Web Application URL  : http://localhost:8000", flush=True)
    print("📚 Interactive API Docs : http://localhost:8000/docs", flush=True)
    print("=" * 65 + "\n", flush=True)
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
