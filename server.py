"""
RideGuard AI - Enterprise Mobility Intelligence Platform
Root Application Launcher
"""

import sys
from pathlib import Path

# Configure paths for segregated architecture
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
MODELS_DIR = ROOT_DIR / "models"

for p in [str(ROOT_DIR), str(BACKEND_DIR), str(MODELS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.server import app

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print("🛡️  RIDEGUARD AI — STANDALONE ENTERPRISE WEB APPLICATION")
    print("=" * 65)
    print("🚀 Initializing FastAPI Backend Engine & Serving Web Dashboard...")
    print("🌐 Web Application URL  : http://localhost:8000")
    print("📚 Interactive API Docs : http://localhost:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=True)
