"""Vercel serverless entry point — re-exports the Flask app."""
import sys
from pathlib import Path

# Ensure the project root and src/ are on the import path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from app import app  # noqa: E402
