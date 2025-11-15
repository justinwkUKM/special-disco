import sys
from pathlib import Path

# Ensure both the repository root and the package directory are on sys.path
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "pii_pipeline"
for path in (str(ROOT), str(PACKAGE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
