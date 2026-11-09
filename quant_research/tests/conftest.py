import os
import sys
import tempfile
from pathlib import Path

# isolate test runs from the real research memory
os.environ.setdefault("QR_DATA_DIR", tempfile.mkdtemp(prefix="qrlab-test-"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
