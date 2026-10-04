"""Repository-root launcher for `python -m ngrst`."""

import importlib.util
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("ngrst_cli", SRC / "ngrst.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


if __name__ == "__main__":
    raise SystemExit(module.main())
