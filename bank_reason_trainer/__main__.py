"""Shim so `python -m bank_reason_trainer …` works."""
from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
