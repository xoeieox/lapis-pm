"""Runnable entrypoint for warmup run-driver: python -m lapis_pm.jagged_seam.warmup"""
import sys

from .warmup import main

if __name__ == "__main__":
    sys.exit(main())
