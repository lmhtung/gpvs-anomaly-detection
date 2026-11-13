"""Redraw stored onset evidence without fitting."""
from pathlib import Path
import sys

if __package__:
    from .workflow import main
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from mmd_onset.workflow import main

if __name__ == '__main__':
    main(plot_only=True)
