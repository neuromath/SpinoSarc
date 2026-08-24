"""SpinoSarc application and bundled TotalSpineSeg worker entry point."""

from __future__ import annotations

import os
import sys
import multiprocessing
from pathlib import Path


WORKER_FLAG = "--spinosarc-tss-worker"


def _run_totalspineseg_worker() -> None:
    # Remove the private dispatcher argument before TotalSpineSeg parses argv.
    sys.argv.remove(WORKER_FLAG)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled_data = Path(sys._MEIPASS) / "totalspineseg_data"
        if bundled_data.is_dir() and "--data-dir" not in sys.argv and "-d" not in sys.argv:
            sys.argv.extend(["--data-dir", str(bundled_data)])

    from totalspineseg.inference import main as totalspineseg_main

    totalspineseg_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if WORKER_FLAG in sys.argv:
        _run_totalspineseg_worker()
    else:
        from spinosarc_app.gui import main

        main()
