"""Wrapper to run a Python script and redirect stdout/stderr to a log file.

Usage:
    python run_with_log.py <script_path> [log_path]

This wrapper uses runpy.run_path to execute the target script in __main__ so
that normal import/run semantics are preserved. It sets SYS stdout/stderr to
an appended, line-buffered file so jax.debug.print and normal prints are
captured.
"""
import sys
import runpy
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: run_with_log.py <script> [logfile]")
        sys.exit(2)
    script = sys.argv[1]
    logfile = sys.argv[2] if len(sys.argv) > 2 else "jax_debug.log"
    logpath = Path(logfile)
    logpath.parent.mkdir(parents=True, exist_ok=True)
    # Open file in append mode with line buffering
    f = open(logpath, "a", buffering=1)
    # Replace stdout/stderr so all prints and jax.debug.print go to file
    sys.stdout = f
    sys.stderr = f
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        try:
            f.flush()
            f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
