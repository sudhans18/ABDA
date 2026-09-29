"""Backward-compatible live entry point for the ABDA facial pipeline."""
if __package__:
    from .live import run_live_session
else:  # Supports the original `python face/main.py` invocation.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from face.live import run_live_session


def main() -> None:
    run_live_session()


if __name__ == "__main__":
    main()
