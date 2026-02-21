"""Streamlit Cloud entrypoint.

Streamlit Community Cloud expects the main app file at the repository root.
This wrapper runs the actual app located in `frontend/app.py`.
"""

from __future__ import annotations

from pathlib import Path
import runpy


def main() -> None:
    app_path = Path(__file__).resolve().parent / "frontend" / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")


if __name__ == "__main__":
    main()
