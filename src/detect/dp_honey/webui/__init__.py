"""Local web UI for DP-HONEY.

A thin FastAPI app over :mod:`detect.dp_honey.webui.service`, which wraps the
core :mod:`detect.dp_honey` library. Every output stays synthetic and shape-only;
the server binds to localhost only. Install with the optional extra::

    pip install -e ".[dp-honey-ui]"
    python -m detect.dp_honey.webui
"""
