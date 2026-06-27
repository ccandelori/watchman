# Aegis Watchman Desktop Launcher

This is the macOS-first Electron launcher for local Aegis protection.

The app uses a C-lite control plane:

- Electron owns the desktop window, menu, lifecycle, Ollama startup, model pulls,
  and the visible run flow.
- The existing Python `aegis-launcher` FastAPI backend remains the Aegis service
  control API for CIFT, gateway, console, smoke checks, profiles, and logs.
- The browser launcher remains available as a development and fallback surface.

Run from the repository root:

```bash
scripts/run-aegis-desktop-launcher.sh
```

For development inside the desktop package:

```bash
cd desktop/aegis-launcher
npm install
npm run dev
```

Build a double-clickable local macOS app:

```bash
cd desktop/aegis-launcher
npm run pack:mac
open "dist/mac-arm64/Aegis Watchman.app"
```

Build a local DMG:

```bash
cd desktop/aegis-launcher
npm run dist:mac
open dist
```

These local builds are unsigned and not notarized. They are suitable for the
demo machine, not for broad distribution.

Optional environment variables:

```text
AEGIS_DESKTOP_REPO_ROOT=/Users/sheep/Desktop/Gauntlet/Capstone
AEGIS_DESKTOP_BACKEND_PORT=8790
```

If the packaged app cannot find the repo automatically, click **Select Aegis
Folder** and choose the repository root. The selection is saved in the app's
macOS user-data directory for later launches.

The app connects to an existing launcher backend on the selected port when it
belongs to the same repo root. Otherwise it starts:

```bash
uv run aegis-launcher --host 127.0.0.1 --port <port> --repo-root <repo-root>
```

The desktop launcher checks or starts Ollama, pulls the selected model when
needed, starts CIFT and the gateway through the Python backend, then gives
Hermes or another local agent the Aegis URL:

```text
http://127.0.0.1:8000/v1
```

Do not point the agent directly at Ollama on `11434` when demonstrating Aegis
protection.
