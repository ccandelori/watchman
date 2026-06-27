#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT/desktop/aegis-launcher"

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to run the Aegis desktop launcher." >&2
  exit 1
fi

if [ ! -x "$APP_DIR/node_modules/.bin/electron" ]; then
  npm --prefix "$APP_DIR" install
fi

AEGIS_DESKTOP_REPO_ROOT="${AEGIS_DESKTOP_REPO_ROOT:-$ROOT}" \
npm --prefix "$APP_DIR" run dev
