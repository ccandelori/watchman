from __future__ import annotations

import sys

from aegis.trace_collection.main import run_assignment_cli


def main() -> None:
    run_assignment_cli(argv=tuple(sys.argv[1:]))


if __name__ == "__main__":
    main()
