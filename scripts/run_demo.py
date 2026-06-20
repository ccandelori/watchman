from __future__ import annotations

from aegis.demo.scenarios import render_demo_scenarios, run_demo_scenarios


def main() -> None:
    print(render_demo_scenarios(run_demo_scenarios()))


if __name__ == "__main__":
    main()
