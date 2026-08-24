"""Render the fleet dashboard from the measured CWRU and process results.

    python run_dashboard.py        # writes out/fleet_dashboard.html
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import dashboard as D  # noqa: E402


def main() -> None:
    out = D.render_from_out("out")
    print(f"wrote {out['path']} ({out['bytes'] / 1024:.0f} KB)")
    print(f"  {out['assets']} assets: {out['faulty']} faulty, "
          f"{out['indeterminate']} abstaining, "
          f"{out['healthy_assets_called_healthy']} healthy")
    print(f"  correct race on {out['correct']}/{out['graded']} where one is expected")
    print(f"  {out['process_runs']} process faults with contribution decompositions")
    if out["healthy_assets_called_healthy"] == 0 and out["truly_healthy_assets"]:
        print(f"  NOTE: {out['truly_healthy_assets']} healthy bearings and none "
              "is called healthy -- the page says so rather than just being red")


if __name__ == "__main__":
    main()
