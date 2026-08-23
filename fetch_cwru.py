"""Fetch the Case Western Reserve University bearing fault dataset.

WHY THIS EXISTS. The first two passes of this project ran entirely on simulated
vibration, and the README's number-one honest gap was "no real data". Everything
here -- the envelope analysis, the BPFO/BPFI collision and its sideband
tie-break, the health index, the detector bake-off -- was built against a
simulator I wrote, which means it was built against my own assumptions about
what bearing vibration looks like. That is the single largest threat to every
claim in this project, because a simulator cannot falsify the physics it was
written from.

CWRU is the standard public benchmark for exactly this. Real accelerometers, a
real 2 hp motor, real spark-eroded faults of known size on known races, at known
shaft loads.

PROVENANCE AND LICENCE. Data is published by the Case Western Reserve University
Bearing Data Center and is freely available for research use with attribution:

    Case Western Reserve University Bearing Data Center Website
    https://engineering.case.edu/bearingdatacenter

The files are NOT redistributed in this repository. This script downloads them
into data/CWRU/ (which is gitignored), so the repository stays small and the
licence stays theirs.

THE TEST BEARING, which is why the geometry in src/bearing.py was chosen:
the drive-end bearing is an SKF 6205-2RS JEM deep groove ball bearing -- 9
rolling elements, 0.3126 in ball diameter, 1.537 in pitch diameter, 0 degree
contact angle. src/bearing.py was written to those dimensions from the start, so
the fault frequencies this project computes apply to these files directly.

    python fetch_cwru.py            # ~120 MB, 40 files
    python fetch_cwru.py --check    # report what is present, download nothing
"""
from __future__ import annotations

import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DEST = ROOT / "data" / "CWRU"
BASE = "https://engineering.case.edu/sites/default/files/{}.mat"

# 12 kHz drive-end sampling, all four motor loads (0-3 hp).
#
# Fault size matters and is part of the design: 0.007", 0.014" and 0.021" spall
# diameters let a detector be scored on EARLY faults rather than only on obvious
# ones, which is the whole difficulty of condition monitoring. A method that only
# finds 0.021" faults has found bearings that were already going to be caught by
# the operator's ear.
FILES: dict[str, dict] = {}


def _add(fault: str, size_in: float, ids: list[int]) -> None:
    for load_hp, fid in enumerate(ids):
        FILES[str(fid)] = {"fault": fault, "size_in": size_in, "load_hp": load_hp,
                           "rpm_nominal": [1797, 1772, 1750, 1730][load_hp]}


_add("normal", 0.0, [97, 98, 99, 100])
_add("inner_race", 0.007, [105, 106, 107, 108])
_add("ball", 0.007, [118, 119, 120, 121])
_add("outer_race", 0.007, [130, 131, 132, 133])   # centred @6 o'clock, in load zone
_add("inner_race", 0.014, [169, 170, 171, 172])
_add("ball", 0.014, [185, 186, 187, 188])
_add("outer_race", 0.014, [197, 198, 199, 200])
_add("inner_race", 0.021, [209, 210, 211, 212])
_add("ball", 0.021, [222, 223, 224, 225])
_add("outer_race", 0.021, [234, 235, 236, 237])


def main() -> None:
    check_only = "--check" in sys.argv
    DEST.mkdir(parents=True, exist_ok=True)

    have, missing, failed = [], [], []
    for fid, meta in FILES.items():
        target = DEST / f"{fid}.mat"
        if target.exists() and target.stat().st_size > 100_000:
            have.append(fid)
            continue
        missing.append(fid)
        if check_only:
            continue
        url = BASE.format(fid)
        # The host truncates responses fairly often (http.client.IncompleteRead on
        # roughly one file in five). Retrying the whole file is the simple fix:
        # they are a few MB each, and a partial .mat is worse than no .mat because
        # scipy will happily load a truncated array.
        try:
            blob = None
            last = None
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req, timeout=120) as f:
                        expect = f.headers.get("Content-Length")
                        got = f.read()
                    if expect and len(got) != int(expect):
                        last = f"short read {len(got)}/{expect}"
                        continue
                    blob = got
                    break
                except Exception as e:                       # noqa: BLE001
                    last = f"{type(e).__name__}: {e}"
            if blob is None:
                failed.append((fid, last or "unknown"))
                continue
            # A HTML error page is also a 200 on this host, so check the magic.
            if not blob.startswith(b"MATLAB"):
                failed.append((fid, f"not a .mat file ({len(blob)} bytes)"))
                continue
            target.write_bytes(blob)
            have.append(fid)
            print(f"  {fid}.mat  {meta['fault']:11s} {meta['size_in']:.3f}in "
                  f"{meta['load_hp']}hp  {len(blob) / 1e6:.1f} MB", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            failed.append((fid, f"{type(e).__name__}: {e}"))

    manifest = {
        "source": "Case Western Reserve University Bearing Data Center",
        "url": "https://engineering.case.edu/bearingdatacenter",
        "sampling_hz": 12000,
        "bearing": "SKF 6205-2RS JEM deep groove ball, drive end",
        "note": "Not redistributed; fetched by fetch_cwru.py into a gitignored dir.",
        "files": {k: v for k, v in FILES.items() if k in have},
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\npresent {len(have)}/{len(FILES)}  missing {len(missing)}  failed {len(failed)}")
    for fid, why in failed:
        print(f"  FAILED {fid}: {why}")
    if not check_only and failed:
        print("\nRe-run to retry the failures; downloads are resumable per file.")


if __name__ == "__main__":
    main()
