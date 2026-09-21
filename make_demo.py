"""Draw the README demo figure from real CWRU recordings.

Three bearings - healthy, outer-race fault, inner-race fault - pushed through the
same envelope pipeline the diagnoser uses. The fault frequencies are computed
from bearing geometry and drawn as dashed lines; the peaks land on them.

    python fetch_cwru.py      # once
    python make_demo.py       # writes docs/img/envelope_demo.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
import bearing  # noqa: E402
import cwru  # noqa: E402
import features  # noqa: E402

OUT = Path(__file__).parent / "docs" / "img"
CASES = [("97", "Healthy bearing"),
         ("130", "Outer-race fault"),
         ("105", "Inner-race fault")]
LINES = {"BPFO": "#d62728", "BPFI": "#1f77b4"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = bearing.BearingGeometry()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for ax, (fid, title) in zip(axes, CASES):
        rec = cwru.load_file(fid)
        seg = cwru.snapshots(rec["de"], n=1)[0]
        fs = rec["fs"]
        freqs, spec, _ = features.envelope_spectrum(seg, cwru.band_for_fs(fs), fs=fs)
        keep = freqs <= 400
        ax.plot(freqs[keep], spec[keep] * 1e3, color="#333333", lw=0.8)
        ff = geom.fault_frequencies(cwru.shaft_hz(rec, 1797))
        for name, color in LINES.items():
            for h in (1, 2, 3):
                f = ff[name] * h
                if f <= 400:
                    ax.axvline(f, color=color, ls="--", lw=1, alpha=0.7,
                               label=name if h == 1 else None)
        ax.set_title(f"{title} (CWRU file {fid})", loc="left", fontsize=11)
        ax.set_ylabel("envelope amplitude")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("frequency (Hz)")
    fig.suptitle("Real vibration data: fault peaks land on the frequencies computed "
                 "from bearing geometry", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "envelope_demo.png", dpi=130)
    print("wrote", OUT / "envelope_demo.png")


if __name__ == "__main__":
    main()
