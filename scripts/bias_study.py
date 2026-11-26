"""When does resolving the mechanism of debris matter? Closed-form bias of a conversion that counts all debris as metal
captured with one efficiency, calibrated at one debris composition and used at another (no data needed).

Observed debris per unit wear, relative to all-metal, fully ferromagnetic debris:
    oxide:    1 / (1 - (1 - c) x)            x = share of the observed debris volume that is oxide, c = its metal fraction
    capture:  y + (1 - y) r                  y = ferromagnetic share of the worn volume, r = eta_nonferrous / eta_ferrous
A calibrated factor absorbs a constant composition, so the bias is the ratio of these terms between use and calibration.
Writes outputs/simulation/bias/bias.csv and prints the composition changes that cost a factor 1.25 and 2.

    .venv/Scripts/python.exe scripts/bias_study.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "simulation" / "bias"
C_OX = 0.47                                     # metal fraction of oxide debris used by the model (DESIGN sec. 3)


def oxide(x, c=C_OX):
    return 1.0 / (1.0 - (1.0 - c) * np.asarray(x, float))


def capture(y, r):
    return np.asarray(y, float) + (1.0 - np.asarray(y, float)) * r


def main():
    rows = [{"term": "oxide", "r_or_c": C_OX, "share": x, "factor": float(oxide(x))} for x in np.linspace(0, 1, 21)]
    for r in (0.1, 1 / 3, 0.5):
        rows += [{"term": "capture", "r_or_c": r, "share": 1 - y, "factor": float(1 / capture(y, r))} for y in np.linspace(0, 1, 21)]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "bias.csv", index=False)
    # share of oxide (from none) and of non-ferrous debris (from none) that changes the conversion by a given factor
    for f in (1.25, 2.0):
        x = (1 - 1 / f) / (1 - C_OX)
        print(f"factor {f}: oxide share {x:.3f}" + "".join(
            f"; non-ferrous share {(1 - 1 / f) / (1 - r):.3f} (r = {r:.2f})" for r in (0.1, 1 / 3, 0.5)
            if (1 - 1 / f) / (1 - r) <= 1))
        assert abs(float(oxide(x)) - f) < 1e-12
    print(f"largest oxide factor {float(oxide(1)):.2f}")
    assert abs(1 / float(capture(0.0, 1 / 3)) - 3) < 1e-12


if __name__ == "__main__":
    main()
