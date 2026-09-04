"""EMD band sweep on R008 merged audit data (replicates detector preprocessing).

Recomputes, for candidate EMD prior bands, the statistics that depend on the
band: window/supported/subject in-band fractions, EAI, OBR, OPF. Detector
preprocessing is replicated exactly from r007/r008_true detector reports:
fs=2000, block_samples=10 -> 5 ms overlap exclusion, max_lag_ms=150,
smooth_sigma_bins=2.0 (scipy gaussian_filter1d).

Usage (from repo root):
    python 00_research/docs/audits/emd_band_sweep.py                 # argmax report (r007)
    python 00_research/docs/audits/emd_band_sweep.py r008_true       # true-mode report

Inputs (must exist):
    03_bspc/05_revision_experiments/results/bspc_revision_v2/r008_merged/true_trained.npz
    03_bspc/05_revision_experiments/results/bspc_revision_v2/{r007_detector_report,r008_true_detector_report}.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[2]  # 05_revision_experiments（bspc 体系内）
ROOT = REPO_ROOT / "results" / "bspc_revision_v2"
CURVES = ROOT / "r008_merged" / "true_trained.npz"
REPORTS = {
    "r007": ROOT / "r007_detector_report.json",          # argmax 口径 (main audit)
    "r008_true": ROOT / "r008_true_detector_report.json",  # true-mode 口径 (secondary)
}

CANDIDATE_BANDS = [
    (10, 120),  # JEK 2024 expert-consensus range
    (20, 120),
    (20, 100),
    (20, 80),
    (30, 100),  # manuscript prior
    (30, 80),
    (10, 60),
    (20, 60),
    (10, 100),
]


def subject_of(curve_id):
    s = str(curve_id).split("_")[0]
    return int(s[1:]) if s[0] == "S" else None


def prep(lags, curve):
    """exclude |lag|<5ms, keep |lag|<=150ms, smooth sigma=2 (detector-identical)."""
    mask = (np.abs(lags) >= 5.0) & (np.abs(lags) <= 150.0)
    l, v = lags[mask], curve[mask]
    if v.size < 2:
        return l, v
    return l, gaussian_filter1d(v, sigma=2.0)


def main(report_key: str) -> None:
    arch = np.load(CURVES, allow_pickle=False)
    lags = arch["lags_ms"].astype(float)
    curves = np.asarray(arch["interactions"], dtype=np.float64)
    ids = arch["ids"]

    report = json.loads(REPORTS[report_key].read_text(encoding="utf-8"))
    dec = {d["id"]: d for d in report["decisions"]}
    assert len(dec) == len(ids)

    pks = np.array([dec[i]["statistics"]["peak_lag_ms"] for i in ids])   # window-level peaks (detector-authoritative)
    sup = np.array([dec[i]["supported"] for i in ids])
    pk_sup = pks[sup]

    # subject-level peaks: mean spectrum per subject then argmax (S(tau) per subject)
    subs = np.array([subject_of(i) for i in ids])
    sub_pk = {}
    for s in np.unique(subs):
        l, v = prep(lags, curves[subs == s].mean(axis=0))
        sub_pk[int(s)] = l[np.argmax(v)]
    sp = np.array(list(sub_pk.values()))

    # window-aggregated spectrum S(tau) = mean I+ curve (matches paper definition)
    lS, vS = prep(lags, curves.mean(axis=0))
    tot = vS.sum()

    n, n_sup = len(pks), int(sup.sum())
    print(f"report={report_key} supported={n_sup}/{n} ({n_sup / n:.1%}) "
          f"report-median={report['summary']['supported_peak_lag_ms']['median']} ms")
    print(f"{'band (ms)':<12}{'win-in':>9}{'sup-in':>9}{'subj-in':>9}{'EAI':>8}{'OBR':>8}{'OPF':>8}")
    for lo, hi in CANDIDATE_BANDS:
        f_all = float(((pks >= lo) & (pks <= hi)).mean())
        f_sup = float(((pk_sup >= lo) & (pk_sup <= hi)).mean())
        f_sub = float(((sp >= lo) & (sp <= hi)).mean())
        eai = float(vS[(lS >= lo) & (lS <= hi)].sum() / tot)
        obr = float(vS[(lS >= 5) & (lS < lo)].sum() / tot) if lo > 5 else 0.0
        opf = float((sp < lo).mean()) if lo > 5 else 0.0
        tag = "  <- manuscript" if (lo, hi) == (30, 100) else ""
        print(f"[{lo:>3},{hi:>4}]  {f_all:>7.1%}{f_sup:>8.1%}{f_sub:>8.1%}{eai:>7.1%}{obr:>7.1%}{opf:>7.1%}{tag}")

    print(f"supported peak lag: min={pk_sup.min():.1f} p25={np.percentile(pk_sup, 25):.1f} "
          f"median={np.median(pk_sup):.1f} p75={np.percentile(pk_sup, 75):.1f} max={pk_sup.max():.1f} ms")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "r007"
    if key not in REPORTS:
        raise SystemExit(f"unknown report key {key!r}; choose from {sorted(REPORTS)}")
    main(key)
