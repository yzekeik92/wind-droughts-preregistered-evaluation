#!/usr/bin/env python3
"""
H1 equivalence / Bayes-factor analysis (v6 revision, reviewer-prompted).

Goal
----
Convert the "inconclusive by the pre-specified bounds" verdict of H1 into a
quantified statement about *absence of transfer*, without touching the
pre-specified decision rule. The frozen prereg (v1.2) pre-specifies the
margin: H1 supported if the 95% spatial-block-bootstrap CI upper bound of the
Spearman rho between G_RMSE and G_pers is < 0.5; refuted if lower bound >= 0.5.
Observed (confirmatory run, v1/v3): rho = 0.16, CI [-0.10, 0.62] -> inconclusive.

Added here (declared post hoc in the deviation log, #16):
  1. TOST against the SAME pre-specified margin (|rho| < 0.5), using the
     pre-specified spatial-block bootstrap (8 single-linkage 200 km blocks).
  2. One-sided JZS Bayes factor BF_neg,tr = evidence for
     H_neg: rho < 0.5 (negligible transfer) vs H_tr: rho >= 0.5 (meaningful),
     plus the classic BF01 (rho = 0 vs rho != 0) and the posterior
     probability P(rho < 0.5 | data) under a stretched-beta prior.
     Correlation likelihood: Ly, Verhagen & Wagenmakers (2016), applied to
     rank-transformed pairs (Spearman approximation, no ties present).
  3. Sensitivities: log-ratio re-expression; Pearson on raw log-ratio pairs;
     prior width a=b in {0.5, 1, 2}; block-mean pairs (n = 8, the most
     conservative treatment of spatial dependence); n_eff = 23 (ICC-based).

Inputs (frozen confirmatory outputs, unchanged):
  download/M2_results/station_metrics_confirmatory.csv  (32 stations)
  download/M1a_evidence/hadisd_catalog.csv              (coordinates)

Output:
  download/M2_results/h1_equivalence_v6.json
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, stats
from scipy.special import betaln, gammaln, hyp2f1

BASE = Path("/home/z/my-project")
M2 = BASE / "download" / "M2_results"
M1A = BASE / "download" / "M1a_evidence"
OUT_JSON = M2 / "h1_equivalence_v6.json"

SEED = 20261017
N_BOOT = 50_000
MARGIN = 0.5          # pre-specified in prereg v1.2 (frozen)
RNG = np.random.default_rng(SEED)


# --------------------------------------------------------------------------
# Spatial blocks: identical to scripts/m2_analysis.py (single-linkage 200 km)
# --------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def spatial_blocks(sids, lats, lons, thr_km=200.0):
    n = len(sids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in combinations(range(n), 2):
        if haversine(lats[i], lons[i], lats[j], lons[j]) <= thr_km:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def block_boot_idx(blocks, n_draw, rng):
    res = []
    for _ in range(n_draw):
        picks = rng.choice(len(blocks), size=len(blocks), replace=True)
        idx = np.concatenate([blocks[p] for p in picks])
        res.append(idx)
    return res


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


# --------------------------------------------------------------------------
# Correlation likelihood (Ly, Verhagen & Wagenmakers 2016, eq. 17 / Bulmer)
# --------------------------------------------------------------------------
def log_lik_rho(r, n, rho):
    """log p(r | n, rho) for a Pearson correlation from n iid bivariate-normal
    pairs. Valid for r, rho in (-1, 1)."""
    r, rho = float(r), float(rho)
    if abs(rho) >= 1.0 or abs(r) >= 1.0:
        return -np.inf
    lr = (np.log(n - 2.0) + gammaln(n - 1.0)
          + 0.5 * (n - 1.0) * np.log1p(-rho * rho)
          + 0.5 * (n - 4.0) * np.log1p(-r * r)
          - 0.5 * np.log(2.0 * np.pi) - gammaln(n - 0.5)
          - (n - 1.5) * np.log1p(-rho * r))
    # hyp2f1(1/2, 1/2; n - 1/2; (1 + rho*r)/2): handle rho*r -> -1 edge
    z = (1.0 + rho * r) / 2.0
    h = hyp2f1(0.5, 0.5, n - 0.5, z)
    if not np.isfinite(h) or h <= 0:
        return -np.inf
    return lr + np.log(h)


def null_log_density(r, n):
    """f(r | rho = 0) = (1-r^2)^((n-4)/2) / B(1/2, (n-2)/2).

    Normaliser verified by simulation (400k null draws, n=10): the closed
    form must be B(1/2, (n-2)/2), NOT B(1/2, (n-1)/2); and the exponent
    carries a positive sign (density decreases with r^2)."""
    return ((0.5 * (n - 4.0)) * np.log1p(-r * r)
            - betaln(0.5, 0.5 * (n - 2.0)))


def stretched_beta_logpdf(rho, a, b):
    """pi(rho) = (1+rho)^(a-1) (1-rho)^(b-1) / (2^(a+b-1) B(a,b)) on (-1,1)."""
    if abs(rho) >= 1.0:
        return -np.inf
    return ((a - 1.0) * np.log1p(rho) + (b - 1.0) * np.log1p(-rho)
            - (a + b - 1.0) * np.log(2.0) - betaln(a, b))


def bf_correlation(r, n, a=1.0, b=1.0, margin=None):
    """JZS correlation Bayes factors.

    All integrals use dense-grid trapezoid on a fixed grid of rho in (-1, 1)
    (the likelihood in rho is sharply peaked; quad with default settings was
    verified to agree exactly with the same integrand, and the grid density
    resolves the peak far below its width).

    Returns dict with:
      bf01        : H0 rho=0 vs H1 rho!=0  = f(r|0) / int f(r|rho) pi d rho
      bf_neg_tr   : H_neg (-1,margin) vs H_tr (margin,1), composite vs
                    composite with prior-mass correction (if margin given);
                    normalisation-independent (ratios of the same integrand)
      post_p_neg  : posterior P(rho < margin | data) under the full prior
      prior_p_neg : prior P(rho < margin)
    """
    base = null_log_density(r, n)

    # log integrand of f(r|rho) * pi(rho) over the fixed grid
    log_w = np.empty_like(RHO_GRID)
    for k, rho_k in enumerate(RHO_GRID):
        lf = log_lik_rho(r, n, rho_k)
        lp = stretched_beta_logpdf(rho_k, a, b)
        log_w[k] = (lf + lp) if np.isfinite(lf) else -np.inf

    # stable trapezoid: offset by the grid max
    off = np.max(log_w[np.isfinite(log_w)])
    w = np.exp(log_w - off)
    Z_unnorm = float(np.trapezoid(w, RHO_GRID))
    Z = Z_unnorm * np.exp(off)            # int f(r|rho) pi(rho) d rho

    # f(r|0) = exp(base); bf01 = f(r|0)/Z  (validated against MC: see below)
    out = {"bf01": float(np.exp(base) / Z), "post_mass_total": float(Z)}

    if margin is not None:
        sel_neg = RHO_GRID < margin
        sel_tr = ~sel_neg
        m_neg = (np.trapezoid(w[sel_neg], RHO_GRID[sel_neg]) * np.exp(off))
        m_tr = Z - m_neg
        a_neg = 0.5 * (margin + 1.0)
        a_tr = 0.5 * (1.0 - margin)
        out["bf_neg_tr"] = float((m_neg / m_tr) / (a_neg / a_tr))
        out["post_p_neg"] = float(m_neg / Z)
        out["prior_p_neg"] = float(a_neg)
        out["post_mass_neg"] = float(m_neg)
        out["post_mass_tr"] = float(m_tr)
    return out


# fixed integration grid (spacing 2.5e-5, far finer than the likelihood
# width ~0.18); reused for every BF call
RHO_GRID = np.linspace(-1.0 + 1e-9, 1.0 - 1e-9, 160_001)


# --------------------------------------------------------------------------
# 1) Load frozen confirmatory per-station metrics
# --------------------------------------------------------------------------
df = pd.read_csv(M2 / "station_metrics_confirmatory.csv", dtype={"station_id": str})
cat = pd.read_csv(M1A / "hadisd_catalog.csv", dtype={"station_id": str}).set_index("station_id")
assert len(df) == 32, f"expected 32 confirmatory stations, got {len(df)}"

sids = df["station_id"].tolist()
rmse_c = df["rmse_cerra"].to_numpy(float)
rmse_e = df["rmse_era5"].to_numpy(float)
E_c = df["E_pers_cerra"].to_numpy(float)
E_e = df["E_pers_era5"].to_numpy(float)

G_rmse = rmse_e - rmse_c
G_pers = E_e - E_c
g_rmse_log = np.log(rmse_e / rmse_c)
g_pers_log = np.log(E_e / E_c)

lats = np.array([cat.loc[s, "lat"] for s in sids])
lons = np.array([cat.loc[s, "lon"] for s in sids])

# sanity vs frozen results (v1/v3): means and point correlations
checks = {
    "mean_G_rmse": float(G_rmse.mean()),       # ~ +0.20
    "mean_G_pers": float(G_pers.mean()),       # ~ +0.005
    "median_log_rmse": float(np.median(g_rmse_log)),   # ~ 0.160 (15%)
    "median_log_pers": float(np.median(g_pers_log)),   # ~ 0.025 (2.5%)
}
assert abs(checks["mean_G_rmse"] - 0.204) < 0.01, checks
assert abs(checks["mean_G_pers"] - 0.005) < 0.002, checks
assert abs(checks["median_log_rmse"] - 0.160) < 0.01, checks
assert abs(checks["median_log_pers"] - 0.025) < 0.01, checks

blocks = spatial_blocks(sids, lats, lons, 200.0)
print(f"spatial blocks (200 km): {len(blocks)} blocks, sizes {[len(b) for b in blocks]}")
assert len(blocks) == 8, "expected the 8 confirmatory spatial blocks"

# --------------------------------------------------------------------------
# 2) Point statistics (verify against manuscript-quoted values)
# --------------------------------------------------------------------------
rho_diff = spearman(G_rmse, G_pers)          # manuscript: 0.16
rho_log = spearman(g_rmse_log, g_pers_log)   # manuscript: 0.21
pearson_log = float(np.corrcoef(g_rmse_log, g_pers_log)[0, 1])
print(f"rho_diff = {rho_diff:.4f} (expect ~0.16)")
print(f"rho_log  = {rho_log:.4f} (expect ~0.21)")
print(f"pearson_log = {pearson_log:.4f}")
assert abs(rho_diff - 0.16) < 0.01 and abs(rho_log - 0.21) < 0.01

# --------------------------------------------------------------------------
# 3) Validate likelihood implementation
# --------------------------------------------------------------------------
_r, _n = 0.25, 32
lhs = log_lik_rho(_r, _n, 0.0)
rhs = null_log_density(_r, _n)
assert abs(lhs - rhs) < 1e-10, (lhs, rhs)
_int = integrate.quad(lambda rr: np.exp(log_lik_rho(rr, _n, 0.3)), -1, 1, limit=400)[0]
assert abs(_int - 1.0) < 1e-6, _int
print("likelihood validation: rho=0 closed form OK; integral over r = 1 OK")

# --------------------------------------------------------------------------
# 4) Spatial block bootstrap (pre-specified scheme, 50k draws, fresh seed)
# --------------------------------------------------------------------------
bidx = block_boot_idx(blocks, N_BOOT, RNG)
boot_rho_diff, boot_rho_log = [], []
for idx in bidx:
    if len(set(idx.tolist())) < 8:
        continue
    r1 = spearman(G_rmse[idx], G_pers[idx])
    r2 = spearman(g_rmse_log[idx], g_pers_log[idx])
    if np.isfinite(r1):
        boot_rho_diff.append(r1)
    if np.isfinite(r2):
        boot_rho_log.append(r2)
boot_rho_diff = np.array(boot_rho_diff)
boot_rho_log = np.array(boot_rho_log)
print(f"bootstrap draws kept: {len(boot_rho_diff)} / {N_BOOT}")

lo_d, hi_d = np.quantile(boot_rho_diff, [0.025, 0.975])
lo_l, hi_l = np.quantile(boot_rho_log, [0.025, 0.975])
print(f"fresh CI diff: [{lo_d:.3f}, {hi_d:.3f}] (frozen run: [-0.10, 0.62])")
print(f"fresh CI log : [{lo_l:.3f}, {hi_l:.3f}] (frozen run: [-0.10, 0.49])")

# --------------------------------------------------------------------------
# 5) TOST against the pre-specified margin (block-bootstrap one-sided p's)
# --------------------------------------------------------------------------
tost = {}
for name, arr, rho_pt in [("diff", boot_rho_diff, rho_diff), ("log", boot_rho_log, rho_log)]:
    p_pos = float(np.mean(arr >= MARGIN))    # H0: rho >= 0.5 (meaningful transfer)
    p_neg = float(np.mean(arr <= -MARGIN))   # H0: rho <= -0.5
    tost[name] = {
        "rho_point": rho_pt,
        "ci95": [float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))],
        "p_pos_ge_margin": p_pos,
        "p_neg_le_minus_margin": p_neg,
        "tost_alpha005_equivalent": bool(p_pos < 0.05 and p_neg < 0.05),
        "prereg_supported_0025": bool(np.quantile(arr, 0.975) < MARGIN),
        "p_pos_ge_0.0": float(np.mean(arr >= 0.0)),
        "boot_mean": float(arr.mean()),
        "boot_sd": float(arr.std(ddof=1)),
    }
    t = tost[name]
    print(f"[{name}] rho={rho_pt:.3f} CI[{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}] "
          f"p+={p_pos:.4f} p-={p_neg:.4f} TOST@0.05={t['tost_alpha005_equivalent']} "
          f"prereg-supported@0.025={t['prereg_supported_0025']}")

# --------------------------------------------------------------------------
# 6) Bayes factors
# --------------------------------------------------------------------------
rank_rmse = stats.rankdata(G_rmse)
rank_pers = stats.rankdata(G_pers)
rank_rmse_log = stats.rankdata(g_rmse_log)
rank_pers_log = stats.rankdata(g_pers_log)

bf = {}

# primary: difference-based Spearman (pre-specified statistic), ranks
prim = bf_correlation(rho_diff, 32, 1.0, 1.0, margin=MARGIN)
bf["primary_diff_spearman"] = {"r": rho_diff, "n": 32, "prior": "stretched-beta a=b=1", **prim}

# log-ratio Spearman
bf["logratio_spearman"] = {"r": rho_log, "n": 32, "prior": "stretched-beta a=b=1",
                           **bf_correlation(rho_log, 32, 1.0, 1.0, margin=MARGIN)}

# Pearson on raw log-ratio pairs (no ranks)
bf["logratio_pearson_raw"] = {"r": pearson_log, "n": 32, "prior": "stretched-beta a=b=1",
                              **bf_correlation(pearson_log, 32, 1.0, 1.0, margin=MARGIN)}

# prior sensitivity on the primary
for ab in (0.5, 2.0):
    bf[f"primary_diff_spearman_prior_ab_{ab}"] = {
        "r": rho_diff, "n": 32, "prior": f"stretched-beta a=b={ab}",
        **bf_correlation(rho_diff, 32, ab, ab, margin=MARGIN)}

# dependence-conservative: block-mean pairs (n = 8)
blk_G_rmse = np.array([G_rmse[b].mean() for b in blocks])
blk_G_pers = np.array([G_pers[b].mean() for b in blocks])
blk_rho = spearman(blk_G_rmse, blk_G_pers)
blk_pearson = float(np.corrcoef(blk_G_rmse, blk_G_pers)[0, 1])
bf["block_means_diff_n8"] = {"r": blk_rho, "n": 8, "prior": "stretched-beta a=b=1",
                             "pearson_r": blk_pearson,
                             **bf_correlation(blk_rho, 8, 1.0, 1.0, margin=MARGIN)}

# ICC-based effective sample size (min n_eff from the v5 share tests: ~23)
bf["primary_diff_spearman_neff23"] = {
    "r": rho_diff, "n": 23, "prior": "stretched-beta a=b=1",
    **bf_correlation(rho_diff, 23, 1.0, 1.0, margin=MARGIN)}

for k, v in bf.items():
    if "bf_neg_tr" in v:
        print(f"BF[{k}]: r={v['r']:.3f} n={v['n']} BF01={v['bf01']:.2f} "
              f"BF_neg,tr={v['bf_neg_tr']:.1f} P(rho<0.5|data)={v['post_p_neg']:.3f}")

# --------------------------------------------------------------------------
# 6b) Independent Monte Carlo cross-check of the BF machinery (primary case):
#     draw rho ~ U(-1,1) from the prior, simulate n=32 bivariate-normal pairs,
#     KDE the marginal density of the sample correlation at r_obs, and compare
#     p(r_obs|H1)_MC with the analytic integral Z. Grid trapezoid must agree.
# --------------------------------------------------------------------------
from scipy.stats import gaussian_kde

_rng = np.random.default_rng(SEED + 1)
_NMC = 300_000
_rho_draw = _rng.uniform(-1, 1, _NMC)
_x = _rng.normal(size=(_NMC, 32))
_y = _rho_draw[:, None] * _x + np.sqrt(1 - _rho_draw[:, None] ** 2) * _rng.normal(size=(_NMC, 32))
_sx = _x - _x.mean(1, keepdims=True)
_sy = _y - _y.mean(1, keepdims=True)
_r_sim = (_sx * _sy).sum(1) / np.sqrt((_sx ** 2).sum(1) * (_sy ** 2).sum(1))
_kde = gaussian_kde(_r_sim, bw_method=0.02)
_p_h1_mc = float(_kde.evaluate([rho_diff])[0])
_p_h0 = float(np.exp(null_log_density(rho_diff, 32)))
_bf01_mc = _p_h0 / _p_h1_mc
_bf01_quad = bf["primary_diff_spearman"]["bf01"]
rel_err = abs(_bf01_mc - _bf01_quad) / _bf01_quad
print(f"MC cross-check: BF01 grid={_bf01_quad:.3f} BF01_MC={_bf01_mc:.3f} "
      f"(rel diff {rel_err:.1%})")
assert rel_err < 0.06, "grid BF01 disagrees with Monte Carlo beyond KDE noise"

# --------------------------------------------------------------------------
# 7) Save
# --------------------------------------------------------------------------
out = {
    "method_note": ("Reviewer-prompted equivalence/BF analysis for H1 (v6). "
                    "Margin 0.5 is pre-specified in frozen prereg v1.2; TOST and "
                    "Bayes factors are post-hoc additions (deviation #16). "
                    "Bootstrap = pre-specified spatial-block scheme, 8 blocks, "
                    f"{N_BOOT} draws, seed {SEED}. BF uses Ly et al. (2016) "
                    "correlation likelihood on rank-transformed pairs."),
    "margin": MARGIN,
    "n_boot": N_BOOT,
    "seed": SEED,
    "blocks_sizes": [len(b) for b in blocks],
    "checks": checks,
    "point": {"rho_diff": rho_diff, "rho_log": rho_log,
              "pearson_log": pearson_log,
              "n_stations_cerra_better_rmse": int((G_rmse > 0).sum())},
    "tost_bootstrap": tost,
    "bayes_factors": bf,
}
OUT_JSON.write_text(json.dumps(out, indent=1))
print(f"\nsaved -> {OUT_JSON}")
