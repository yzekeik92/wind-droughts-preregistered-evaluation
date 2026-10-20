#!/usr/bin/env python3
"""M2 CONFIRMATORY ANALYSIS — implements prereg v1.2 sect.1/7 exactly.

H1 no-transfer: Spearman rho between G_RMSE and G_pers across stations;
   supported if block-bootstrap 95% CI upper < 0.5, refuted if lower >= 0.5.
H2 persistence error: spells >= 5 d below each series' own annual P10; share of
   stations whose mean spell duration falls outside the observational 95%
   sampling CI exceeds nominal 5% (excess test, Holm, spatial-adjusted
   variance). D6 guard: if any station-year P10 = 0 -> floored threshold
   max(P10, 0.5) applied uniformly, both variants reported.
H3 interannual reliability: P90/P50 of annual means inside observational
   block-bootstrap CI in a station proportion compatible with 95%.
H4 homogeneity: Pettitt breaks (block permutation, BH-FDR 5%) in annual
   reanalysis-minus-station differences; excess vs chance + neighbour
   coincidence (+/-2 y); ERA5-CERRA cross-check.

Conventions fixed before results (documented as D7-D9 in REVIEWER_PROTOCOL):
  D7: day-of-year climatology, centered +/-15 d window (leap 366->365).
  D8: spatial blocks = single-linkage clusters at 200 km; year blocks L=3.
  D9: O5 gaps within qualifying years linearly interpolated (identical masks).

Outputs: verification/m2_results.json, download/M2_results/*.csv, npz for figures.
"""
import json
import math
import os
from itertools import combinations

import numpy as np
import pandas as pd

PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
DEM = '/tmp/my-project/verification/dem_covariates.csv'
PILOT_JSON = '/tmp/my-project/verification/pilot_draw.json'
OUT_JSON = '/tmp/my-project/verification/m2_results.json'
OUT_DIR = '/home/z/my-project/download/M2_results'
NPERM = 10000
RNG = np.random.default_rng(20261017)
SENS3 = ['603400', '605220', '605570']

CLASSES = {  # explicit, documented assignment (Fase 1)
    'archipelago': ['600010', '600050', '600150', '600200', '600250', '600300',
                    '600350', '600400'],
    'atlantic': ['600330', '600600', '602520', '614150'],
    'strait_med': ['601010', '601050', '603180', '604900', '605200', '605310',
                   '603400'],
    'interior': ['601150', '601270', '601350', '601410', '601500', '601560',
                 '601950', '602100', '602300', '602650', '605220', '605570',
                 '605710', '606020', '606200', '606560'],
}
CLASS_OF = {s: c for c, ss in CLASSES.items() for s in ss}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def spatial_blocks(sids, lats, lons, thr_km=200.0):
    """Single-linkage clusters at thr_km; returns list of index-lists."""
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


def block_boot_idx(blocks, n_draw, n_items, rng):
    """Indices of resampled items by resampling whole blocks."""
    res = []
    for _ in range(n_draw):
        picks = rng.choice(len(blocks), size=len(blocks), replace=True)
        idx = np.concatenate([blocks[p] for p in picks])
        res.append(idx)
    return res


def runs_gek(low, day_int, k):
    """Spells of consecutive calendar days where low is True; returns
    (list_of_lengths >=1). Gaps (day diff > 1) break runs."""
    if len(low) == 0:
        return []
    brk = np.r_[True, np.diff(day_int) != 1, True]
    seg_start = np.flatnonzero(brk)[:-1]
    seg_end = np.flatnonzero(brk)[1:]
    out = []
    for s, e in zip(seg_start, seg_end):
        seg = low[s:e]
        if not seg.any():
            continue
        d = np.diff(np.r_[0, seg.astype('int8'), 0])
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        out.extend((ends - starts).tolist())
    return out


def spell_stats(low, day_int, kmin=5):
    lens = runs_gek(low, day_int, kmin)
    if not lens:
        return dict(n_spells=0, mean_dur=np.nan, max_dur=np.nan,
                    frac_days=0.0, n_low_days=int(low.sum()))
    lens = np.array(lens)
    big = lens[lens >= kmin]
    return dict(n_spells=int(len(big)), mean_dur=float(big.mean()) if len(big) else np.nan,
                max_dur=int(big.max()) if len(big) else 0,
                frac_days=float(big.sum() / len(low)),
                n_low_days=int(low.sum()))


def acf_series(anom, day_int, lags=range(1, 11)):
    pos = {d: i for i, d in enumerate(day_int)}
    keys = np.fromiter(pos.keys(), dtype="int64", count=len(pos))
    out = {}
    for k in lags:
        d2 = day_int + k
        sel = np.isin(d2, keys)
        if sel.sum() < 30:
            out[k] = np.nan
            continue
        i1 = np.flatnonzero(sel)
        i2 = np.array([pos[d] for d in d2[sel]])
        a, b = anom[i1], anom[i2]
        ok = np.isfinite(a) & np.isfinite(b)
        out[k] = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 30 else np.nan
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    draw = json.load(open(PILOT_JSON))
    CONF = draw["confirmatory_stations"]
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    dem = pd.read_csv(DEM, dtype={"station_id": str}).set_index("station_id")

    # ---------------- per-station tables (confirmatory, valid days) ----------
    stations = {}
    any_zero_p10 = False
    print("== per-station metrics ==")
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)].copy()
        g["year"] = g["date"].dt.year
        n = len(g)
        rec = {"n_days": n, "class": CLASS_OF[sid],
               "lat": float(g["lat"].iloc[0]) if "lat" in g else np.nan}
        for prod, col in (("cerra", "ws_cerra"), ("era5", "ws_era5")):
            e = g[col] - g.ws_obs
            rec[f"bias_{prod}"] = float(e.mean())
            rec[f"mae_{prod}"] = float(e.abs().mean())
            rec[f"rmse_{prod}"] = float(np.sqrt((e ** 2).mean()))
            rec[f"pearson_{prod}"] = float(np.corrcoef(g.ws_obs, g[col])[0, 1])
        # day-of-year climatology anomalies (D7)
        doy = g["date"].dt.dayofyear.to_numpy()
        doy = np.where(doy == 366, 365, doy)
        day_int = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        anom = {}
        for col in ("ws_obs", "ws_cerra", "ws_era5"):
            x = g[col].to_numpy()
            clim = np.full(366, np.nan)
            for d in range(1, 366):
                w = np.abs(((doy - d + 182) % 365) - 182) <= 15
                if w.sum() >= 30:
                    clim[d] = x[w].mean()
            anom[col] = x - clim[doy]
        acf_o = acf_series(anom["ws_obs"], day_int)
        for prod in ("cerra", "era5"):
            a = acf_series(anom[f"ws_{prod}"], day_int)
            e = [a[k] - acf_o[k] for k in range(1, 11)]
            rec[f"E_pers_{prod}"] = float(np.sqrt(np.nanmean(np.array(e) ** 2)))
            rec[f"acf1_{prod}"] = a[1]
        rec["acf1_obs"] = acf_o[1]
        # AR(1) reference (analytic ACF phi^k)
        phi = acf_o[1]
        rec["E_ar1"] = float(np.sqrt(np.mean([(phi ** k - acf_o[k]) ** 2
                                              for k in range(1, 11)])))
        # annual means & ratios (P90 = value exceeded 90% of years = q10)
        ann = g.groupby("year")[["ws_obs", "ws_cerra", "ws_era5"]].mean()
        ann_n = g.groupby("year").size()
        ann = ann[ann_n >= 274]
        rec["n_years"] = len(ann)
        for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
            v = ann[col].to_numpy()
            q10, q50, q90 = np.quantile(v, [0.10, 0.5, 0.90])
            rec[f"P90_{prod}"] = float(q10)
            rec[f"P50_{prod}"] = float(q50)
            rec[f"P10_{prod}"] = float(q90)
            rec[f"ratio_{prod}"] = float(q10 / q50) if q50 > 0 else np.nan
            rec[f"cv_{prod}"] = float(v.std(ddof=1) / v.mean())
        # annual P10 of daily values per series (H2 thresholds, D6 trigger)
        thr = {}
        for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
            t_y = g.groupby("year")[col].quantile(0.10)
            thr[prod] = t_y
            if (t_y == 0.0).any():
                any_zero_p10 = True
        rec["thr_obs"] = {int(y): float(v) for y, v in thr["obs"].items()}
        # primary spell stats (P10 own, k=5), computed later in H2 block
        rec["_g"] = g
        rec["_thr"] = thr
        stations[sid] = rec
        print(f"  {sid} n={n} RMSE c/e {rec['rmse_cerra']:.2f}/{rec['rmse_era5']:.2f} "
              f"E_pers {rec['E_pers_cerra']:.2f}/{rec['E_pers_era5']:.2f} "
              f"ratio {rec['ratio_obs']:.3f}/{rec['ratio_cerra']:.3f}/{rec['ratio_era5']:.3f}")

    sids = CONF
    lats = np.array([dem.loc[s, "lat"] for s in sids])
    lons = np.array([dem.loc[s, "lon"] for s in sids])
    blocks = spatial_blocks(sids, lats, lons, 200.0)
    print(f"spatial blocks (200 km): {len(blocks)} blocks for {len(sids)} stations")

    results = {"n_confirmatory": len(CONF), "any_zero_p10": bool(any_zero_p10),
               "n_spatial_blocks": len(blocks)}

    # ---------------- H1 -----------------------------------------------------
    G_rmse = np.array([stations[s]["rmse_era5"] - stations[s]["rmse_cerra"] for s in sids])
    G_pers = np.array([stations[s]["E_pers_era5"] - stations[s]["E_pers_cerra"] for s in sids])
    rho_obs = float(pd.Series(G_rmse).rank().corr(pd.Series(G_pers).rank()))
    boot_rho = []
    bidx = block_boot_idx(blocks, NPERM, len(sids), RNG)
    for idx in bidx:
        if len(set(idx)) < 8:
            continue
        r = pd.Series(G_rmse[idx]).rank().corr(pd.Series(G_pers[idx]).rank())
        if np.isfinite(r):
            boot_rho.append(r)
    lo, hi = np.quantile(boot_rho, [0.025, 0.975])
    h1 = {"rho": rho_obs, "ci": [float(lo), float(hi)],
          "verdict": ("supported" if hi < 0.5 else "refuted" if lo >= 0.5
                      else "inconclusive"),
          "mean_G_rmse": float(G_rmse.mean()), "mean_G_pers": float(G_pers.mean()),
          "n_stations_cerra_better_rmse": int((G_rmse > 0).sum())}
    results["H1"] = h1
    print(f"H1 rho={rho_obs:.3f} CI[{lo:.3f},{hi:.3f}] -> {h1['verdict']}")

    # ---------------- H2 -----------------------------------------------------
    kmin = 5
    print("== H2 spells ==")

    def spell_table(sids_, variant):
        """variant: 'raw' (own P10) or 'floored' (max(P10,0.5)) — applied
        uniformly when the D6 trigger fires."""
        rows = {}
        for sid in sids_:
            g = stations[sid]["_g"]
            thr = stations[sid]["_thr"]
            day_int = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
            yr = g["year"].to_numpy()
            stats = {}
            for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
                t_map = dict(zip(thr[prod].index, thr[prod].to_numpy()))
                if variant == "floored":
                    t_arr = np.array([max(t_map.get(y, np.nan), 0.5) for y in yr])
                else:
                    t_arr = np.array([t_map.get(y, np.nan) for y in yr])
                low = g[col].to_numpy() <= t_arr
                stats[prod] = spell_stats(low, day_int, kmin)
            rows[sid] = stats
        return rows

    variant = "floored" if any_zero_p10 else "raw"
    sp = spell_table(sids, variant)
    sp_raw = spell_table(sids, "raw") if any_zero_p10 else None

    def obs_spell_ci(sid):
        """Observational sampling CI of mean spell duration (>=kmin) via
        moving-block bootstrap over years (L=3, circular)."""
        g = stations[sid]["_g"]
        thr = stations[sid]["_thr"]
        yrs = sorted(thr["obs"].index)
        t_map = dict(zip(thr["obs"].index, thr["obs"].to_numpy()))
        vals = [(g[g.year == y].ws_obs.to_numpy(),
                 max(t_map[y], 0.5) if variant == "floored" else t_map[y])
                for y in yrs]
        blocksy = [list(range(s, min(s + 3, len(yrs))))
                   for s in range(0, len(yrs), 3)]
        boots = []
        for _ in range(NPERM):
            picks = RNG.choice(len(blocksy), size=len(blocksy), replace=True)
            chosen = [b for p in picks for b in blocksy[p]]
            low = np.concatenate([vals[b][0] <= vals[b][1] for b in chosen])
            dint = np.concatenate([np.arange(len(vals[b][0])) + (i * 10**6)
                                   for i, b in enumerate(chosen)])
            st = spell_stats(low, dint, kmin)
            if st["n_spells"] >= 1:
                boots.append(st["mean_dur"])
        if len(boots) >= 500:
            return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        return None

    ci_obs = {sid: obs_spell_ci(sid) for sid in CONF}
    outside = {"obs_spells": 0}
    tab = []
    for sid in sids:
        m = sp[sid]["obs"]
        if sid not in ci_obs or m["n_spells"] == 0:
            continue
        lohi = ci_obs[sid]
        row = {"station_id": sid, "class": CLASS_OF[sid],
               "obs_mean_dur": m["mean_dur"], "obs_ci_low": lohi[0],
               "obs_ci_high": lohi[1],
               "cerra_mean_dur": sp[sid]["cerra"]["mean_dur"],
               "era5_mean_dur": sp[sid]["era5"]["mean_dur"],
               "cerra_out": int(not (lohi[0] <= sp[sid]["cerra"]["mean_dur"] <= lohi[1])),
               "era5_out": int(not (lohi[0] <= sp[sid]["era5"]["mean_dur"] <= lohi[1])),
               "cerra_n_spells": sp[sid]["cerra"]["n_spells"],
               "era5_n_spells": sp[sid]["era5"]["n_spells"]}
        tab.append(row)
    h2tab = pd.DataFrame(tab)
    assert len(h2tab) > 0, "H2: no station with obs spells >= kmin"
    def p_upper(z):
        return 0.5 * math.erfc(z / math.sqrt(2))
    holm = {}
    h2_detail = {}
    for prod in ("cerra", "era5"):
        phat = float(h2tab[f"{prod}_out"].mean())
        idx_arr = h2tab[f"{prod}_out"].to_numpy().astype(float)
        bs = []
        for _ in range(NPERM):
            picks = RNG.choice(len(blocks), size=len(blocks), replace=True)
            sel = np.concatenate([blocks[p] for p in picks])
            sel = sel[sel < len(idx_arr)]
            bs.append(float(idx_arr[sel].mean()))
        sd = float(np.std(bs, ddof=1))
        z = (phat - 0.05) / sd if sd > 0 else np.nan
        p = p_upper(z) if np.isfinite(z) else np.nan
        holm[prod] = {"p_hat": phat, "sd_boot": sd, "z": z, "p_one_sided": p,
                      "n_stations": int(len(h2tab))}
        print(f"  H2 {prod}: outside {phat:.3f} (sd {sd:.3f}) z={z:.2f} p={p:.4f}")
    # Holm over 2 products
    ps = sorted([(v["p_one_sided"], k) for k, v in holm.items()])
    m = len(ps)
    running = 0.0
    for i, (p, k) in enumerate(ps):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        holm[k]["p_holm"] = float(running)
    excess_any = any(holm[k]["p_holm"] < 0.05 for k in ("cerra", "era5"))
    h2 = {"variant": variant, "guard_triggered": bool(any_zero_p10),
          "tests": {k: {kk: vv for kk, vv in v.items()} for k, v in holm.items()},
          "excess_significant": bool(excess_any),
          "verdict": ("excess confirmed" if excess_any else "refuted"),
          "n_stations_tested": int(len(h2tab))}
    results["H2"] = h2
    if sp_raw is not None:
        results["H2_variant_raw"] = {
            k: {p: sp_raw[k][p]["mean_dur"] for p in ("obs", "cerra", "era5")}
            for k in sids}
    h2tab.to_csv(os.path.join(OUT_DIR, "h2_spell_duration.csv"), index=False)

    # ---------------- H3 -----------------------------------------------------
    print("== H3 interannual ==")
    h3rows = []
    for sid in sids:
        g = stations[sid]["_g"]
        ann = g.groupby(g["date"].dt.year)[["ws_obs", "ws_cerra", "ws_era5"]].mean()
        cnt = g.groupby(g["date"].dt.year).size()
        ann = ann[cnt >= 274]
        if len(ann) < 15:
            continue
        ro = ann.ws_obs.quantile(0.10) / ann.ws_obs.quantile(0.5)
        rc = ann.ws_cerra.quantile(0.10) / ann.ws_cerra.quantile(0.5)
        re5 = ann.ws_era5.quantile(0.10) / ann.ws_era5.quantile(0.5)
        boots = {"obs": [], "cerra": [], "era5": []}
        arr = {c: ann[c].to_numpy() for c in ("ws_obs", "ws_cerra", "ws_era5")}
        n = len(arr["ws_obs"])
        bstart = list(range(0, n, 3))
        for _ in range(NPERM):
            picks = RNG.choice(len(bstart), size=len(bstart), replace=True)
            idx = np.concatenate([[bstart[p] + j for j in range(3)
                                   if bstart[p] + j < n] for p in picks])
            for c, key in (("ws_obs", "obs"), ("ws_cerra", "cerra"), ("ws_era5", "era5")):
                v = arr[c][idx]
                boots[key].append(v.quantile(0.10) / v.quantile(0.5)
                                  if isinstance(v, pd.Series) else
                                  np.quantile(v, 0.10) / np.quantile(v, 0.5))
        ci_o = [float(np.quantile(boots["obs"], 0.025)),
                float(np.quantile(boots["obs"], 0.975))]
        h3rows.append({"station_id": sid, "class": CLASS_OF[sid],
                       "ratio_obs": float(ro), "ratio_cerra": float(rc),
                       "ratio_era5": float(re5),
                       "ci_low": ci_o[0], "ci_high": ci_o[1],
                       "ci_width": ci_o[1] - ci_o[0],
                       "cerra_in": int(ci_o[0] <= rc <= ci_o[1]),
                       "era5_in": int(ci_o[0] <= re5 <= ci_o[1]),
                       "n_years": int(n)})
    h3tab = pd.DataFrame(h3rows)
    h3_detail = {}
    for prod in ("cerra", "era5"):
        p_in = float(h3tab[f"{prod}_in"].mean())
        idx_arr = 1.0 - h3tab[f"{prod}_in"].to_numpy().astype(float)
        bs = []
        for _ in range(NPERM):
            picks = RNG.choice(len(blocks), size=len(blocks), replace=True)
            sel = np.concatenate([blocks[p] for p in picks])
            sel = sel[sel < len(idx_arr)]
            bs.append(float(idx_arr[sel].mean()))
        sd = float(np.std(bs, ddof=1))
        z = (1 - p_in - 0.05) / sd if sd > 0 else np.nan
        p = p_upper(z) if np.isfinite(z) else np.nan
        h3_detail[prod] = {"p_inside": p_in, "p_outside": 1 - p_in, "sd_boot": sd,
                           "z": z, "p_one_sided_excess": p,
                           "median_ci_width": float(h3tab.ci_width.median())}
        print(f"  H3 {prod}: inside {p_in:.3f} excess p={p:.4f}")
    # effect sizes: paired ratio differences + station-block bootstrap CI
    eff = {}
    for prod in ("cerra", "era5"):
        d = (h3tab[f"ratio_{prod}"] - h3tab["ratio_obs"]).to_numpy()
        bs = []
        for _ in range(NPERM):
            picks = RNG.choice(len(blocks), size=len(blocks), replace=True)
            sel = np.concatenate([blocks[p] for p in picks])
            sel = sel[sel < len(d)]
            bs.append(float(np.median(d[sel])))
        eff[prod] = {"median_diff": float(np.median(d)),
                     "ci": [float(np.quantile(bs, 0.025)),
                            float(np.quantile(bs, 0.975))]}
    excess_h3 = any(h3_detail[p]["p_one_sided_excess"] < 0.05 for p in ("cerra", "era5"))
    results["H3"] = {"tests": h3_detail, "effect_sizes": eff,
                     "verdict": ("refuted" if excess_h3 else
                                 "not refuted (compatible with 95%)"),
                     "n_stations": int(len(h3tab))}
    h3tab.to_csv(os.path.join(OUT_DIR, "h3_annual_ratios.csv"), index=False)

    # ---------------- H4 -----------------------------------------------------
    print("== H4 Pettitt ==")

    def pettitt_perm(z, nperm=NPERM, rng=RNG):
        n = len(z)
        ranks = pd.Series(z).rank().to_numpy()
        srt = np.sort(z)
        sgn = np.where(z[:, None] > z[None, :], 1.0,
                       np.where(z[:, None] < z[None, :], -1.0, 0.0))
        U = np.cumsum(sgn.sum(axis=1))
        K = float(np.max(np.abs(U)))
        tstar = int(np.argmax(np.abs(U))) + 1
        # block permutation (L=3 consecutive years)
        bstart = list(range(0, n, 3))
        cnt = 0
        for _ in range(nperm):
            picks = rng.permutation(len(bstart))
            idx = np.concatenate([[bstart[p] + j for j in range(3)
                                   if bstart[p] + j < n] for p in picks])
            zp = z[idx]
            sgnp = np.where(zp[:, None] > zp[None, :], 1.0,
                            np.where(zp[:, None] < zp[None, :], -1.0, 0.0))
            Up = np.cumsum(sgnp.sum(axis=1))
            if np.max(np.abs(Up)) >= K:
                cnt += 1
        return K, tstar, (cnt + 1) / (nperm + 1)

    h4rows = []
    breaks = {"cerra": {}, "era5": {}, "ec": {}}
    for sid in sids:
        g = stations[sid]["_g"]
        ann = g.groupby(g["date"].dt.year)[["ws_obs", "ws_cerra", "ws_era5"]].mean()
        cnt = g.groupby(g["date"].dt.year).size()
        ann = ann[cnt >= 274]
        if len(ann) < 15:
            continue
        row = {"station_id": sid, "class": CLASS_OF[sid], "n_years": int(len(ann))}
        for prod, col in (("cerra", "ws_cerra"), ("era5", "ws_era5"), ("ec", None)):
            d = (ann[col] - ann.ws_obs).to_numpy() if col else \
                (ann.ws_era5 - ann.ws_cerra).to_numpy()
            z = (d - d.mean()) / (d.std(ddof=1) if d.std(ddof=1) > 0 else 1.0)
            K, t, p = pettitt_perm(z)
            row[f"pettitt_K_{prod}"] = K
            row[f"pettitt_t_{prod}"] = int(t)
            row[f"pettitt_p_{prod}"] = p
            breaks[prod][sid] = (int(t), p)
        h4rows.append(row)
    h4tab = pd.DataFrame(h4rows)
    years_of = h4tab.set_index("station_id")
    # BH-FDR per product
    sig_counts = {}
    for prod in ("cerra", "era5", "ec"):
        ps = h4tab[f"pettitt_p_{prod}"].to_numpy()
        order = np.argsort(ps)  # ascending
        m = len(ps)
        adj = np.empty(m)
        running = 1.0
        for i, oi in enumerate(order[::-1]):  # descending p, step-up BH
            rank = m - i
            running = min(running, ps[oi] * m / rank)
            adj[oi] = min(1.0, running)
        h4tab[f"pettitt_q_{prod}"] = adj
        sig_counts[prod] = int((adj < 0.05).sum())
    # condition A: excess vs chance 5% (spatial-adjusted variance)
    condA = {}
    for prod in ("cerra", "era5"):
        idx_arr = (h4tab[f"pettitt_q_{prod}"].to_numpy() < 0.05).astype(float)
        phat = float(idx_arr.mean())
        bs = []
        for _ in range(NPERM):
            picks = RNG.choice(len(blocks), size=len(blocks), replace=True)
            sel = np.concatenate([blocks[p] for p in picks])
            sel = sel[sel < len(idx_arr)]
            bs.append(float(idx_arr[sel].mean()))
        sd = float(np.std(bs, ddof=1))
        z = (phat - 0.05) / sd if sd > 0 else np.nan
        p = p_upper(z) if np.isfinite(z) else np.nan
        condA[prod] = {"p_hat": phat, "sd_boot": sd, "z": z, "p_one_sided": p,
                       "n_sig": sig_counts[prod]}
    # condition B: neighbour coincidence +/-2 y
    def coincidence(prod):
        having = [s for s in sids if s in breaks[prod] and breaks[prod][s][1] is not None
                  and h4tab.loc[h4tab.station_id == s, f"pettitt_q_{prod}"].iloc[0] < 0.05]
        years = {s: breaks[prod][s][0] for s in having}
        nbr_pairs = [(i, j) for i, j in combinations(range(len(having)), 2)
                     if haversine(dem.loc[having[i], "lat"], dem.loc[having[i], "lon"],
                                  dem.loc[having[j], "lat"], dem.loc[having[j], "lon"]) <= 200]
        C = sum(1 for i, j in nbr_pairs if abs(years[having[i]] - years[having[j]]) <= 2)
        permC = []
        ys = list(years.values())
        for _ in range(NPERM):
            perm = RNG.permutation(ys)
            permC.append(sum(1 for i, j in nbr_pairs
                             if abs(perm[i] - perm[j]) <= 2))
        p = float(np.mean(np.array(permC) >= C))
        return {"n_break_stations": len(having), "n_neighbour_pairs": len(nbr_pairs),
                "C_observed": int(C), "p_perm": p}
    condB = {prod: coincidence(prod) for prod in ("cerra", "era5")}
    h4 = {"sig_counts_fdr5": sig_counts, "conditionA": condA, "conditionB": condB,
          "verdict": ("confirmed" if any(condA[p]["p_one_sided"] < 0.05 or
                                         condB[p]["p_perm"] < 0.05
                                         for p in ("cerra", "era5")) else "refuted")}
    results["H4"] = h4
    h4tab.to_csv(os.path.join(OUT_DIR, "h4_pettitt.csv"), index=False)
    print(f"  H4 sig counts {sig_counts}; condA {condA}; condB {condB} -> {h4['verdict']}")

    # ---------------- per-station table export -------------------------------
    rows = []
    for sid in sids:
        r = stations[sid]
        d = dem.loc[sid]
        rows.append({
            "station_id": sid, "class": r["class"], "n_days": r["n_days"],
            "n_years": r["n_years"],
            "bias_cerra": r["bias_cerra"], "bias_era5": r["bias_era5"],
            "rmse_cerra": r["rmse_cerra"], "rmse_era5": r["rmse_era5"],
            "pearson_cerra": r["pearson_cerra"], "pearson_era5": r["pearson_era5"],
            "acf1_obs": r["acf1_obs"], "acf1_cerra": r["acf1_cerra"],
            "acf1_era5": r["acf1_era5"],
            "E_pers_cerra": r["E_pers_cerra"], "E_pers_era5": r["E_pers_era5"],
            "E_ar1": r["E_ar1"],
            "P90_obs": r["P90_obs"], "P50_obs": r["P50_obs"],
            "ratio_obs": r["ratio_obs"], "ratio_cerra": r["ratio_cerra"],
            "ratio_era5": r["ratio_era5"],
            "cv_obs": r["cv_obs"], "cv_cerra": r["cv_cerra"], "cv_era5": r["cv_era5"],
            "dz_cerra": d.dz_cerra, "dz_era5": d.dz_era5,
            "relief_cerra3x3": d.relief_cerra3x3,
            "relief_era5_corners": d.relief_era5_corners,
            "elev_station": d.elev_station,
        })
    st_tab = pd.DataFrame(rows)
    st_tab.to_csv(os.path.join(OUT_DIR, "station_metrics_confirmatory.csv"), index=False)

    # ---------------- O5 storage depth (exploratory) -------------------------
    print("== O5 storage depth (exploratory) ==")

    def power_curve(v, alpha):
        vh = v * (10.0 ** alpha)
        p = np.where(vh < 3.0, 0.0,
                     np.where(vh > 25.0, 0.0,
                              np.where(vh < 12.0, (vh ** 3 - 27.0) / (1728.0 - 27.0),
                                       1.0)))
        return p

    o5rows = []
    for alpha in (0.10, 0.14, 0.19, 0.25):
        for q in (0.6, 0.7, 0.8):
            med = {"obs": [], "cerra": [], "era5": []}
            for sid in sids:
                g = stations[sid]["_g"]
                g = g.sort_values("date").copy()
                g["year"] = g["date"].dt.year
                # completeness >=90% per year, identical masks
                cnt = g.groupby("year").size()
                ok_years = set(cnt[cnt >= 329].index)
                gg = g[g.year.isin(ok_years)].copy()
                if gg.year.nunique() < 5:
                    continue
                # linear interpolation over gaps for g (same mask all series)
                out = {}
                for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
                    p = power_curve(gg[col].to_numpy(), alpha)
                    mu = p.mean()
                    gt = p / mu if mu > 0 else p
                    # interpolate internal NaNs (there are none in valid rows)
                    out[prod] = gt
                day = gg["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
                yr = gg["year"].to_numpy()
                for prod in ("obs", "cerra", "era5"):
                    gtv = out[prod]
                    S_list = []
                    for y in sorted(set(yr)):
                        m = yr == y
                        yy, gy = day[m], gtv[m]
                        if len(yy) < 300:
                            continue
                        # linear interpolate gaps (<=5 d) in g
                        if np.diff(yy).max() > 1:
                            gd = np.interp(np.arange(yy[0], yy[-1] + 1), yy, gy)
                            dd = np.arange(yy[0], yy[-1] + 1)
                        else:
                            gd, dd = gy, yy
                        D = 0.0
                        S = 0.0
                        for gv in gd:
                            D = max(0.0, D + q - gv)
                            S = max(S, D)
                        S_list.append(S)
                    if S_list:
                        med[prod].append(float(np.median(S_list)))
            row = {"alpha": alpha, "q": q,
                   "S_obs_median_days": float(np.median(med["obs"])) if med["obs"] else np.nan,
                   "S_cerra_minus_obs": (float(np.median(med["cerra"]) - np.median(med["obs"]))
                                         if med["cerra"] and med["obs"] else np.nan),
                   "S_era5_minus_obs": (float(np.median(med["era5"]) - np.median(med["obs"]))
                                        if med["era5"] and med["obs"] else np.nan),
                   "S_cerra_over_obs": (float(np.median(med["cerra"]) / np.median(med["obs"]))
                                        if med["cerra"] and med["obs"] and np.median(med["obs"]) > 0 else np.nan),
                   "S_era5_over_obs": (float(np.median(med["era5"]) / np.median(med["obs"]))
                                       if med["era5"] and med["obs"] and np.median(med["obs"]) > 0 else np.nan),
                   "n_stations": len(med["obs"])}
            o5rows.append(row)
            print(f"  alpha={alpha} q={q}: obs {row['S_obs_median_days']:.2f} d "
                  f"CERRA {row['S_cerra_over_obs']:.3f}x ERA5 {row['S_era5_over_obs']:.3f}x")
    o5tab = pd.DataFrame(o5rows)
    o5tab.to_csv(os.path.join(OUT_DIR, "o5_storage_depth.csv"), index=False)
    results["O5"] = o5tab.to_dict(orient="records")

    # ---------------- sensitivities (exploratory) ----------------------------
    print("== sensitivities ==")
    sens = {}
    # N38 / N39
    for label, extra in (("N38", ["603400"]), ("N39", ["603400", "605220", "605570"])):
        sids2 = CONF + extra
        G_r, G_p = [], []
        out_c, out_e, tot = 0, 0, 0
        for sid in extra:
            g = df[(df.station_id == sid) & (df.valid == 1)].copy()
            if sid == "605220":
                g = g[g.ws_obs <= 40]  # documented artifact days (dev #6) removed
            g["year"] = g["date"].dt.year
            stations[sid] = {"_g": g, "_thr": {p: g.groupby("year")[c].quantile(0.10)
                                               for p, c in (("obs", "ws_obs"),
                                                            ("cerra", "ws_cerra"),
                                                            ("era5", "ws_era5"))}}
            ci_obs[sid] = obs_spell_ci(sid)
            doy = g["date"].dt.dayofyear.to_numpy()
            doy = np.where(doy == 366, 365, doy)
            day_int = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
            r_e = {}
            for col in ("ws_obs", "ws_cerra", "ws_era5"):
                x = g[col].to_numpy()
                clim = np.full(366, np.nan)
                for d in range(1, 366):
                    w = np.abs(((doy - d + 182) % 365) - 182) <= 15
                    if w.sum() >= 30:
                        clim[d] = x[w].mean()
                r_e[col] = x - clim[doy]
            ao = acf_series(r_e["ws_obs"], day_int)
            Ec = math.sqrt(np.nanmean([(acf_series(r_e["ws_cerra"], day_int)[k] - ao[k]) ** 2
                                       for k in range(1, 11)]))
            Ee = math.sqrt(np.nanmean([(acf_series(r_e["ws_era5"], day_int)[k] - ao[k]) ** 2
                                       for k in range(1, 11)]))
            G_r.append(float(np.sqrt(((g.ws_era5 - g.ws_obs) ** 2).mean()) -
                             np.sqrt(((g.ws_cerra - g.ws_obs) ** 2).mean())))
            G_p.append(Ee - Ec)
            # H2 outside for the extra stations
            thr = {p: g.groupby("year")[c].quantile(0.10)
                   for p, c in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5"))}
            if (thr["obs"] == 0).any() or any((thr[p] == 0).any() for p in ("cerra", "era5")):
                tmap = {p: {y: max(v, 0.5) for y, v in t.items()} for p, t in thr.items()}
            else:
                tmap = {p: {y: v for y, v in t.items()} for p, t in thr.items()}
            dint = day_int
            st_o = spell_stats(g.ws_obs.to_numpy() <= np.array([tmap["obs"][y] for y in g.year]), dint, 5)
            lohi = ci_obs.get(sid)
            if lohi and st_o["n_spells"] >= 1:
                sc = spell_stats(g.ws_cerra.to_numpy() <= np.array([tmap["cerra"][y] for y in g.year]), dint, 5)
                se = spell_stats(g.ws_era5.to_numpy() <= np.array([tmap["era5"][y] for y in g.year]), dint, 5)
                out_c += int(not (lohi[0] <= sc["mean_dur"] <= lohi[1]))
                out_e += int(not (lohi[0] <= se["mean_dur"] <= lohi[1]))
                tot += 1
        G_r = np.array(G_r + list(G_rmse))
        G_p = np.array(G_p + list(G_pers))
        rho = float(pd.Series(G_r).rank().corr(pd.Series(G_p).rank()))
        sens[label] = {"H1_rho": rho,
                       "H2_out_cerra": (out_c / tot if tot else np.nan),
                       "H2_out_era5": (out_e / tot if tot else np.nan),
                       "n": int(len(G_r))}
        print(f"  {label}: rho={rho:.3f} H2out c/e {sens[label]['H2_out_cerra']:.3f}/"
              f"{sens[label]['H2_out_era5']:.3f}")
    # ERA5 unmasked + interp-order + alt pairings + QC strict (RMSE/bias/rho deltas)
    sens_rows = []
    for sid in sids:
        g = df[(df.station_id == sid) & (df.valid == 1)]
        gq = df[(df.station_id == sid) & (df.valid_qc == 1)]
        base_c = stations[sid]
        row = {"station_id": sid,
               "rmse_era5_unmasked": float(np.sqrt(((g.era5_all24 - g.ws_obs) ** 2).mean())),
               "rmse_cerra_all8": float(np.sqrt(((g.cerra_all8 - g.ws_obs) ** 2).mean())),
               "rmse_era5_spd": float(np.sqrt(((g.ws_era5_spd - g.ws_obs) ** 2).mean())),
               "rmse_cerra_alt": float(np.sqrt(((g.ws_cerra_alt - g.ws_obs) ** 2).mean())) if g.ws_cerra_alt.notna().all() else np.nan,
               "rmse_era5_near": float(np.sqrt(((g.ws_era5_near - g.ws_obs) ** 2).mean())),
               "rmse_qc_strict_cerra": float(np.sqrt(((gq.ws_cerra - gq.ws_obs_qc) ** 2).mean())) if len(gq) > 500 else np.nan,
               "rmse_qc_strict_era5": float(np.sqrt(((gq.ws_era5 - gq.ws_obs_qc) ** 2).mean())) if len(gq) > 500 else np.nan,
               "rmse_cerra": base_c["rmse_cerra"], "rmse_era5": base_c["rmse_era5"]}
        sens_rows.append(row)
    senstab = pd.DataFrame(sens_rows)
    sens["median_deltas"] = {
        "era5_unmasked_minus_masked_rmse": float((senstab.rmse_era5_unmasked - senstab.rmse_era5).median()),
        "era5_spd_minus_comp_rmse": float((senstab.rmse_era5_spd - senstab.rmse_era5).median()),
        "cerra_alt_minus_near_rmse": float((senstab.rmse_cerra_alt - senstab.rmse_cerra).median()),
        "era5_near_minus_bilinear_rmse": float((senstab.rmse_era5_near - senstab.rmse_era5).median()),
        "qc_strict_minus_default_cerra": float((senstab.rmse_qc_strict_cerra - senstab.rmse_cerra).median()),
        "qc_strict_minus_default_era5": float((senstab.rmse_qc_strict_era5 - senstab.rmse_era5).median()),
        "cerra_all8_minus_masked_rmse": float((senstab.rmse_cerra_all8 - senstab.rmse_cerra).median()),
    }
    print(f"  median deltas: {json.dumps(sens['median_deltas'], indent=2)}")
    senstab.to_csv(os.path.join(OUT_DIR, "sensitivity_pairings.csv"), index=False)
    results["sensitivities"] = sens

    # spell sensitivity grid P{5,10,15} x k{3,5,10} + seasonal P10 (obs-threshold variant T2)
    grid_rows = []
    for sid in sids[:]:
        g = stations[sid]["_g"]
        dint = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        yr = g["year"].to_numpy()
        for pct in (5, 10, 15):
            q = pct / 100.0
            thr = {p: g.groupby("year")[c].quantile(q)
                   for p, c in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5"))}
            trig = any((thr[p] == 0).any() for p in thr)
            tmap = {}
            for p in thr:
                tmap[p] = np.array([max(thr[p].get(y, np.nan), 0.5) if trig
                                    else thr[p].get(y, np.nan) for y in yr])
            for k in (3, 5, 10):
                so = spell_stats(g.ws_obs.to_numpy() <= tmap["obs"], dint, k)
                if so["n_spells"] == 0:
                    continue
                # T2: obs threshold applied to models (distribution+structure error)
                sc2 = spell_stats(g.ws_cerra.to_numpy() <= tmap["obs"], dint, k)
                se2 = spell_stats(g.ws_era5.to_numpy() <= tmap["obs"], dint, k)
                grid_rows.append({"station_id": sid, "pct": pct, "k": k,
                                  "regime": "T1_own",
                                  "obs": so["mean_dur"],
                                  "cerra": spell_stats(g.ws_cerra.to_numpy() <= tmap["cerra"], dint, k)["mean_dur"],
                                  "era5": spell_stats(g.ws_era5.to_numpy() <= tmap["era5"], dint, k)["mean_dur"]})
                grid_rows.append({"station_id": sid, "pct": pct, "k": k,
                                  "regime": "T2_obsthreshold",
                                  "obs": so["mean_dur"], "cerra": sc2["mean_dur"],
                                  "era5": se2["mean_dur"]})
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(os.path.join(OUT_DIR, "spell_threshold_grid.csv"), index=False)
    results["spell_grid_median"] = {
        f"{r.regime}_P{r.pct}_k{r.k}": {"obs": float(r.obs), "cerra": float(r.cerra),
                                        "era5": float(r.era5)}
        for r in grid.itertuples() if np.isfinite(r.cerra) and np.isfinite(r.era5)}

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=float)
    # save station data for figures
    st_tab.to_csv(os.path.join(OUT_DIR, "station_metrics_confirmatory.csv"), index=False)
    np.savez(os.path.join(OUT_DIR, "fig_data.npz"),
             G_rmse=G_rmse, G_pers=G_pers, rho=h1["rho"], ci=np.array(h1["ci"]),
             sids=np.array(sids))
    print(f"\nDONE -> {OUT_JSON} and {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
