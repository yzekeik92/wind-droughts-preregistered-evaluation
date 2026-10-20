#!/usr/bin/env python3
"""REVISION v3 ANALYSIS — implements the Q1 pre-submission audit fix list.

R2 fixes:
  1. H1 re-expressed in relative (log-ratio) gains with per-station CIs;
     preregistered difference-based rho kept for continuity.
  2. Exact binomial p-values (one method, documented) for H2/H3/H4 excesses.
  3. H2 direction breakdown incl. no-spell stations; calm-station list;
     H2/H3/O5 sensitivities excluding the 7 rupture stations and calm stations.
  4. H3 with mean-bias-adjusted series + SD ratio + interannual correlation.
  5. O5 storage redefined on paired per-station log-ratios with block CIs;
     resource-poor subset; reconciliation of 15.3 vs 21.7 d.
  6. Counts for the station funnel; Table 1 additions (CSI_E, dz_E);
     spell-grid replacement numbers for the "within a few percent" claim.

Outputs: download/M2_results/revision_v3.json (+ helper CSVs).
Seed fixed at 20261017 throughout.
"""
import json
import math
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binom

PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
DEM = '/tmp/my-project/verification/dem_covariates.csv'
RES = '/home/z/my-project/download/M2_results'
OUT_JSON = os.path.join(RES, 'revision_v3.json')
RNG = np.random.default_rng(20261017)

CLASSES = {
    'archipelago': ['600010', '600050', '600150', '600200', '600250', '600300',
                    '600350', '600400'],
    'atlantic': ['600330', '600600', '602520', '614150'],
    'strait_med': ['601010', '601050', '603180', '604900', '605200', '605310'],
    'interior': ['601150', '601270', '601350', '601410', '601500', '601560',
                 '601950', '602100', '602300', '602650', '605710', '606020',
                 '606200', '606560'],
}
CLASS_OF = {s: c for c, ss in CLASSES.items() for s in ss}
CONF = sorted(CLASS_OF)  # 32 confirmatory
KMIN = 5


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


def runs_gek(low, day_int, kmin=5):
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


def acf_pairs(anom, day_int, lags=range(1, 11)):
    """Precompute lag-k pair index arrays once (valid-day calendar pairs)."""
    pos = {d: i for i, d in enumerate(day_int)}
    out = []
    for k in lags:
        d2 = day_int + k
        sel = np.isin(d2, list(pos.keys()))
        i1 = np.flatnonzero(sel)
        i2 = np.array([pos[d] for d in d2[sel]])
        ok = np.isfinite(anom[i1]) & np.isfinite(anom[i2])
        out.append((i1[ok], i2[ok]))
    return out


def acf_from_pairs(anom, pairs):
    res = np.empty(10)
    for li, (i1, i2) in enumerate(pairs):
        a, b = anom[i1], anom[i2]
        res[li] = np.corrcoef(a, b)[0, 1] if len(a) > 30 else np.nan
    return res


def e_pers(anom_p, pairs_p, anom_o, pairs_o):
    ap = acf_from_pairs(anom_p, pairs_p)
    ao = acf_from_pairs(anom_o, pairs_o)
    return float(np.sqrt(np.nanmean((ap - ao) ** 2)))


def holm(pvals):
    """Holm step-down; returns adjusted p-values in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, oi in enumerate(order):
        a = min(1.0, (m - i) * pvals[oi])
        running = max(running, a)
        adj[oi] = running
    return adj


def main():
    out = {}
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    dem = pd.read_csv(DEM, dtype={"station_id": str}).set_index("station_id")
    st = pd.read_csv(os.path.join(RES, "station_metrics_confirmatory.csv"),
                     dtype={"station_id": str}).set_index("station_id")
    h4 = pd.read_csv(os.path.join(RES, "h4_pettitt.csv"), dtype={"station_id": str})
    h2 = pd.read_csv(os.path.join(RES, "h2_spell_duration.csv"), dtype={"station_id": str})

    # ---------------- counts / funnel ---------------------------------------
    d32 = df[df.station_id.isin(CONF)]
    rows32, valid32 = len(d32), int(d32.valid.sum())
    audit = pd.read_csv('/tmp/my-project/verification/m1a_station_audit.csv',
                        dtype={'station_id': str})
    elig = audit[audit.eligible_v2 == True]
    in42 = elig[elig.in_user42 == True]
    beyond = elig[elig.in_user42 != True]
    fail42 = audit[(audit.in_user42 == True) & (audit.eligible_v2 != True)]
    reasons = fail42.exclude_reason.value_counts().to_dict()
    out['counts'] = {
        'hadisd_frame_600xxx': 306, 'in_geometric_frame': 55, 'eligible': 37,
        'pilots': 5, 'confirmatory': 32,
        'prescreen_supplied': 42, 'prescreen_eligible': int(len(in42)),
        'prescreen_failed': int(len(fail42)), 'fail_reasons': reasons,
        'eligible_beyond_prescreen': beyond.station_id.tolist(),
        'rows32': rows32, 'valid32': valid32,
        'max_rows32': 32 * 10958, 'valid_frac_of_max': valid32 / (32 * 10958),
        'rows35_incl_sens': int(len(df)),
        'mean_days_per_station': rows32 / 32,
    }

    # ---------------- per-station arrays -------------------------------------
    gby = {}
    calm_obs, calm_any = [], []
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)].copy()
        g["year"] = g["date"].dt.year
        gby[sid] = g
        for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
            t = g.groupby("year")[col].quantile(0.10)
            if (t == 0.0).any():
                calm_any.append(sid)
                if prod == "obs":
                    calm_obs.append(sid)
                break
    rupt7 = h4[h4.pettitt_q_cerra < 0.05].station_id.tolist()
    out['calm_stations'] = {
        'obs_p10_zero_any_year': calm_obs,
        'any_series_p10_zero_any_year': sorted(set(calm_any)),
        'n_station_years_obs_zero': int(sum(
            int((gby[s].groupby("year").ws_obs.quantile(0.10) == 0).sum()) for s in CONF)),
    }
    out['rupture_stations'] = rupt7
    no_spell_e = h2[h2.era5_mean_dur.isna()].station_id.tolist()
    no_spell_c = h2[h2.cerra_mean_dur.isna()].station_id.tolist()
    out['h2_no_spell_stations'] = {'cerra': no_spell_c, 'era5': no_spell_e}

    # ---------------- H1 v3: log-ratio gains ---------------------------------
    rmse_c = st.loc[CONF, 'rmse_cerra'].to_numpy()
    rmse_e = st.loc[CONF, 'rmse_era5'].to_numpy()
    E_c = st.loc[CONF, 'E_pers_cerra'].to_numpy()
    E_e = st.loc[CONF, 'E_pers_era5'].to_numpy()
    g_rmse = np.log(rmse_e / rmse_c)          # >0 CERRA better
    g_pers = np.log(E_e / E_c)                # >0 CERRA better
    lats = dem.loc[CONF, 'lat'].to_numpy()
    lons = dem.loc[CONF, 'lon'].to_numpy()
    blocks = spatial_blocks(CONF, lats, lons, 200.0)

    def boot_rho(x, y, nperm=10000):
        vals = []
        for _ in range(nperm):
            picks = RNG.choice(len(blocks), size=len(blocks), replace=True)
            idx = np.concatenate([blocks[p] for p in picks])
            if len(set(idx.tolist())) < 8:
                continue
            r = pd.Series(x[idx]).rank().corr(pd.Series(y[idx]).rank())
            if np.isfinite(r):
                vals.append(r)
        return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

    rho_log = float(pd.Series(g_rmse).rank().corr(pd.Series(g_pers).rank()))
    lo, hi = boot_rho(g_rmse, g_pers)
    # preregistered difference-based version, for continuity
    d_rmse = rmse_e - rmse_c
    d_pers = E_e - E_c
    rho_diff = float(pd.Series(d_rmse).rank().corr(pd.Series(d_pers).rank()))
    lod, hid = boot_rho(d_rmse, d_pers)
    out['H1_v3'] = {
        'median_ln_rmse_ratio': float(np.median(g_rmse)),
        'median_ln_pers_ratio': float(np.median(g_pers)),
        'rmse_rel_gain_pct': float(100 * (1 - np.median(rmse_c / rmse_e))),
        'pers_rel_gain_pct': float(100 * (1 - np.median(E_c / E_e))),
        'rho_logratio': rho_log, 'ci_logratio': [lo, hi],
        'rho_diff_prereg': rho_diff, 'ci_diff_prereg': [lod, hid],
        'n_cerra_better_rmse': int((g_rmse > 0).sum()),
        'n_cerra_better_pers': int((g_pers > 0).sum()),
    }

    # per-station CI of ln(E_E/E_C) via year-block bootstrap of anomalies
    rows_ci = []
    for sid in CONF:
        g = gby[sid]
        doy = g["date"].dt.dayofyear.to_numpy()
        doy = np.where(doy == 366, 365, doy)
        day_int = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        yr = g["year"].to_numpy()
        years = np.unique(yr)
        anomalies = {}
        pairs = {}
        for col in ("ws_obs", "ws_cerra", "ws_era5"):
            x = g[col].to_numpy()
            clim = np.full(366, np.nan)
            for dd in range(1, 366):
                w = np.abs(((doy - dd + 182) % 365) - 182) <= 15
                if w.sum() >= 30:
                    clim[dd] = x[w].mean()
            a = x - clim[doy]
            anomalies[col] = a
            pairs[col] = acf_pairs(a, day_int)
        pyr = yr  # year of index i1 (pairs chosen within same-year neighbourhoods)
        draws = []
        bs_blocks = [list(range(s, min(s + 3, len(years))))
                     for s in range(0, len(years), 3)]
        for _ in range(1000):
            picks = RNG.choice(len(bs_blocks), size=len(bs_blocks), replace=True)
            sel_years = years[np.concatenate([bs_blocks[p] for p in picks])]
            mask = np.isin(pyr, sel_years)
            ao = acf_from_pairs(anomalies["ws_obs"], pairs["ws_obs"])
            ec = e_pers(anomalies["ws_cerra"], pairs["ws_cerra"], anomalies["ws_obs"], pairs["ws_obs"])
            # mask pairs for products using selected years only
            def acf_sel(a, pr):
                res = np.empty(10)
                for li, (i1, i2) in enumerate(pr):
                    m = mask[i1] & mask[i2]
                    if m.sum() > 30:
                        res[li] = np.corrcoef(a[i1[m]], a[i2[m]])[0, 1]
                    else:
                        res[li] = np.nan
                return res
            ao_s = acf_sel(anomalies["ws_obs"], pairs["ws_obs"])
            ac = acf_sel(anomalies["ws_cerra"], pairs["ws_cerra"])
            ae = acf_sel(anomalies["ws_era5"], pairs["ws_era5"])
            Ec = float(np.sqrt(np.nanmean((ac - ao_s) ** 2)))
            Ee = float(np.sqrt(np.nanmean((ae - ao_s) ** 2)))
            if np.isfinite(Ec) and np.isfinite(Ee) and Ec > 0 and Ee > 0:
                draws.append(math.log(Ee / Ec))
        if draws:
            loq, hiq = np.quantile(draws, [0.025, 0.975])
            rows_ci.append({"station_id": sid, "class": CLASS_OF[sid],
                            "g_pers_log": float(math.log(E_e[CONF.index(sid)] / E_c[CONF.index(sid)])),
                            "ci_low": float(loq), "ci_high": float(hiq),
                            "se": float(np.std(draws, ddof=1)),
                            "excl_zero": int(not (loq <= 0 <= hiq))})
    ci_tab = pd.DataFrame(rows_ci)
    ci_tab.to_csv(os.path.join(RES, "h1_pers_logratio_ci_v3.csv"), index=False)
    out['H1_pers_station_CI'] = {
        'n_excl_zero': int(ci_tab.excl_zero.sum()),
        'n_negative': int((ci_tab.ci_high < 0).sum()),
        'n_positive': int((ci_tab.ci_low > 0).sum()),
        'median_se': float(ci_tab.se.median()),
        'sd_across_stations': float(ci_tab.g_pers_log.std(ddof=1)),
        'attenuation_note': ('median bootstrap SE of ln(E_E/E_C) vs its cross-station SD; '
                             'ratio ~1 means sampling noise dominates and attenuates rho'),
        'se_over_sd': float(ci_tab.se.median() / ci_tab.g_pers_log.std(ddof=1)),
    }

    # ---------------- exact binomial p-values --------------------------------
    def pbin(k, n=32, p0=0.05):
        return float(binom.sf(k - 1, n, p0))

    n_out = {"cerra": int(h2.cerra_out.sum()), "era5": int(h2.era5_out.sum())}
    n_in_h3 = {"cerra": 32 - 13, "era5": 32 - 16}
    h2_raw_p = [pbin(n_out['cerra']), pbin(n_out['era5'])]
    h2_holm = holm(h2_raw_p)
    h3_raw_p = [pbin(13), pbin(16)]
    h3_holm = holm(h3_raw_p)
    out['pvalues_exact_binomial'] = {
        'method': ('exact binomial one-sided P(X>=k), n=32, p0=0.05; Holm within '
                   'each hypothesis family (2 per-product tests)'),
        'H2_cerra': {'k': n_out['cerra'], 'p': h2_raw_p[0], 'p_holm': float(h2_holm[0])},
        'H2_era5': {'k': n_out['era5'], 'p': h2_raw_p[1], 'p_holm': float(h2_holm[1])},
        'H3_cerra': {'k': 13, 'p': h3_raw_p[0], 'p_holm': float(h3_holm[0])},
        'H3_era5': {'k': 16, 'p': h3_raw_p[1], 'p_holm': float(h3_holm[1])},
        'H4_cerra': {'k': 7, 'p': pbin(7)},
        'H4_era5': {'k': 3, 'p': pbin(3)},
        'H4_control': {'k': 0, 'p': pbin(0)},
    }

    # ---------------- H2 sensitivities ---------------------------------------
    def obs_ci_for(sid, floored):
        g = gby[sid]
        yrs = sorted(g.year.unique())
        tmap = {y: g[g.year == y].ws_obs.quantile(0.10) for y in yrs}
        vals = [(g[g.year == y].ws_obs.to_numpy(),
                 max(tmap[y], 0.5) if floored else tmap[y]) for y in yrs]
        blocksy = [list(range(s, min(s + 3, len(yrs))))
                   for s in range(0, len(yrs), 3)]
        boots = []
        for _ in range(2000):
            picks = RNG.choice(len(blocksy), size=len(blocksy), replace=True)
            chosen = [b for p in picks for b in blocksy[p]]
            low = np.concatenate([vals[b][0] <= vals[b][1] for b in chosen])
            dint = np.concatenate([np.arange(len(vals[b][0])) + (i * 10**6)
                                   for i, b in enumerate(chosen)])
            sst = spell_stats(low, dint, KMIN)
            if sst["n_spells"] >= 1:
                boots.append(sst["mean_dur"])
        if len(boots) >= 200:
            return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        return None

    def spell_out_table(sids_, floored):
        rows = []
        for sid in sids_:
            g = gby[sid]
            day_int = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
            yr = g["year"].to_numpy()
            stats = {}
            for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
                t_y = g.groupby("year")[col].quantile(0.10)
                t_arr = np.array([max(t_y.get(y, np.nan), 0.5) if floored
                                  else t_y.get(y, np.nan) for y in yr])
                stats[prod] = spell_stats(g[col].to_numpy() <= t_arr, day_int, KMIN)
            lohi = obs_ci_for(sid, floored) if sid not in cache_ci[floored] else cache_ci[floored][sid]
            cache_ci[floored][sid] = lohi
            if lohi is None or stats["obs"]["n_spells"] == 0:
                continue
            def outside(v):
                if not np.isfinite(v):
                    return 1, 'none'   # no qualifying spells at all
                if v > lohi[1]:
                    return 1, 'long'
                if v < lohi[0]:
                    return 1, 'short'
                return 0, 'in'
            oc, dc = outside(stats["cerra"]["mean_dur"])
            oe, de = outside(stats["era5"]["mean_dur"])
            rows.append({"station_id": sid, "class": CLASS_OF[sid],
                         "obs": stats["obs"]["mean_dur"], "cerra": stats["cerra"]["mean_dur"],
                         "era5": stats["era5"]["mean_dur"],
                         "cerra_out": oc, "era5_out": oe,
                         "cerra_dir": dc, "era5_dir": de})
        return pd.DataFrame(rows)

    cache_ci = {True: {}, False: {}}
    full_floored = spell_out_table(CONF, True)
    full_raw = spell_out_table(CONF, False)
    calm_set = sorted(set(calm_any))
    subset_no_rupt = [s for s in CONF if s not in rupt7]
    subset_no_calm = [s for s in CONF if s not in calm_set]
    subset_neither = [s for s in subset_no_rupt if s not in calm_set]

    def share_test(tab, prod, label):
        sids_ = tab.station_id.tolist()
        idx = tab[f"{prod}_out"].to_numpy().astype(float)
        lats_ = dem.loc[sids_, 'lat'].to_numpy()
        lons_ = dem.loc[sids_, 'lon'].to_numpy()
        blks = spatial_blocks(sids_, lats_, lons_, 200.0)
        phat = float(idx.mean())
        bs = []
        for _ in range(10000):
            picks = RNG.choice(len(blks), size=len(blks), replace=True)
            sel = np.concatenate([blks[p] for p in picks])
            sel = sel[sel < len(idx)]
            bs.append(float(idx[sel].mean()))
        sd = float(np.std(bs, ddof=1))
        z = (phat - 0.05) / sd if sd > 0 else np.nan
        p_boot = 0.5 * math.erfc(z / math.sqrt(2)) if np.isfinite(z) else np.nan
        p_exact = pbin(int(idx.sum()), n=len(idx))
        dirs = tab[f"{prod}_dir"].value_counts().to_dict() if prod in ('cerra', 'era5') else {}
        return {"label": label, "product": prod, "n": int(len(idx)),
                "k_out": int(idx.sum()), "share": phat,
                "p_binomial_exact": p_exact, "p_boot_one_sided": p_boot,
                "directions": dirs}

    h2_sens = []
    for tab, label, floored in ((full_floored, 'full_floored (confirmatory)', True),
                                (full_raw, 'full_raw', False),
                                (spell_out_table(subset_no_rupt, True), 'drop7_floored', True),
                                (spell_out_table(subset_no_calm, False), 'dropCalm_raw', False),
                                (spell_out_table(subset_neither, False), 'dropBoth_raw', False)):
        for prod in ("cerra", "era5"):
            h2_sens.append(share_test(tab, prod, label))
    h2s = pd.DataFrame(h2_sens)
    h2s.to_csv(os.path.join(RES, "h2_sensitivity_v3.csv"), index=False)
    out['H2_sensitivity'] = h2s.to_dict(orient='records')
    full_floored.to_csv(os.path.join(RES, "h2_full_floored_v3.csv"), index=False)

    # ---------------- H3 sensitivities ---------------------------------------
    h3 = pd.read_csv(os.path.join(RES, "h3_annual_ratios.csv"), dtype={"station_id": str})
    h3i = h3.set_index("station_id")

    def h3_share(sids_, tag):
        sub = h3i.loc[[s for s in sids_ if s in h3i.index]]
        res = {}
        for prod in ("cerra", "era5"):
            outs = int((sub[f"{prod}_in"] == 0).sum())
            idx = (1 - sub[f"{prod}_in"]).to_numpy().astype(float)
            lats_ = dem.loc[sub.index, 'lat'].to_numpy()
            lons_ = dem.loc[sub.index, 'lon'].to_numpy()
            blks = spatial_blocks(sub.index.tolist(), lats_, lons_, 200.0)
            bs = []
            for _ in range(10000):
                picks = RNG.choice(len(blks), size=len(blks), replace=True)
                sel = np.concatenate([blks[p] for p in picks])
                sel = sel[sel < len(idx)]
                bs.append(float(idx[sel].mean()))
            sd = float(np.std(bs, ddof=1))
            z = (idx.mean() - 0.05) / sd if sd > 0 else np.nan
            res[prod] = {"n": int(len(sub)), "k_out": outs,
                         "share": float(idx.mean()),
                         "p_binomial_exact": pbin(outs, n=len(sub)),
                         "median_excess": float((sub[f"ratio_{prod}"] - sub["ratio_obs"]).median()),
                         "high_side": int(((sub[f"ratio_{prod}"] > sub["ratio_obs"]) &
                                           (sub[f"{prod}_in"] == 0)).sum()),
                         "low_side": int(((sub[f"ratio_{prod}"] <= sub["ratio_obs"]) &
                                          (sub[f"{prod}_in"] == 0)).sum()),
                         "sd_boot_share": sd, "z": z}
        # annual-mean diagnostics (all products vs obs)
        sd_r = {"cerra": [], "era5": []}
        r_int = {"cerra": [], "era5": []}
        sd_r_adj = {"cerra": [], "era5": []}
        r_int_adj = {"cerra": [], "era5": []}
        ratio_adj = {"cerra": [], "era5": []}
        for sid in sids_:
            if sid not in h3i.index:
                continue
            g = gby[sid]
            ann = g.groupby("year")[["ws_obs", "ws_cerra", "ws_era5"]].mean()
            cnt = g.groupby("year").size()
            ann = ann[cnt >= 274]
            if len(ann) < 15:
                continue
            for prod, col in (("cerra", "ws_cerra"), ("era5", "ws_era5")):
                sd_r[prod].append(ann[col].std(ddof=1) / ann.ws_obs.std(ddof=1))
                r_int[prod].append(np.corrcoef(ann[col], ann.ws_obs)[0, 1])
                b = (ann[col] - ann.ws_obs).mean()
                adj = ann[col] - b
                sd_r_adj[prod].append(adj.std(ddof=1) / ann.ws_obs.std(ddof=1))
                r_int_adj[prod].append(np.corrcoef(adj, ann.ws_obs)[0, 1])
                q10a, q50a = np.quantile(adj, [0.10, 0.5])
                ratio_adj[prod].append(q10a / q50a)
        res['diagnostics'] = {
            f"{prod}": {"median_sd_ratio": float(np.median(sd_r[prod])),
                        "median_r_interannual": float(np.median(r_int[prod])),
                        "median_sd_ratio_badj": float(np.median(sd_r_adj[prod])),
                        "median_r_interannual_badj": float(np.median(r_int_adj[prod])),
                        "median_ratio_badj": float(np.median(ratio_adj[prod])),
                        "median_ratio_obs": float(np.median(
                            [h3i.loc[s, "ratio_obs"] for s in sids_ if s in h3i.index]))}
            for prod in ("cerra", "era5")}
        res['tag'] = tag
        return res

    h3_sens = [h3_share(CONF, 'full'),
               h3_share(subset_no_rupt, 'drop7'),
               h3_share(subset_no_calm, 'dropCalm'),
               h3_share(subset_neither, 'dropBoth')]
    out['H3_sensitivity'] = h3_sens
    with open(os.path.join(RES, 'h3_sensitivity_v3.json'), 'w') as f:
        json.dump(h3_sens, f, indent=2, default=float)

    # H3 low-side obs outliers (Fig 7 audit)
    out['H3_obs_outliers'] = {
        'stations_ratio_obs_below_0.75': h3i[h3i.ratio_obs < 0.75].index.tolist(),
        'stations_ratio_obs_below_0.6': h3i[h3i.ratio_obs < 0.6].index.tolist(),
        'min_ratio_obs': float(h3i.ratio_obs.min()),
        'are_rupture_flagged': [s for s in h3i[h3i.ratio_obs < 0.75].index if s in rupt7],
    }

    # ---------------- O5 storage v3 ------------------------------------------
    o5ps = pd.read_csv(os.path.join(RES, "o5_per_station.csv"), dtype={"station_id": str})
    mean_wind = {sid: float(gby[sid].ws_obs.mean()) for sid in CONF}
    res_rows = []
    for (a, q), s in o5ps.groupby(['alpha', 'q']):
        piv = s.pivot_table(index='station_id', columns='product', values='S_median').dropna()
        piv = piv.loc[[i for i in piv.index if i in CONF]]
        lnr_c = np.log(piv.cerra / piv.obs)
        lnr_e = np.log(piv.era5 / piv.obs)

        def blk_ci(v):
            idx = v.to_numpy()
            blks = spatial_blocks(piv.index.tolist(),
                                  dem.loc[piv.index, 'lat'].to_numpy(),
                                  dem.loc[piv.index, 'lon'].to_numpy(), 200.0)
            bs = []
            for _ in range(10000):
                picks = RNG.choice(len(blks), size=len(blks), replace=True)
                sel = np.concatenate([blks[p] for p in picks])
                sel = sel[sel < len(idx)]
                bs.append(float(np.median(idx[sel])))
            return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

        ci_c, ci_e = blk_ci(lnr_c), blk_ci(lnr_e)
        res_rows.append({
            'alpha': a, 'q': q, 'n': len(piv),
            'obs_median_S': float(piv.obs.median()),
            'cerra_pers_ratio_median': float((piv.cerra / piv.obs).median()),
            'era5_pers_ratio_median': float((piv.era5 / piv.obs).median()),
            'cerra_lnratio_median': float(lnr_c.median()), 'cerra_lnratio_ci': ci_c,
            'era5_lnratio_median': float(lnr_e.median()), 'era5_lnratio_ci': ci_e,
            'cerra_ratio_of_medians': float(piv.cerra.median() / piv.obs.median()),
            'era5_ratio_of_medians': float(piv.era5.median() / piv.obs.median()),
        })
    o5v3 = pd.DataFrame(res_rows)
    o5v3.to_csv(os.path.join(RES, 'o5_logratio_v3.csv'), index=False)
    central = o5v3[(o5v3.alpha == 0.14) & (o5v3.q == 0.7)].iloc[0]
    # resource-poor subset (mean 10m wind < 4 m/s) and dropBoth subset
    selc = o5ps[(o5ps.alpha == 0.14) & (o5ps.q == 0.7)]
    pivc = selc.pivot_table(index='station_id', columns='product', values='S_median').dropna()
    pivc = pivc.loc[[i for i in pivc.index if i in CONF]]
    poor = [s for s in pivc.index if mean_wind[s] < 4.0]
    rich = [s for s in pivc.index if mean_wind[s] >= 4.0]
    piv_nr = pivc.loc[[s for s in pivc.index if s not in rupt7 and s not in calm_set]]
    def _conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [float(x) for x in v]
        return float(v)

    out['O5_v3'] = {
        'central': {k: _conv(v) for k, v in central.items()},
        'obs_S_grid_range_days': [float(o5v3.obs_median_S.min()), float(o5v3.obs_median_S.max())],
        'mean_wind_10m': {s: round(mean_wind[s], 2) for s in CONF},
        'n_resource_poor_lt4ms': len(poor),
        'resource_poor_median_ratio': {
            'cerra': float((pivc.loc[poor].cerra / pivc.loc[poor].obs).median()),
            'era5': float((pivc.loc[poor].era5 / pivc.loc[poor].obs).median()),
            'obs_median_S': float(pivc.loc[poor].obs.median())},
        'resource_ok_median_ratio': {
            'cerra': float((pivc.loc[rich].cerra / pivc.loc[rich].obs).median()),
            'era5': float((pivc.loc[rich].era5 / pivc.loc[rich].obs).median()),
            'obs_median_S': float(pivc.loc[rich].obs.median())},
        'dropBoth_median_ratio': {
            'cerra': float((piv_nr.cerra / piv_nr.obs).median()),
            'era5': float((piv_nr.era5 / piv_nr.obs).median()),
            'n': int(len(piv_nr))},
    }
    pivc.assign(ratio_cerra=pivc.cerra / pivc.obs, ratio_era5=pivc.era5 / pivc.obs,
                mean_wind=pd.Series(mean_wind))\
        .to_csv(os.path.join(RES, 'o5_central_per_station_v3.csv'))

    # ---------------- spell-grid replacement numbers --------------------------
    grid = pd.read_csv(os.path.join(RES, 'spell_threshold_grid.csv'), dtype={'station_id': str})
    g5 = grid[(grid.regime == 'T1_own') & (grid.pct == 10)]
    out['spell_grid_T1_medians'] = {
        f'k{k}': {'obs': float(g5[g5.k == k].obs.median()),
                  'cerra': float(g5[g5.k == k].cerra.median()),
                  'era5': float(g5[g5.k == k].era5.median())}
        for k in (5, 10)}

    # ---------------- Table 1 additions: CSI_E, dz_E --------------------------
    ev = pd.read_csv(os.path.join(RES, 'event_hit_rates.csv'), dtype={'station_id': str})
    ev1 = ev[ev.regime == 'T1_own']
    csiE = ev1[ev1['product'] == 'era5'].groupby('station_id').CSI.median()
    csiC = ev1[ev1['product'] == 'cerra'].groupby('station_id').CSI.median()
    podE = ev1[ev1['product'] == 'era5'].groupby('station_id').POD.median()
    podC = ev1[ev1['product'] == 'cerra'].groupby('station_id').POD.median()
    farE = ev1[ev1['product'] == 'era5'].groupby('station_id').FAR.median()
    farC = ev1[ev1['product'] == 'cerra'].groupby('station_id').FAR.median()
    t1rows = []
    for cls, ss in CLASSES.items():
        ss = [s for s in ss if s in csiE.index]
        t1rows.append({'class': cls, 'n': len(ss),
                       'POD_C': float(podC[ss].median()), 'POD_E': float(podE[ss].median()),
                       'FAR_C': float(farC[ss].median()), 'FAR_E': float(farE[ss].median()),
                       'CSI_C': float(csiC[ss].median()), 'CSI_E': float(csiE[ss].median()),
                       'dz_C': float(st.loc[ss, 'dz_cerra'].median()),
                       'dz_E': float(st.loc[ss, 'dz_era5'].median()),
                       'relief': float(st.loc[ss, 'relief_cerra3x3'].median()),
                       'rmse_C': float(st.loc[ss, 'rmse_cerra'].median()),
                       'rmse_E': float(st.loc[ss, 'rmse_era5'].median()),
                       'bias_C': float(st.loc[ss, 'bias_cerra'].median()),
                       'bias_E': float(st.loc[ss, 'bias_era5'].median())})
    t1 = pd.DataFrame(t1rows)
    t1.to_csv(os.path.join(RES, 'class_table_v3.csv'), index=False)
    out['class_table'] = t1.to_dict(orient='records')

    # Table 3 (skill): observed ACF1 + E dimensionless check
    out['table_skill_additions'] = {
        'median_acf1_obs': float(st.loc[CONF, 'acf1_obs'].median()),
        'median_E_cerra': float(st.loc[CONF, 'E_pers_cerra'].median()),
        'median_E_era5': float(st.loc[CONF, 'E_pers_era5'].median()),
        'median_E_ar1': float(st.loc[CONF, 'E_ar1'].median()),
        'E_rel_gain_pct': float(100 * (1 - st.loc[CONF, 'E_pers_cerra'].median() /
                                       st.loc[CONF, 'E_pers_era5'].median())),
        'rmse_rel_gain_pct': float(100 * (1 - st.loc[CONF, 'rmse_cerra'].median() /
                                          st.loc[CONF, 'rmse_era5'].median())),
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ('counts', 'calm_stations', 'rupture_stations',
                                          'h2_no_spell_stations')}, indent=2, default=str))
    print('H1 v3:', json.dumps(out['H1_v3'], indent=2))
    print('binomial:', json.dumps(out['pvalues_exact_binomial'], indent=2))
    print('DONE ->', OUT_JSON)


if __name__ == '__main__':
    main()
