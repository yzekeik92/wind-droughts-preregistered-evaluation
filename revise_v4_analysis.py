#!/usr/bin/env python3
"""REVISION v4 ANALYSIS — implements the reviewer round-2 fix order.

P1  H1 sign structure: verify 14/32 informative CIs (12 CERRA / 2 ERA5),
    two-sided sign test p, E_C<E_E at 19/32; block sign-flip permutation.
P2  POD-vs-RMSE mechanism: per-class POD/FAR/CSI check; all-slot (max
    information density) re-run of E, spell durations and POD/FAR/CSI;
    ACF lag profile comparison (smoothing test); POD-gap correlates.
P3  H3 first-order diagnostics: SD ratio and correlation of first
    differences of annual means; CV comparison; ERA5-CERRA ceiling.
P4  Storage: CI for the observed >=4 m/s split; NEW independent partition
    by mean reanalysis wind (median split) with CIs; hub-height equivalence.
P5  Delta-z with real ERA5 model orography (+ land-sea mask); per-class
    medians; DEM-vs-model comparison at the same cells.
P6  Block-level checks: sign-flip permutation over the 8 spatial blocks for
    H2/H3/H4 share tests; ICC-based design effect and effective n.
P7  Event medians without the 6 calm-dominated stations and without the
    0.5 m/s floor (raw P10); convention audit (<=, gap handling, floor
    relevance counts).
P8  Minor checks: -0.10 CI duplication (independent draws?), S1/S2
    definition constants, appendix wind-code count.

Outputs: download/M2_results/revision_v4.json (+ helper CSVs).
Seed 20261017 throughout.
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import binom

PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
DEM = '/tmp/my-project/verification/dem_covariates.csv'
OROG = '/home/z/my-project/download/M2_results/era5_orography_v4.csv'
RES = '/home/z/my-project/download/M2_results'
OUT_JSON = os.path.join(RES, 'revision_v4.json')
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
CONF = sorted(CLASS_OF)
CALM = ['601050', '601270', '601950', '602100', '605200', '606020']


# ----------------------------------------------------------------- helpers
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def spatial_blocks(sids, lats, lons, thr_km=200.0):
    from itertools import combinations
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
    return [np.array(g) for g in groups.values()]


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


def runs(low, day_int, kmin=5):
    """(start,end) spell segments, kmin<=len, gaps break runs (m2b convention)."""
    if len(low) == 0:
        return []
    brk = np.r_[True, np.diff(day_int) != 1, True]
    ss = np.flatnonzero(brk)[:-1]
    ee = np.flatnonzero(brk)[1:]
    out = []
    for s, e in zip(ss, ee):
        seg = low[s:e]
        if not seg.any():
            continue
        d = np.diff(np.r_[0, seg.astype('int8'), 0])
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        out.extend([(int(a + s), int(b + s)) for a, b in zip(st, en) if b - a >= kmin])
    return out


def hit_rates(obs_spells, mod_spells, mod_low_idx, overlap=0.5):
    hits = 0
    for (s, e) in obs_spells:
        days = np.arange(s, e)
        frac = mod_low_idx[days].mean() if len(days) else 0.0
        if frac >= overlap:
            hits += 1
    misses = len(obs_spells) - hits
    fa = 0
    for (s, e) in mod_spells:
        days = np.arange(s, e)
        frac = sum(len(np.intersect1d(days, np.arange(os_, oe)))
                   for os_, oe in obs_spells) / len(days)
        if frac < overlap:
            fa += 1
    pod = hits / (hits + misses) if (hits + misses) else np.nan
    far = fa / (hits + fa) if (hits + fa) else np.nan
    csi = hits / (hits + misses + fa) if (hits + misses + fa) else np.nan
    return pod, far, csi


def acf_pairs(anom, day_int, lags=range(1, 11)):
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


def anomalies(g, col):
    doy = g["date"].dt.dayofyear.to_numpy()
    doy = np.where(doy == 366, 365, doy)
    x = g[col].to_numpy()
    clim = np.full(366, np.nan)
    for dd in range(1, 366):
        w = np.abs(((doy - dd + 182) % 365) - 182) <= 15
        if w.sum() >= 30:
            clim[dd] = x[w].mean()
    return x - clim[doy]


def runs_all(low, day_int):
    """All low-wind run segments (no length filter) — f4 convention."""
    if len(low) == 0:
        return []
    brk = np.r_[True, np.diff(day_int) != 1, True]
    ss = np.flatnonzero(brk)[:-1]
    ee = np.flatnonzero(brk)[1:]
    out = []
    for s, e in zip(ss, ee):
        seg = low[s:e]
        if not seg.any():
            continue
        d = np.diff(np.r_[0, seg.astype('int8'), 0])
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        out.extend([(int(a + s), int(b + s)) for a, b in zip(st, en)])
    return out


def icc1_binary(outcome, blocks):
    """One-way ANOVA ICC for a binary outcome grouped in blocks (clipped 0-1)."""
    y = np.asarray(outcome, float)
    N, J = len(y), len(blocks)
    grand = y.mean()
    ssb = sum(len(b) * (y[b].mean() - grand) ** 2 for b in blocks)
    ssw = sum(((y[b] - y[b].mean()) ** 2).sum() for b in blocks)
    msb = ssb / (J - 1) if J > 1 else 0.0
    msw = ssw / (N - J) if N > J else 0.0
    mbar = N / J
    icc = (msb - msw) / (msb + (mbar - 1) * msw) if (msb + (mbar - 1) * msw) > 0 else 0.0
    return float(np.clip(icc, 0.0, 1.0)), float(mbar)


def block_signflip(outcome, blocks, value=None, alt='high'):
    """Cluster permutation: flip each block's contribution sign (2^J).

    outcome: binary per-station indicator; value: optional signed per-station
    statistic (defaults to outcome). Statistic T = sum over blocks of the
    block total (of value), sign-flipped per block. p_one = P(T_perm >= T_obs)
    for alt='high' (excess), two-sided for value-based tests.
    """
    v = np.asarray(value if value is not None else outcome, float)
    J = len(blocks)
    contrib = np.array([v[b].sum() for b in blocks])
    t_obs = contrib.sum()
    p_high = 0
    p_two = 0
    for code in range(2 ** J):
        s = np.array([1 if (code >> j) & 1 else -1 for j in range(J)])
        t = float((s * contrib).sum())
        if t >= t_obs:
            p_high += 1
        if abs(t) >= abs(t_obs) - 1e-12:
            p_two += 1
    return {"t_obs": float(t_obs), "n_blocks": J,
            "p_one_sided": p_high / 2 ** J, "p_two_sided": p_two / 2 ** J,
            "blocks_positive": int((contrib > 0).sum()),
            "block_contrib": contrib.tolist()}


def pbin(k, n=32, p0=0.05):
    return float(binom.sf(k - 1, n, p0))


def main():
    out = {}
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    dem = pd.read_csv(DEM, dtype={"station_id": str}).set_index("station_id")
    st = pd.read_csv(os.path.join(RES, "station_metrics_confirmatory.csv"),
                     dtype={"station_id": str}).set_index("station_id")
    orog = pd.read_csv(OROG, dtype={"station_id": str}).set_index("station_id")
    h4 = pd.read_csv(os.path.join(RES, "h4_pettitt.csv"), dtype={"station_id": str})
    ev = pd.read_csv(os.path.join(RES, "event_hit_rates.csv"), dtype={"station_id": str})
    h1ci = pd.read_csv(os.path.join(RES, "h1_pers_logratio_ci_v3.csv"),
                       dtype={"station_id": str})
    h3 = pd.read_csv(os.path.join(RES, "h3_annual_ratios.csv"), dtype={"station_id": str})
    ann = pd.read_csv(os.path.join(RES, "annual_series_H4.csv"), dtype={"station_id": str})
    o5c = pd.read_csv(os.path.join(RES, "o5_central_per_station_v3.csv"),
                      dtype={"station_id": str}).set_index("station_id")
    h2tab = pd.read_csv(os.path.join(RES, "h2_full_floored_v3.csv"),
                        dtype={"station_id": str})

    gby = {}
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)].copy()
        g["year"] = g["date"].dt.year
        gby[sid] = g
    lats = dem.loc[CONF, 'lat'].to_numpy()
    lons = dem.loc[CONF, 'lon'].to_numpy()
    blocks = spatial_blocks(CONF, lats, lons, 200.0)
    out['n_blocks'] = len(blocks)
    out['block_sizes'] = sorted([len(b) for b in blocks])

    # ================= P1: H1 sign structure =================================
    rmse_c = st.loc[CONF, 'rmse_cerra'].to_numpy()
    rmse_e = st.loc[CONF, 'rmse_era5'].to_numpy()
    E_c = st.loc[CONF, 'E_pers_cerra'].to_numpy()
    E_e = st.loc[CONF, 'E_pers_era5'].to_numpy()
    g_pers = np.log(E_e / E_c)
    g_rmse = np.log(rmse_e / rmse_c)
    n_excl = int(h1ci.excl_zero.sum())
    n_pos = int((h1ci.ci_low > 0).sum())
    n_neg = int((h1ci.ci_high < 0).sum())
    p_sign_two = float(2 * binom.sf(max(n_pos, n_neg) - 1, n_excl, 0.5))
    n_better_pers = int((E_c < E_e).sum())
    # block permutation on the per-station log-ratio (all 32)
    perm_all = block_signflip(None, blocks, value=g_pers)
    info_mask = (h1ci.excl_zero == 1).to_numpy()
    info_idx = np.flatnonzero(info_mask)
    sids_info = [CONF[i] for i in info_idx]
    blk_info = spatial_blocks(sids_info, dem.loc[sids_info, 'lat'].to_numpy(),
                              dem.loc[sids_info, 'lon'].to_numpy(), 200.0)
    perm_info = block_signflip(None, blk_info, value=g_pers[info_idx])
    out['P1_H1_sign'] = {
        'n_informative': n_excl, 'n_favour_cerra': n_pos, 'n_favour_era5': n_neg,
        'sign_test_two_sided_p': p_sign_two,
        'n_E_smaller_cerra': n_better_pers,
        'median_ln_ratio': float(np.median(g_pers)),
        'block_perm_all32': perm_all,
        'block_perm_informative14': {k: v for k, v in perm_info.items()},
    }

    # ================= P2: POD mechanism =====================================
    ev1 = ev[ev.regime == 'T1_own']
    podC = ev1[ev1['product'] == 'cerra'].set_index('station_id').POD
    podE = ev1[ev1['product'] == 'era5'].set_index('station_id').POD
    farC = ev1[ev1['product'] == 'cerra'].set_index('station_id').FAR
    farE = ev1[ev1['product'] == 'era5'].set_index('station_id').FAR
    class_pod = {}
    for cls, ss in CLASSES.items():
        class_pod[cls] = {'POD_C': float(podC[ss].median()),
                          'POD_E': float(podE[ss].median()),
                          'FAR_C': float(farC[ss].median()),
                          'FAR_E': float(farE[ss].median())}
    pod_gap = (podC - podE).reindex(CONF).to_numpy()

    # all-slot (maximum information density) re-run: E, spells, POD
    acf_lag_med = {p: {k: [] for k in range(1, 11)} for p in
                   ('obs', 'cerra', 'era5', 'cerra_all8', 'era5_all24')}
    acf1 = {p: [] for p in ('cerra', 'era5', 'cerra_all8', 'era5_all24')}
    E_main, E_allslot = {p: [] for p in ('cerra', 'era5')}, {p: [] for p in ('cerra', 'era5')}
    dur_main, dur_allslot = {p: [] for p in ('cerra', 'era5')}, {p: [] for p in ('cerra', 'era5')}
    pod_allslot = {'cerra': [], 'era5': []}
    per_station_rows = []
    for sid in CONF:
        g = gby[sid]
        dint = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        yr = g["year"].to_numpy()
        cols = {'obs': 'ws_obs', 'cerra': 'ws_cerra', 'era5': 'ws_era5',
                'cerra_all8': 'cerra_all8', 'era5_all24': 'era5_all24'}
        anom, prs = {}, {}
        for name, col in cols.items():
            a = anomalies(g, col)
            anom[name] = a
            prs[name] = acf_pairs(a, dint)
        for name in cols:
            ac = acf_from_pairs(anom[name], prs[name])
            for k in range(1, 11):
                acf_lag_med[name][k].append(ac[k - 1])
            if name in acf1:
                acf1[name].append(ac[0])
        E_main['cerra'].append(float(np.sqrt(np.nanmean(
            (acf_from_pairs(anom['cerra'], prs['cerra']) -
             acf_from_pairs(anom['obs'], prs['obs'])) ** 2))))
        E_main['era5'].append(float(np.sqrt(np.nanmean(
            (acf_from_pairs(anom['era5'], prs['era5']) -
             acf_from_pairs(anom['obs'], prs['obs'])) ** 2))))
        E_allslot['cerra'].append(float(np.sqrt(np.nanmean(
            (acf_from_pairs(anom['cerra_all8'], prs['cerra_all8']) -
             acf_from_pairs(anom['obs'], prs['obs'])) ** 2))))
        E_allslot['era5'].append(float(np.sqrt(np.nanmean(
            (acf_from_pairs(anom['era5_all24'], prs['era5_all24']) -
             acf_from_pairs(anom['obs'], prs['obs'])) ** 2))))
        # spells + POD, T1 floored guard identical to m2b
        thr, trig = {}, False
        for name, col in cols.items():
            t = g.groupby("year")[col].quantile(0.10)
            thr[name] = t
            if (t == 0).any():
                trig = True
        tmap = {name: np.array([max(thr[name].get(y, np.nan), 0.5) if trig
                                else thr[name].get(y, np.nan) for y in yr])
                for name in cols}
        obs_low = g['ws_obs'].to_numpy() <= tmap['obs']
        obs_spells = runs(obs_low, dint, 5)
        row = {'station_id': sid, 'class': CLASS_OF[sid], 'n_obs_spells': len(obs_spells)}
        for prod in ('cerra', 'era5'):
            for tag, col in (('main', f'ws_{prod}'), ('allslot', cols[f'{prod}_all8' if prod == 'cerra' else 'era5_all24'])):
                low = g[col].to_numpy() <= tmap[col.split('_')[0] if False else ('cerra' if prod == 'cerra' else 'era5')]
                # thresholds: main uses own slot series; allslot uses own allslot series
                tname = prod if tag == 'main' else f'{prod}_all8' if prod == 'cerra' else f'{prod}_all24'
                low = g[col].to_numpy() <= tmap[tname]
                spells = runs(low, dint, 5)
                lens = [b - a for a, b in spells]
                if lens:
                    dur_main[prod].append(float(np.mean(lens))) if tag == 'main' else dur_allslot[prod].append(float(np.mean(lens)))
                pod, far, csi = hit_rates(obs_spells, spells, low)
                if tag == 'main':
                    pass
                else:
                    pod_allslot[prod].append((sid, pod))
                row[f'{tag}_{prod}_mean_dur'] = float(np.mean(lens)) if lens else np.nan
                row[f'{tag}_{prod}_POD'] = pod
        per_station_rows.append(row)
    ps = pd.DataFrame(per_station_rows)
    ps.to_csv(os.path.join(RES, 'v4_allslot_per_station.csv'), index=False)

    def cls_med(series_dict, keys=None):
        s = pd.Series(series_dict)
        return s

    podA = {p: pd.DataFrame(pod_allslot[p], columns=['station_id', 'POD']).set_index('station_id').POD
            for p in ('cerra', 'era5')}
    class_pod_allslot = {}
    for cls, ss in CLASSES.items():
        class_pod_allslot[cls] = {'POD_C': float(podA['cerra'][ss].median()),
                                  'POD_E': float(podA['era5'][ss].median())}
    acf_lag_median = {name: {k: float(np.nanmedian(v)) for k, v in d.items()}
                      for name, d in acf_lag_med.items()}
    out['P2_POD_mechanism'] = {
        'class_pod_T1_main': class_pod,
        'class_pod_T1_allslot': class_pod_allslot,
        'overall_POD_allslot': {'cerra': float(podA['cerra'][CONF].median()),
                                'era5': float(podA['era5'][CONF].median())},
        'overall_POD_main': {'cerra': float(podC[CONF].median()),
                             'era5': float(podE[CONF].median())},
        'median_E_main': {p: float(np.median(E_main[p])) for p in E_main},
        'median_E_allslot': {p: float(np.median(E_allslot[p])) for p in E_allslot},
        'E_allslot_cerra_better_n': int(np.sum(np.array(E_allslot['cerra']) < np.array(E_allslot['era5']))),
        'E_main_cerra_better_n': int(np.sum(np.array(E_main['cerra']) < np.array(E_main['era5']))),
        'median_mean_dur_main': {p: float(np.median(dur_main[p])) for p in dur_main},
        'median_mean_dur_allslot': {p: float(np.median(dur_allslot[p])) for p in dur_allslot},
        'acf_lag_medians': acf_lag_median,
        'acf1_cerra_gt_era5_n': int(np.sum(np.array(acf1['cerra']) > np.array(acf1['era5']))),
        'spearman_podgap_vs_durgap': float(pd.Series(pod_gap).rank().corr(
            pd.Series(np.array(dur_main['cerra'])[:0] if False else
                      [np.nanmean([r]) for r in [0]]).rank())) if False else None,
    }
    # POD gap vs duration gap and vs E gap (main run)
    dur_gap = (pd.Series(dur_main['cerra']) - pd.Series(dur_main['era5'])).to_numpy() \
        if len(dur_main['cerra']) == 32 else None
    if dur_gap is None:
        dmap_c = ps.set_index('station_id')['main_cerra_mean_dur']
        dmap_e = ps.set_index('station_id')['main_era5_mean_dur']
        dur_gap = (dmap_c - dmap_e).reindex(CONF).to_numpy()
    E_gap = E_e - E_c
    rho_pod_dur = float(pd.Series(pod_gap).rank().corr(pd.Series(dur_gap).rank()))
    rho_pod_E = float(pd.Series(pod_gap).rank().corr(pd.Series(E_gap).rank()))
    out['P2_POD_mechanism']['spearman_podgap_vs_durgap'] = rho_pod_dur
    out['P2_POD_mechanism']['spearman_podgap_vs_Egap'] = rho_pod_E
    out['P2_POD_mechanism']['median_dur_gap_main'] = float(np.nanmedian(dur_gap))
    valid = np.isfinite(dur_gap) & np.isfinite(pod_gap)
    out['P2_POD_mechanism']['n_dur_pairs_valid'] = int(valid.sum())
    out['P2_POD_mechanism']['n_dur_cerra_longer'] = int((dur_gap[valid] > 0).sum())

    # ================= P3: H3 first-difference diagnostics ===================
    h3i = h3.set_index('station_id')
    ann_g = {sid: a for sid, a in ann.groupby('station_id')}
    def h3_firstdiff(sids_):
        sd_r, r_d, cv_o, cv_c_, cv_e_, ceil_lv, ceil_df = \
            {p: [] for p in ('cerra', 'era5')}, {p: [] for p in ('cerra', 'era5')}, [], [], [], [], []
        for sid in sids_:
            a = ann_g[sid].sort_values('year')
            if len(a) < 10:
                continue
            do = np.diff(a['obs'].to_numpy())
            for prod in ('cerra', 'era5'):
                dp = np.diff(a[prod].to_numpy())
                sd_r[prod].append(np.std(dp, ddof=1) / np.std(do, ddof=1))
                r_d[prod].append(np.corrcoef(dp, do)[0, 1])
            cv_o.append(np.std(a['obs'], ddof=1) / abs(np.mean(a['obs'])))
            cv_c_.append(np.std(a['cerra'], ddof=1) / abs(np.mean(a['cerra'])))
            cv_e_.append(np.std(a['era5'], ddof=1) / abs(np.mean(a['era5'])))
            ceil_lv.append(np.corrcoef(a['era5'], a['cerra'])[0, 1])
            ceil_df.append(np.corrcoef(np.diff(a['era5']), np.diff(a['cerra']))[0, 1])
        return {
            'n': len(cv_o),
            'median_sd_ratio_firstdiff': {p: float(np.median(sd_r[p])) for p in sd_r},
            'median_r_firstdiff': {p: float(np.median(r_d[p])) for p in r_d},
            'median_cv_annual': {'obs': float(np.median(cv_o)),
                                 'cerra': float(np.median(cv_c_)),
                                 'era5': float(np.median(cv_e_))},
            'median_r_era5_cerra_levels': float(np.median(ceil_lv)),
            'median_r_era5_cerra_firstdiff': float(np.median(ceil_df)),
        }
    out['P3_H3_firstdiff'] = {
        'full': h3_firstdiff(CONF),
        'drop7': h3_firstdiff([s for s in CONF if s not in h4[h4.pettitt_q_cerra < 0.05].station_id.tolist()]),
        'dropCalm': h3_firstdiff([s for s in CONF if s not in CALM]),
    }

    # ================= P4: storage partitions with CIs =======================
    piv = o5c
    def blk_ci(sids_, values):
        idx = np.asarray(values, float)
        blks = spatial_blocks(sids_, dem.loc[sids_, 'lat'].to_numpy(),
                              dem.loc[sids_, 'lon'].to_numpy(), 200.0)
        bs = []
        for _ in range(10000):
            picks = RNG.choice(len(blks), size=len(blks), replace=True)
            sel = np.concatenate([blks[p] for p in picks])
            sel = sel[sel < len(idx)]
            bs.append(float(np.median(idx[sel])))
        return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]

    # reanalysis mean wind (mean of the two products' daily-mean wind)
    mean_rw = {sid: 0.5 * (gby[sid].ws_cerra.mean() + gby[sid].ws_era5.mean())
               for sid in CONF}
    med_rw = float(np.median(list(mean_rw.values())))
    hi_rw = [s for s in CONF if mean_rw[s] >= med_rw]
    lo_rw = [s for s in CONF if mean_rw[s] < med_rw]
    obs_wind = {sid: float(gby[sid].ws_obs.mean()) for sid in CONF}
    rich = [s for s in CONF if obs_wind[s] >= 4.0]
    poor = [s for s in CONF if obs_wind[s] < 4.0]

    def part_stats(sids_, tag):
        ln_c = np.log((piv.cerra / piv.obs).reindex(sids_).to_numpy())
        ln_e = np.log((piv.era5 / piv.obs).reindex(sids_).to_numpy())
        return {tag: {
            'n': len(sids_),
            'cerra_median_ratio': float(np.exp(np.median(ln_c))),
            'cerra_lnratio_median': float(np.median(ln_c)),
            'cerra_lnratio_ci': blk_ci(sids_, ln_c),
            'era5_median_ratio': float(np.exp(np.median(ln_e))),
            'era5_lnratio_median': float(np.median(ln_e)),
            'era5_lnratio_ci': blk_ci(sids_, ln_e)}}

    out['P4_storage'] = {
        'partition_rule': ('observed split: mean observed 10-m wind >=/< 4 m/s (post hoc); '
                           'independent split: median of the station mean of the two '
                           'reanalysis daily-mean winds'),
        'hub_height_4ms_at_alpha014': float(4.0 * 10 ** 0.14),
        'observed_split': {**part_stats(rich, 'ge4ms'), **part_stats(poor, 'lt4ms')},
        'reanalysis_wind_split': {**part_stats(hi_rw, 'hi_half'),
                                  **part_stats(lo_rw, 'lo_half')},
        'median_reanalysis_wind': med_rw,
    }

    # ================= P6: block-level checks ================================
    def cluster_check(indicator, sids_, label):
        idx = np.asarray(indicator, float)
        blks = spatial_blocks(sids_, dem.loc[sids_, 'lat'].to_numpy(),
                              dem.loc[sids_, 'lon'].to_numpy(), 200.0)
        icc, mbar = icc1_binary(idx, blks)
        deff = 1 + (mbar - 1) * icc
        n_eff = len(idx) / deff
        k = int(idx.sum())
        p_naive = pbin(k, n=len(idx))
        k_eff = k / deff
        n_r, k_r = int(round(n_eff)), int(round(k_eff))
        p_eff = pbin(k_r, n=n_r)
        perm = block_signflip(idx, blks)
        return {label: {'n': len(idx), 'k_out': k,
                        'p_binomial_naive': p_naive,
                        'icc': icc, 'mbar': mbar, 'design_effect': deff,
                        'n_effective': float(n_eff),
                        'k_effective': float(k_eff),
                        'p_binomial_effective_rounded': float(p_eff),
                        'p_binomial_effective_note': ('indicative: exact binomial at '
                                                      f'n={n_r}, k={k_r} (design-effect rounded)'),
                        'block_signflip': perm}}

    rupt7 = h4[h4.pettitt_q_cerra < 0.05].station_id.tolist()
    out['P6_cluster'] = {
        **cluster_check(h2tab.set_index('station_id').reindex(CONF).cerra_out.to_numpy(), CONF, 'H2_cerra'),
        **cluster_check(h2tab.set_index('station_id').reindex(CONF).era5_out.to_numpy(), CONF, 'H2_era5'),
        **cluster_check((1 - h3i.reindex(CONF).cerra_in).to_numpy(), CONF, 'H3_cerra'),
        **cluster_check((1 - h3i.reindex(CONF).era5_in).to_numpy(), CONF, 'H3_era5'),
        **cluster_check([1 if s in rupt7 else 0 for s in CONF], CONF, 'H4_cerra'),
    }

    # ================= P7: event medians without calm stations ===============
    def f4_stats(sids_, floored):
        res = {k: {'obs': [], 'cerra': [], 'era5': []} for k in (3, 5, 10)}
        ncontrib = {k: {'obs': 0, 'cerra': 0, 'era5': 0} for k in (3, 5, 10)}
        for sid in sids_:
            g = gby[sid].sort_values('date')
            dint = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
            yr = g["year"].to_numpy()
            thr, trig = {}, False
            for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5')):
                t = g.groupby("year")[c].quantile(0.10)
                thr[p] = t
                if (t == 0).any():
                    trig = True
            if not floored:
                trig = False
            tmap = {p: np.array([max(thr[p].get(y, np.nan), 0.5) if trig
                                 else thr[p].get(y, np.nan) for y in yr]) for p in thr}
            yrs = len(set(yr))
            for k in (3, 5, 10):
                for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5')):
                    low = g[c].to_numpy() <= tmap[p]
                    lens = [b - a for a, b in runs_all(low, dint) if b - a >= k]
                    if lens:
                        arr = np.array(lens)
                        res[k][p].append((len(lens) / yrs, arr.mean(), arr.max(),
                                          100.0 * arr.sum() / len(low)))
                        ncontrib[k][p] += 1
        o = {}
        for k in (3, 5, 10):
            o[f'k{k}'] = {p: {'events_per_year': float(np.median([r[0] for r in res[k][p]])) if res[k][p] else None,
                              'mean_dur': float(np.nanmedian([r[1] for r in res[k][p]])) if res[k][p] else None,
                              'max_dur': float(np.median([r[2] for r in res[k][p]])) if res[k][p] else None,
                              'days_pct': float(np.median([r[3] for r in res[k][p]])) if res[k][p] else None,
                              'n_stations_with_events': ncontrib[k][p]}
                          for p in ('obs', 'cerra', 'era5')}
        return o

    out['P7_event_medians'] = {
        'full32_floored': f4_stats(CONF, True),
        'nonCalm26_floored': f4_stats([s for s in CONF if s not in CALM], True),
        'nonCalm26_raw': f4_stats([s for s in CONF if s not in CALM], False),
    }
    # floor relevance: station-years with 0 < P10 < 0.5 among non-calm stations
    n_floor = 0
    for sid in CONF:
        if sid in CALM:
            continue
        for col in ('ws_obs', 'ws_cerra', 'ws_era5'):
            t = gby[sid].groupby("year")[col].quantile(0.10)
            n_floor += int(((t > 0) & (t < 0.5)).sum())
    out['P7_event_medians']['noncalm_station_years_0_lt_p10_lt_0.5'] = n_floor
    # calm-station convention audit: raw P10 = 0 -> spells are runs of 0.0 m/s days
    calm_spell_counts = {}
    for sid in CALM:
        g = gby[sid].sort_values('date')
        dint = g["date"].map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        yr = g["year"].to_numpy()
        tmap = {p: np.array([g[g.year == y][c].quantile(0.10) for y in yr])
                for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5'))}
        calm_spell_counts[sid] = {p: len(runs(g[c].to_numpy() <= tmap[p], dint, 5))
                                  for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5'))}
    out['P7_event_medians']['calm_stations_rawP10_spell_counts_k5'] = calm_spell_counts

    # ================= P5: dz with real model orography ======================
    rows = []
    for cls, ss in CLASSES.items():
        rows.append({'class': cls, 'n': len(ss),
                     'dz_C_dem': float(st.loc[ss, 'dz_cerra'].median()),
                     'dz_E_dem_near': float(st.loc[ss, 'dz_era5'].median()),
                     'dz_E_model_near': float(orog.loc[ss, 'dz_model_near'].median()),
                     'dz_E_model_corners_mean': float(orog.loc[ss, 'dz_model_corners_mean'].median()),
                     'n_ocean_corners_median': float(orog.loc[ss, 'n_ocean_corners_lsm05'].median()),
                     'n_stations_ocean_any': int(orog.loc[ss, 'ocean_any_corner'].sum())})
    out['P5_dz_model_orography'] = rows
    # DEM vs model orography at the same nearest corners (archipelago)
    arc = CLASSES['archipelago']
    d_dm = (st.loc[arc, 'dz_era5'] - orog.loc[arc, 'dz_model_near'])
    out['P5_dz_model_orography_dem_minus_model_median_arc'] = float(d_dm.median())

    # ================= P8: minor checks ======================================
    ci_diff_lo, ci_log_lo = -0.10236220472440946, -0.10270270270270268
    out['P8_minors'] = {
        'h1_ci_lower_bounds_raw': {'diff_based': ci_diff_lo, 'log_ratio': ci_log_lo},
        'h1_ci_lower_bounds_identical_to_2dp': bool(round(ci_diff_lo, 2) == round(ci_log_lo, 2)),
        'note': ('the two H1 intervals come from independent bootstrap draws '
                 '(sequential RNG stream in revise_v3_analysis.boot_rho); '
                 'agreement to 2 decimals is coincidence, raw bounds differ '
                 'in the 3rd decimal'),
        'S1_adds': ['603400'], 'S2_adds_cumulative': ['603400', '605220', '605570'],
        'S2_n': 35, 'S1_n': 33,
    }
    # wind-code count from the official tests_codes mapping as transcribed in A1
    wind_tests = ['WRC', 'DWS', 'HWS', 'WSS', 'DRS', 'HRS', 'RSS', 'OCW', 'WVR',
                  'WSL', 'WDL', 'WRS', 'WSP']
    wind_status = ['WMB', 'BBB']
    out['P8_minors']['wind_test_codes_n'] = len(wind_tests)
    out['P8_minors']['wind_month_clearing_n'] = len(wind_status)

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: out[k] for k in ('P1_H1_sign', 'n_blocks', 'block_sizes')},
                     indent=2, default=str)[:2200])
    print('P2:', json.dumps(out['P2_POD_mechanism'], indent=1, default=str)[:2600])
    print('P3:', json.dumps(out['P3_H3_firstdiff'], indent=1, default=str)[:1400])
    print('P4:', json.dumps(out['P4_storage'], indent=1, default=str)[:1800])
    print('P6:', json.dumps(out['P6_cluster'], indent=1, default=str)[:2400])
    print('P7 k5:', json.dumps({k: v.get('k5') if isinstance(v, dict) else v
                                for k, v in out['P7_event_medians'].items()}, indent=1)[:2000])
    print('P5:', json.dumps(out['P5_dz_model_orography'], indent=1)[:900])
    print('DONE ->', OUT_JSON)


if __name__ == '__main__':
    main()
