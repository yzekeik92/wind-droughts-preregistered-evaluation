#!/usr/bin/env python3
"""REVISION v5 ANALYSIS — implements the reviewer round-3 fix order (steps 2-3).

Step 2 (reviewer points 5, 2, 3, 9):
  A. Event-level metrics (POD/FAR/CSI, E, spell statistics) for EVERY
     pipeline sensitivity of Table 4 (tab:sens), not just dRMSE:
     ERA5 all-24h, CERRA all-8-slot, ERA5 speed-then-interpolate,
     CERRA runner-up cell, ERA5 nearest cell, strict-QC observations.
     Conventions identical to m2b_events.py / revise_v4_analysis.py
     (T1 own-percentile with the 0.5 m/s calm guard, k>=5, <=, 50% overlap).
     Validation: main-config medians must reproduce event_hit_rates.csv,
     station_metrics_confirmatory.csv (E) and v4_allslot values.
  B. Direct test of the calm/windy storage split (point 2): per-station
     regression of ln(S_prod/S_obs) on mean reanalysis wind at the central
     configuration, with a station bootstrap; counts of ratio>1 per half.
  C. Beta-binomial share tests calibrated on the design ICC (point 3):
     replaces the miscalibrated block sign-flip as the quoted
     cluster-aware conservative check; + cluster block bootstrap.
  D. Jensen bias: variance ratios of the daily-mean distributions of the
     three series (point 9) — the bias is regime-dependent through these.

Step 3 (point 10):
  E. Bilinear ERA5 model-orography dz and land-sea fraction at the exact
     bilinear footprint of the pairing (meta wy/wx); Canary bias vs
     land-sea-fraction regression (n=8, labelled hypothesis-generating).

Outputs: /home/z/my-project/download/M2_results/
  v5_table4_event_metrics.csv, era5_dz_bilinear_v5.csv,
  o5_wind_regression_v5.csv, revision_v5.json
Seed 20261017 throughout.
"""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
PARTS = '/tmp/my-project/M1b/parts'
NPZ = '/tmp/my-project/verification/era5_invariants.npz'
DEM = '/tmp/my-project/verification/dem_covariates.csv'
RES = '/home/z/my-project/download/M2_results'
RNG = np.random.default_rng(20261017)
CALM = ['601050', '601270', '601950', '602100', '605200', '606020']
CLASSES = {
    'archipelago': ['600010', '600050', '600150', '600200', '600250', '600300',
                    '600350', '600400'],
    'atlantic': ['600330', '600600', '602520', '614150'],
    'strait_med': ['601010', '601050', '603180', '604900', '605200', '605310'],
    'interior': ['601150', '601270', '601350', '601410', '601500', '601560',
                 '601950', '602100', '602300', '602650', '605710', '606020',
                 '606200', '606560'],
}


def runs(low, day_int, kmin=5):
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
        out.extend([(int(a), int(b)) for a, b in zip(st + s, en + s) if b - a >= kmin])
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


def anomalies(x, doy):
    clim = np.full(366, np.nan)
    for d in range(1, 366):
        w = np.abs(((doy - d + 182) % 365) - 182) <= 15
        if w.sum() >= 30:
            clim[d] = x[w].mean()
    return x - clim[doy]


def acf_series(anom, day_int, lags=range(1, 11)):
    pos = {d: i for i, d in enumerate(day_int)}
    keys = np.fromiter(pos.keys(), dtype='int64', count=len(pos))
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
    return np.array([out[k] for k in range(1, 11)])


def beta_binomial_tail(k, n, pi, rho):
    """P(X >= k) under Beta-binomial(n, alpha, beta) with mean pi, ICC rho."""
    if rho <= 0:
        return float(stats.binom.sf(k - 1, n, pi))
    ab = (1.0 - rho) / rho
    a, b = pi * ab, (1.0 - pi) * ab
    lg = math.lgamma(n + 1)
    logB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    tail = 0.0
    for x in range(k, n + 1):
        lp = (lg - math.lgamma(x + 1) - math.lgamma(n - x + 1)
              + math.lgamma(x + a) + math.lgamma(n - x + b) - math.lgamma(n + a + b)
              - logB)
        tail += math.exp(lp)
    return float(min(tail, 1.0))


def cluster_boot_p(block_counts, block_sizes, n_boot=10000, rng=None):
    """Cluster bootstrap p for H0 share<=0.05: fraction of resampled shares<=0.05."""
    rng = rng or np.random.default_rng(20261017)
    counts = np.asarray(block_counts, float)
    sizes = np.asarray(block_sizes, float)
    n = sizes.sum()
    draws = rng.choice(len(counts), size=(n_boot, len(counts)), replace=True)
    kboot = (counts[draws]).sum(axis=1)
    return float((kboot / n <= 0.05).mean())


def main():
    os.makedirs(RES, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    df['date'] = pd.to_datetime(df['date'])
    draw = json.load(open('/tmp/my-project/verification/pilot_draw.json'))
    CONF = draw['confirmatory_stations']
    out = {}

    gby = {sid: df[(df.station_id == sid) & (df.valid == 1)].sort_values('date').copy()
           for sid in CONF}
    for sid in CONF:
        gby[sid]['year'] = gby[sid]['date'].dt.year

    # ================= A. event metrics for every pipeline variant ===========
    VARIANTS = [('main', 'cerra', 'ws_cerra'), ('main', 'era5', 'ws_era5'),
                ('cerra_all8', 'cerra', 'cerra_all8'),
                ('era5_all24', 'era5', 'era5_all24'),
                ('cerra_alt', 'cerra', 'ws_cerra_alt'),
                ('era5_near', 'era5', 'ws_era5_near'),
                ('era5_spd', 'era5', 'ws_era5_spd')]
    rows = []
    qc_rows = []
    rmse_rows = []
    for sid in CONF:
        g = gby[sid]
        dint = g['date'].map(pd.Timestamp.toordinal).to_numpy().astype('int64')
        yr = g['year'].to_numpy()
        doy = g['date'].dt.dayofyear.to_numpy()
        doy = np.where(doy == 366, 365, doy)
        yrs = len(set(yr))

        # --- RMSE of every variant (validation vs tab:sens) ---
        r_obs = g['ws_obs'].to_numpy()
        rms = {c: float(np.sqrt(((g[c].to_numpy() - r_obs) ** 2).mean()))
               for _, _, c in VARIANTS}
        rmse_rows.append({'station_id': sid, **rms,
                          'rmse_cerra': rms['ws_cerra'], 'rmse_era5': rms['ws_era5']})

        # --- anomalies + ACF ---
        anom = {'obs': anomalies(r_obs, doy)}
        acf = {'obs': acf_series(anom['obs'], dint)}
        for _, _, c in VARIANTS:
            if c not in anom:
                anom[c] = anomalies(g[c].to_numpy(), doy)
                acf[c] = acf_series(anom[c], dint)

        # --- thresholds: T1 own-percentile, calm guard over all series ---
        thr = {}
        trig = False
        for c in ['ws_obs'] + [c for _, _, c in VARIANTS]:
            t = g.groupby('year')[c].quantile(0.10)
            thr[c] = t
            if (t == 0).any():
                trig = True
        tmap = {c: np.array([max(thr[c].get(y, np.nan), 0.5) if trig
                             else thr[c].get(y, np.nan) for y in yr]) for c in thr}
        obs_spells = runs(r_obs <= tmap['ws_obs'], dint, 5)

        for tag, prod, col in VARIANTS:
            low = g[col].to_numpy() <= tmap[col]
            spells = runs(low, dint, 5)
            pod, far, csi = hit_rates(obs_spells, spells, low)
            lens = np.array([b - a for a, b in spells])
            spells10 = runs(low, dint, 10)
            m10 = max([b - a for a, b in spells10], default=0)
            rows.append({
                'station_id': sid, 'variant': tag, 'product': prod,
                'E': float(np.sqrt(np.nanmean((acf[col] - acf['obs']) ** 2))),
                'POD': pod, 'FAR': far, 'CSI': csi,
                'events_per_year': len(spells) / yrs,
                'mean_dur': float(lens.mean()) if len(lens) else np.nan,
                'days_pct': float(100.0 * lens.sum() / len(low)) if len(lens) else 0.0,
                'max_dur_k10': float(m10),
                'rmse': rms[col],
            })
            if tag == 'main':
                # observed anchor
                olens = np.array([b - a for a, b in obs_spells])
                ospells10 = runs(r_obs <= tmap['ws_obs'], dint, 10)
                rows.append({
                    'station_id': sid, 'variant': 'obs_anchor', 'product': 'obs',
                    'E': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan,
                    'events_per_year': len(obs_spells) / yrs,
                    'mean_dur': float(olens.mean()) if len(olens) else np.nan,
                    'days_pct': float(100.0 * olens.sum() / len(r_obs)) if len(olens) else 0.0,
                    'max_dur_k10': float(max([b - a for a, b in ospells10], default=0)),
                    'rmse': np.nan,
                })

        # --- strict-QC variant (different day set) ---
        gq = df[(df.station_id == sid) & (df.valid_qc == 1)].sort_values('date').copy()
        if len(gq) > 300:
            gq['year'] = gq['date'].dt.year
            dq = gq['date'].map(pd.Timestamp.toordinal).to_numpy().astype('int64')
            yq = gq['year'].to_numpy()
            doyq = gq['date'].dt.dayofyear.to_numpy()
            doyq = np.where(doyq == 366, 365, doyq)
            yrsq = len(set(yq))
            xo = gq['ws_obs_qc'].to_numpy()
            rmse_q = {p: float(np.sqrt(((gq[f'ws_{p}'].to_numpy() - xo) ** 2).mean()))
                      for p in ('cerra', 'era5')}
            anomq = {'obs': anomalies(xo, doyq)}
            acfq = {'obs': acf_series(anomq['obs'], dq)}
            for p in ('cerra', 'era5'):
                anomq[p] = anomalies(gq[f'ws_{p}'].to_numpy(), doyq)
                acfq[p] = acf_series(anomq[p], dq)
            thrq, trigq = {}, False
            for c in ('ws_obs_qc', 'ws_cerra', 'ws_era5'):
                t = gq.groupby('year')[c].quantile(0.10)
                thrq[c] = t
                if (t == 0).any():
                    trigq = True
            tmapq = {c: np.array([max(thrq[c].get(y, np.nan), 0.5) if trigq
                                  else thrq[c].get(y, np.nan) for y in yq]) for c in thrq}
            obs_spq = runs(xo <= tmapq['ws_obs_qc'], dq, 5)
            for p in ('cerra', 'era5'):
                low = gq[f'ws_{p}'].to_numpy() <= tmapq[f'ws_{p}']
                spells = runs(low, dq, 5)
                pod, far, csi = hit_rates(obs_spq, spells, low)
                lens = np.array([b - a for a, b in spells])
                qc_rows.append({
                    'station_id': sid, 'variant': 'qc_strict', 'product': p,
                    'E': float(np.sqrt(np.nanmean((acfq[p] - acfq['obs']) ** 2))),
                    'POD': pod, 'FAR': far, 'CSI': csi,
                    'events_per_year': len(spells) / yrsq,
                    'mean_dur': float(lens.mean()) if len(lens) else np.nan,
                    'days_pct': float(100.0 * lens.sum() / len(low)) if len(lens) else 0.0,
                    'max_dur_k10': float(max([b - a for a, b in runs(low, dq, 10)], default=0)),
                    'rmse': rmse_q[p],
                })
            qolens = np.array([b - a for a, b in obs_spq])
            qc_rows.append({
                'station_id': sid, 'variant': 'qc_strict', 'product': 'obs',
                'E': np.nan, 'POD': np.nan, 'FAR': np.nan, 'CSI': np.nan,
                'events_per_year': len(obs_spq) / yrsq,
                'mean_dur': float(qolens.mean()) if len(qolens) else np.nan,
                'days_pct': float(100.0 * qolens.sum() / len(xo)) if len(qolens) else 0.0,
                'max_dur_k10': float(max([b - a for a, b in runs(xo <= tmapq['ws_obs_qc'], dq, 10)], default=0)),
                'rmse': np.nan,
            })

    ev = pd.DataFrame(rows)
    evq = pd.DataFrame(qc_rows)
    ev_all = pd.concat([ev, evq], ignore_index=True)
    ev_all.to_csv(os.path.join(RES, 'v5_table4_per_station.csv'), index=False)

    rmdf = pd.DataFrame(rmse_rows)
    v4sens = pd.read_csv('/tmp/my-project/download/M2_results/sensitivity_pairings.csv',
                         dtype={'station_id': str}).set_index('station_id')
    d_rmse = {
        'era5_all24': float((rmdf.era5_all24 - rmdf.rmse_era5).median()),
        'cerra_all8': float((rmdf.cerra_all8 - rmdf.rmse_cerra).median()),
        'era5_spd': float((rmdf.ws_era5_spd - rmdf.rmse_era5).median()),
        'cerra_alt': float((rmdf.ws_cerra_alt - rmdf.rmse_cerra).median()),
        'era5_near': float((rmdf.ws_era5_near - rmdf.rmse_era5).median()),
        'qc_strict_c': float((v4sens.rmse_qc_strict_cerra - v4sens.rmse_cerra).median()),
        'qc_strict_e': float((v4sens.rmse_qc_strict_era5 - v4sens.rmse_era5).median()),
    }

    def med(variant, prod, col):
        s = ev_all[(ev_all['variant'] == variant) &
                   (ev_all['product'] == prod)][col]
        return float(s.median()) if len(s) else np.nan

    order = [('main', 'cerra', 'C'), ('main', 'era5', 'E'),
             ('cerra_all8', 'cerra', 'C'), ('era5_all24', 'era5', 'E'),
             ('cerra_alt', 'cerra', 'C'), ('era5_near', 'era5', 'E'),
             ('era5_spd', 'era5', 'E'), ('qc_strict', 'cerra', 'C'),
             ('qc_strict', 'era5', 'E'), ('obs_anchor', 'obs', 'obs')]
    tab = []
    for variant, prod, lab in order:
        tab.append({
            'variant': variant, 'product': lab,
            'E': med(variant, prod, 'E'),
            'POD': med(variant, prod, 'POD'), 'FAR': med(variant, prod, 'FAR'),
            'CSI': med(variant, prod, 'CSI'),
            'events_per_year': med(variant, prod, 'events_per_year'),
            'mean_dur': med(variant, prod, 'mean_dur'),
            'days_pct': med(variant, prod, 'days_pct'),
            'max_dur_k10': med(variant, prod, 'max_dur_k10'),
            'rmse_median': float(ev_all[(ev_all['variant'] == variant) &
                                        (ev_all['product'] == prod)]['rmse'].median())
            if prod != 'obs' else np.nan,
        })
    tabdf = pd.DataFrame(tab)
    tabdf.to_csv(os.path.join(RES, 'v5_table4_event_metrics.csv'), index=False)

    # validation against v4 artefacts
    val = {
        'main_POD': ([round(v, 3) for v in tabdf[(tabdf.variant == 'main')].POD.tolist()],
                     [0.80, 0.62]),
        'main_CSI': ([round(v, 3) for v in tabdf[(tabdf.variant == 'main')].CSI.tolist()],
                     [0.48, 0.33]),
        'main_E': ([round(v, 4) for v in tabdf[(tabdf.variant == 'main')].E.tolist()],
                   [0.0396, 0.0486]),
        'allslot_POD': ([round(v, 3) for v in tabdf[tabdf.variant.isin(['cerra_all8', 'era5_all24'])].POD.tolist()],
                        [0.698, 0.581]),
        'allslot_dur': ([round(v, 3) for v in tabdf[tabdf.variant.isin(['cerra_all8', 'era5_all24'])].mean_dur.tolist()],
                        [5.944, 5.724]),
        'd_rmse': {k: round(v, 3) for k, v in d_rmse.items()},
    }
    out['A_table4'] = {'summary': tab, 'd_rmse': d_rmse, 'validation': val}
    print('== A. Table 4 event metrics ==')
    print(tabdf.to_string(index=False))
    print('validation:', json.dumps(val, indent=1)[:600])

    # ================= B. storage log-ratio vs mean reanalysis wind ==========
    o5 = pd.read_csv('/tmp/my-project/download/M2_results/o5_central_per_station_v3.csv',
                     dtype={'station_id': str})
    st = pd.read_csv(os.path.join(RES, 'station_metrics_confirmatory.csv'),
                     dtype={'station_id': str})
    o5 = o5.merge(st[['station_id', 'class']], on='station_id', how='left')
    # reviewer point 2: regress on the mean REANALYSIS wind (not the observed
    # one, which carries the selection effect). Definition identical to the v4
    # independent partition: mean over valid days of the two products' daily
    # mean winds.
    mw = {}
    for sid in CONF:
        g = gby[sid]
        mw[sid] = float(((g['ws_cerra'] + g['ws_era5']) / 2.0).mean())
    o5['mean_wind_reanalysis'] = o5.station_id.map(mw)
    o5['mean_wind_observed'] = o5['mean_wind']
    o5['mean_wind'] = o5['mean_wind_reanalysis']
    o5['ln_e'] = np.log(o5.ratio_era5)
    o5['ln_c'] = np.log(o5.ratio_cerra)
    o5['ln_gap'] = o5.ln_e - o5.ln_c
    medw = float(np.median(o5.mean_wind))
    assert abs(medw - 3.341385527069739) < 1e-3, medw

    def ols_boot(x, y, n_boot=10000):
        X = np.column_stack([np.ones_like(x), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
        bs = np.array([np.linalg.lstsq(X[i], y[i], rcond=None)[0][1] for i in idx])
        return float(beta[1]), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    reg = {}
    for name, ycol in (('era5', 'ln_e'), ('cerra', 'ln_c'), ('gap_era5_minus_cerra', 'ln_gap')):
        x = o5.mean_wind.to_numpy()
        y = o5[ycol].to_numpy()
        slope, lo, hi = ols_boot(x, y)
        r_p = float(stats.pearsonr(x, y)[0])
        r_p_p = float(stats.pearsonr(x, y)[1])
        rho_s, rho_p = stats.spearmanr(x, y)
        reg[name] = {'slope_per_ms': slope, 'ci95': [lo, hi],
                     'pearson_r': r_p, 'pearson_p': r_p_p,
                     'spearman_rho': float(rho_s), 'spearman_p': float(rho_p), 'n': 32}
    counts = {}
    for prod, rcol in (('era5', 'ratio_era5'), ('cerra', 'ratio_cerra')):
        hi = o5[o5.mean_wind >= medw]
        lo = o5[o5.mean_wind < medw]
        counts[prod] = {
            'windy16_gt1': int((hi[rcol] > 1).sum()), 'windy16_n': len(hi),
            'calm16_gt1': int((lo[rcol] > 1).sum()), 'calm16_n': len(lo),
        }
    out['B_wind_regression'] = {'median_wind': medw, 'regressions': reg, 'counts_gt1': counts}
    o5[['station_id', 'class', 'mean_wind', 'mean_wind_observed',
        'ratio_cerra', 'ratio_era5', 'ln_c', 'ln_e', 'ln_gap']].to_csv(
        os.path.join(RES, 'o5_wind_regression_v5.csv'), index=False)
    print('\n== B. wind regression ==')
    print(json.dumps({'reg': reg, 'counts': counts}, indent=1))

    # ================= C. beta-binomial share tests ==========================
    v4 = json.load(open('/tmp/my-project/download/M2_results/revision_v4.json'))
    blocks = v4['block_sizes']
    tests = {}
    for key, icc in (('H2_cerra', v4['P6_cluster']['H2_cerra']),
                     ('H2_era5', v4['P6_cluster']['H2_era5']),
                     ('H3_cerra', v4['P6_cluster']['H3_cerra']),
                     ('H3_era5', v4['P6_cluster']['H3_era5']),
                     ('H4_cerra', v4['P6_cluster']['H4_cerra'])):
        k, n = icc['k_out'], icc['n']
        rho = icc['icc']
        p_bb = beta_binomial_tail(k, n, 0.05, rho)
        contrib = icc['block_signflip']['block_contrib']
        p_cb = cluster_boot_p(contrib, blocks)
        tests[key] = {'k': k, 'n': n, 'icc': rho,
                      'p_binomial_naive': icc['p_binomial_naive'],
                      'p_beta_binomial': p_bb,
                      'p_cluster_bootstrap_share_le_05': p_cb,
                      'p_signflip_one_sided': icc['block_signflip']['p_one_sided'],
                      'n_effective': icc['n_effective']}
    # ERA5 excluding the two no-spell stations
    from scipy.stats import binom as _binom
    tests['H2_era5_excl_nospell'] = {'k': 12, 'n': 30,
                                     'p_binomial_naive': float(_binom.sf(11, 30, 0.05))}
    out['C_share_tests'] = tests
    print('\n== C. share tests ==')
    print(json.dumps(tests, indent=1))

    # ================= D. variance ratios (Jensen regime dependence) =========
    vobs, vc, ve = [], [], []
    for sid in CONF:
        g = gby[sid]
        vobs.append(g.ws_obs.var(ddof=1))
        vc.append(g.ws_cerra.var(ddof=1))
        ve.append(g.ws_era5.var(ddof=1))
    vobs, vc, ve = map(np.array, (vobs, vc, ve))
    out['D_variance'] = {
        'median_sd_daily_means_ms': {'obs': float(np.sqrt(np.median(vobs))),
                                     'cerra': float(np.sqrt(np.median(vc))),
                                     'era5': float(np.sqrt(np.median(ve)))},
        'median_var_ratio_cerra_over_obs': float(np.median(vc / vobs)),
        'median_var_ratio_era5_over_obs': float(np.median(ve / vobs)),
        'median_var_ratio_era5_over_cerra': float(np.median(ve / vc)),
        'pooled_var': {'obs': float(vobs.sum()), 'cerra': float(vc.sum()),
                       'era5': float(ve.sum())},
        'pooled_var_ratio_era5_over_obs': float(ve.sum() / vobs.sum()),
        'pooled_var_ratio_cerra_over_obs': float(vc.sum() / vobs.sum()),
    }
    print('\n== D. variance ratios ==')
    print(json.dumps(out['D_variance'], indent=1))

    # ================= E. bilinear dz + lsm regression =======================
    inv = np.load(NPZ)
    z, lsm, lat_g, lon_g = inv['z'], inv['lsm'], inv['lat'], inv['lon']
    G0, I0 = 200, 1360
    dem = pd.read_csv(DEM, dtype={'station_id': str})
    rows = []
    for sid in CONF:
        m = json.load(open(os.path.join(PARTS, f'meta_{sid}.json')))
        ec = m['era5_cell']
        j0, i0, wy, wx = ec['j0'], ec['i0'], ec['wy'], ec['wx']
        corners = [(G0 + j0, (I0 + i0) % 1440), (G0 + j0, (I0 + i0 + 1) % 1440),
                   (G0 + j0 + 1, (I0 + i0) % 1440), (G0 + j0 + 1, (I0 + i0 + 1) % 1440)]
        wts = [(1 - wy) * (1 - wx), (1 - wy) * wx, wy * (1 - wx), wy * wx]
        zc = np.array([z[jj, ii] for jj, ii in corners])
        lc = np.array([lsm[jj, ii] for jj, ii in corners])
        z_bil = float((zc * wts).sum())
        lsm_bil = float((lc * wts).sum())
        elev = m['elev']
        rows.append({'station_id': sid, 'class': dict(st[['station_id', 'class']]
                                                     .values.tolist())[sid] if False else
                     st.set_index('station_id').loc[sid, 'class'],
                     'elev_station': elev, 'z_era5_bilinear': round(z_bil, 1),
                     'dz_era5_bilinear': round(elev - z_bil, 1),
                     'lsm_bilinear': round(lsm_bil, 3),
                     'lsm_corners_mean': round(float(lc.mean()), 3),
                     'n_ocean_corners_lsm05': int((lc < 0.5).sum())})
    dz = pd.DataFrame(rows)
    dz.to_csv(os.path.join(RES, 'era5_dz_bilinear_v5.csv'), index=False)
    # validation vs nearest-corner values of v4
    v4o = pd.read_csv(os.path.join(RES, 'era5_orography_v4.csv'),
                      dtype={'station_id': str}).set_index('station_id')
    dzv = dz.set_index('station_id')
    can = dz[dz['class'] == 'archipelago'].copy()
    can['bias_era5'] = can.station_id.map(st.set_index('station_id').bias_era5)
    regres = {}
    for xcol in ('lsm_bilinear', 'lsm_corners_mean'):
        x = can[xcol].to_numpy()
        y = can.bias_era5.to_numpy()
        sl, ic, r, p, se = stats.linregress(x, y)
        rho_s, rho_p = stats.spearmanr(x, y)
        regres[xcol] = {'slope_ms_per_lsm': float(sl), 'intercept': float(ic),
                        'r': float(r), 'p': float(p), 'se': float(se),
                        'spearman_rho': float(rho_s), 'spearman_p': float(rho_p),
                        'n': 8}
    out['E_dz_bilinear'] = {
        'class_medians': dz.groupby('class')[
            ['dz_era5_bilinear', 'lsm_bilinear']].median().round(2).to_dict('index'),
        'validation_z_near_corr': float(np.corrcoef(
            dzv.z_era5_bilinear, v4o.loc[dzv.index, 'z_era5_near'])[0, 1]),
        'canary_bias_regression': regres,
    }
    print('\n== E. dz bilinear ==')
    print(dz.groupby('class')[['dz_era5_bilinear', 'lsm_bilinear']].median().round(2))
    print(json.dumps(regres, indent=1))

    # external validity numbers (reviewer point 12)
    mwind = pd.concat([gby[s].ws_obs for s in CONF])
    per_st = np.array([gby[s].ws_obs.mean() for s in CONF])
    out['F_external_validity'] = {
        'obs_mean_wind_median': float(np.median(per_st)),
        'obs_mean_wind_iqr': [float(np.quantile(per_st, 0.25)),
                              float(np.quantile(per_st, 0.75))],
        'obs_mean_wind_range': [float(per_st.min()), float(per_st.max())],
        'stations_in_atlantic_hub_bbox': 0,
    }
    with open(os.path.join(RES, 'revision_v5.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nrevision_v5.json written')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
