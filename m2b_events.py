#!/usr/bin/env python3
"""M2b: event-hit verification (POD/FAR/CSI, >=50 % overlap rule), annual
difference series for H4 figures, O5 per-station storage-depth distributions,
and per-class exploratory summary.

Exploratory layer of the preregistered plan (proposal Phase 2 metrics).
"""
import json
import os

import numpy as np
import pandas as pd

PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
DEM = '/tmp/my-project/verification/dem_covariates.csv'
RES = '/home/z/my-project/download/M2_results'
SENS3 = ['603400', '605220', '605570']

CLASSES = {
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
    """POD/FAR/CSI with >=overlap fraction of the spell's days matched."""
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
    return pod, far, csi, hits, misses, fa


def main():
    os.makedirs(RES, exist_ok=True)
    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    draw = json.load(open('/tmp/my-project/verification/pilot_draw.json'))
    CONF = draw["confirmatory_stations"]

    # ---------------- event hit rates (exploratory) --------------------------
    rows = []
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)].sort_values("date")
        yr = g.date.dt.year.to_numpy()
        dint = g.date.map(pd.Timestamp.toordinal).to_numpy().astype("int64")
        thr = {}
        trig = False
        for p, c in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
            t = g.groupby(g.date.dt.year)[c].quantile(0.10)
            thr[p] = t
            if (t == 0).any():
                trig = True
        tmap = {}
        for p in thr:
            t = thr[p]
            tmap[p] = np.array([max(t.get(y, np.nan), 0.5) if trig else t.get(y, np.nan)
                                for y in yr])
        obs_spells_T1 = runs(g.ws_obs.to_numpy() <= tmap["obs"], dint)
        for prod in ("cerra", "era5"):
            col = f"ws_{prod}"
            for regime, thr_arr in (("T1_own", tmap[prod]), ("T2_obsthreshold", tmap["obs"])):
                mlow = g[col].to_numpy() <= thr_arr
                idx = pd.Series(mlow).to_numpy()
                mod_spells = runs(mlow, dint)
                pod, far, csi, h, m, fa = hit_rates(obs_spells_T1, mod_spells, mlow)
                rows.append({"station_id": sid, "class": CLASS_OF[sid], "product": prod,
                             "regime": regime, "POD": pod, "FAR": far, "CSI": csi,
                             "hits": h, "misses": m, "false_alarms": fa,
                             "n_obs_spells": len(obs_spells_T1),
                             "n_mod_spells": len(mod_spells)})
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(RES, "event_hit_rates.csv"), index=False)
    print("POD/FAR/CSI medians:")
    print(ev.groupby(["product", "regime"])[["POD", "FAR", "CSI"]].median().round(3).to_string())

    # ---------------- annual difference series (H4 figure data) --------------
    h4 = pd.read_csv(os.path.join(RES, "h4_pettitt.csv"),
                     dtype={"station_id": str})
    rows = []
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)]
        ann = g.groupby(g.date.dt.year)[["ws_obs", "ws_cerra", "ws_era5"]].mean()
        cnt = g.groupby(g.date.dt.year).size()
        ann = ann[cnt >= 274]
        for y, r in ann.iterrows():
            row = {"station_id": sid, "class": CLASS_OF[sid], "year": int(y),
                   "obs": r.ws_obs, "cerra": r.ws_cerra, "era5": r.ws_era5,
                   "d_cerra": r.ws_cerra - r.ws_obs, "d_era5": r.ws_era5 - r.ws_obs,
                   "d_ec": r.ws_era5 - r.ws_cerra}
            for prod in ("cerra", "era5"):
                if sid in h4.station_id.values:
                    rr = h4[h4.station_id == sid].iloc[0]
                    row[f"q_{prod}"] = rr[f"pettitt_q_{prod}"]
                    row[f"t_{prod}"] = rr[f"pettitt_t_{prod}"]
            rows.append(row)
    ann_df = pd.DataFrame(rows)
    ann_df.to_csv(os.path.join(RES, "annual_series_H4.csv"), index=False)

    # ---------------- O5 per-station storage distributions -------------------
    def power_curve(v, alpha):
        vh = v * (10.0 ** alpha)
        return np.where((vh < 3.0) | (vh > 25.0), 0.0,
                        np.where(vh < 12.0, (vh ** 3 - 27.0) / (1728.0 - 27.0), 1.0))

    rows = []
    for alpha in (0.10, 0.14, 0.19, 0.25):
        for q in (0.6, 0.7, 0.8):
            for sid in CONF:
                g = df[(df.station_id == sid) & (df.valid == 1)].sort_values("date")
                g = g.copy()
                g["year"] = g.date.dt.year
                cnt = g.groupby("year").size()
                ok = set(cnt[cnt >= 329].index)
                gg = g[g.year.isin(ok)]
                if gg.year.nunique() < 5:
                    continue
                for prod, col in (("obs", "ws_obs"), ("cerra", "ws_cerra"), ("era5", "ws_era5")):
                    p = power_curve(gg[col].to_numpy(), alpha)
                    mu = p.mean()
                    gt = p / mu if mu > 0 else p
                    day = gg.date.map(pd.Timestamp.toordinal).to_numpy().astype("int64")
                    yv = gg.year.to_numpy()
                    S_list = []
                    for y in sorted(set(yv)):
                        m = yv == y
                        yy, gy = day[m], gt[m]
                        if len(yy) < 300:
                            continue
                        if np.diff(yy).max() > 1:
                            dd = np.arange(yy[0], yy[-1] + 1)
                            gy = np.interp(dd, yy, gy)
                        D = S = 0.0
                        for gv in gy:
                            D = max(0.0, D + q - gv)
                            S = max(S, D)
                        S_list.append(S)
                    if S_list:
                        rows.append({"station_id": sid, "class": CLASS_OF[sid],
                                     "alpha": alpha, "q": q, "product": prod,
                                     "S_median": float(np.median(S_list)),
                                     "S_max": float(np.max(S_list)),
                                     "S_p90": float(np.quantile(S_list, 0.9))})
    o5 = pd.DataFrame(rows)
    o5.to_csv(os.path.join(RES, "o5_per_station.csv"), index=False)

    # ---------------- per-class summary (exploratory) ------------------------
    st = pd.read_csv(os.path.join(RES, "station_metrics_confirmatory.csv"),
                     dtype={"station_id": str})
    cls = st.groupby("class").agg(
        n=("station_id", "size"),
        rmse_cerra=("rmse_cerra", "median"), rmse_era5=("rmse_era5", "median"),
        bias_cerra=("bias_cerra", "median"), bias_era5=("bias_era5", "median"),
        ratio_obs=("ratio_obs", "median"), ratio_cerra=("ratio_cerra", "median"),
        ratio_era5=("ratio_era5", "median"),
        dz_cerra=("dz_cerra", "median"), relief_cerra=("relief_cerra3x3", "median"),
    ).round(3)
    ev_m = ev[ev.regime == "T1_own"].groupby("class")[["POD", "FAR", "CSI"]].median().round(3)
    cls = cls.join(ev_m, how="left")
    cls.to_csv(os.path.join(RES, "class_summary.csv"))
    print("\nClass summary:")
    print(cls.to_string())
    print("\nDONE m2b")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
