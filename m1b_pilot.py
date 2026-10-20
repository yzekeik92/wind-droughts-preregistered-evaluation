#!/usr/bin/env python3
"""M1b PILOT: common like-for-like sampling + spatial pairing (5 pilot stations).

Scope: code calibration and feasibility ONLY (prereg v1.1 sect.6, deviation
#10). These numbers are NEVER evidence for/against H1-H4.

Implements prereg v1.1 sect.4-5 + REVIEWER_PROTOCOL D3/D4:
  - obs daily values recomputed with the EXACT M1a logic, then compared to
    daily_obs_v2.parquet (equivalence proof);
  - same matched-slot set used for station, CERRA and ERA5 (identical masking);
  - CERRA nearest cell = min great-circle distance to cell centre (2D lat/lon
    arrays shipped in the files), asserted identical across years;
  - ERA5 bilinear on u10 & v10 then ws = sqrt(u^2+v^2); speed-then-interp
    variant kept for the D3 sensitivity;
  - slot maps: obs slot h -> CERRA 3*((h+1)//3), ERA5 exact h.

Outputs:
  /tmp/my-project/M1b/pilot_daily_paired.parquet
  /tmp/my-project/verification/m1b_pilot_diagnostics.json
"""
import datetime as dt
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import netCDF4 as nc

EPOCH = dt.datetime(1931, 1, 1)
W0 = dt.datetime(1995, 1, 1)
W1 = dt.datetime(2025, 1, 1)
LO = (W0 - EPOCH).total_seconds() / 3600.0
HI = (W1 - EPOCH).total_seconds() / 3600.0
OFF1970 = (dt.date(1970, 1, 1) - dt.date(1931, 1, 1)).days

PILOTS = ['601850', '601900', '605170', '605180', '605600']
ELIG = '/tmp/my-project/verification/m1a_eligibility_v2.csv'
HAD_DIR = '/tmp/my-project/HadISD_subdaily'
CERRA_DIR = '/tmp/my-project/CERRA_data'
ERA5_DIR = '/tmp/my-project/ERA5_data'
PARQUET = '/tmp/my-project/M1a/daily_obs_v2.parquet'
OUT_DIR = '/tmp/my-project/M1b'
DIAG_JSON = '/tmp/my-project/verification/m1b_pilot_diagnostics.json'
YEARS = list(range(1995, 2025))


def circ_dist(h1, h2):
    return np.abs(((h1 - h2 + 12.0) % 24.0) - 12.0)


def lst_hour(utc_h, lon):
    return (utc_h + lon / 15.0) % 24.0


def is_diurnal(lst):
    return 6.0 <= lst < 18.0


def spearman(a, b):
    sa = pd.Series(a).rank()
    sb = pd.Series(b).rank()
    return float(np.corrcoef(sa, sb)[0, 1])


# ---------------------------------------------------------------- observations
def obs_daily(sid, H, lon):
    """Replicates m1a_observation_audit.py exactly; adds per-day slot list."""
    p = os.path.join(HAD_DIR, f"{sid}.nc")
    ds = nc.Dataset(p)
    t = np.asarray(ds.variables["time"][:], dtype="float64")
    ws = np.asarray(ds.variables["windspeeds"][:], dtype="float64")
    ds.close()
    m = (t >= LO) & (t < HI) & np.isfinite(ws) & (ws >= 0)
    t_h, w = t[m], ws[m]
    hour = t_h % 24.0
    day_idx = np.floor(t_h / 24.0).astype("int64")

    need = math.ceil(0.75 * len(H))
    lst = {h: lst_hour(h, lon) for h in range(24)}
    H_night = [h for h in H if not is_diurnal(lst[h])]
    H_day = [h for h in H if is_diurnal(lst[h])]

    slot_days, slot_vals = {}, {}
    for h in H:
        mask = circ_dist(hour, float(h)) <= 1.0
        if not mask.any():
            continue
        d = day_idx[mask]
        dt_abs = np.abs(((t_h[mask] % 24.0) - h + 12.0) % 24.0 - 12.0)
        w_sub = w[mask]  # BUGFIX 2026-10-17: was w[order] on the global array
        order = np.lexsort((dt_abs, d))
        d_s, dt_s, w_s = d[order], dt_abs[order], w_sub[order]
        uniq_days, first_idx = np.unique(d_s, return_index=True)
        slot_days[h] = uniq_days
        slot_vals[h] = w_s[first_idx]

    per_day = {}
    for h in H:
        for d, val in zip(slot_days.get(h, []), slot_vals.get(h, [])):
            per_day.setdefault(d, []).append((h, val))

    rows = []
    for d, pairs in per_day.items():
        slots = [h for h, _ in pairs]
        night_ok = any(h in H_night for h in slots)
        day_ok = any(h in H_day for h in slots)
        ok = len(slots) >= need and night_ok and day_ok
        date = (EPOCH + dt.timedelta(days=int(d))).strftime("%Y-%m-%d")
        rows.append({"station_id": sid, "date": date, "day_abs": int(d),
                     "ws_daily": float(np.mean([v for _, v in pairs])),
                     "n_slots": len(slots), "night_ok": int(night_ok),
                     "day_ok": int(day_ok), "valid": int(ok),
                     "slots": ";".join(map(str, slots))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True), H_night, H_day


def equivalence_check(mine, sid):
    ref = pd.read_parquet(PARQUET)
    ref = ref[ref["station_id"] == sid].reset_index(drop=True)
    mg = mine.merge(ref, on="date", suffixes=("_m", "_r"), how="outer",
                    indicator=True)
    problems = []
    if not (mg["_merge"] == "both").all():
        problems.append(f"row-set mismatch: {int((mg['_merge'] != 'both').sum())}")
    b = mg[mg["_merge"] == "both"]
    dw = float(np.max(np.abs(b["ws_daily_m"] - b["ws_daily_r"]))) if len(b) else 0.0
    if dw > 1e-9:
        problems.append(f"ws_daily max|diff|={dw:.3e}")
    for c in ("n_slots", "night_ok", "day_ok", "valid"):
        nd = int((b[f"{c}_m"] != b[f"{c}_r"]).sum())
        if nd:
            problems.append(f"{c} mismatches: {nd}")
    return problems, len(b)


# ------------------------------------------------------------------- CERRA/ERA5
def haversine_nearest(la2d, lo2d, lat0, lon0):
    R = 6371.0088
    p1, l1 = np.radians(lat0), np.radians(lon0)
    p2, l2 = np.radians(la2d), np.radians(lo2d)
    dl = (l2 - l1 + np.pi) % (2 * np.pi) - np.pi
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a))
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    return int(iy), int(ix), float(dist[iy, ix])


def read_cerra_year(year, cell_cache):
    path = os.path.join(CERRA_DIR, f"cerra_{year}_morocco.nc")
    d = nc.Dataset(path)
    vt = np.asarray(d.variables["valid_time"][:], dtype="int64")
    days = vt // 86400
    hours = (vt % 86400) // 3600
    la2d = d.variables["latitude"][:]
    lo2d = d.variables["longitude"][:]
    out = {}
    for sid, (lat0, lon0) in CELLS.items():
        iy, ix, dist = haversine_nearest(la2d, lo2d, lat0, lon0)
        if sid in cell_cache:
            if cell_cache[sid][:2] != (iy, ix):
                print(f"  WARNING {sid} {year}: cell moved "
                      f"{cell_cache[sid][:2]} -> {(iy, ix)}")
        else:
            cell_cache[sid] = (iy, ix, dist, float(la2d[iy, ix]),
                               float(lo2d[iy, ix]) if float(lo2d[iy, ix]) <= 180
                               else float(lo2d[iy, ix]) - 360.0)
        col = d.variables["si10"][:, iy, ix]
        col = np.ma.filled(np.ma.asarray(col, dtype="float64"), np.nan)
        per = {}
        for k in range(vt.size):
            dd = per.setdefault(int(days[k]), {})
            dd[int(hours[k])] = col[k]
        out[sid] = per
        all8 = {dd: (float(np.mean(list(hh.values()))), len(hh))
                for dd, hh in per.items()}
        out[sid + "_all8"] = all8
    d.close()
    return out


def read_era5_year(year):
    path = os.path.join(ERA5_DIR, f"{year}.nc")
    d = nc.Dataset(path)
    vt = np.asarray(d.variables["valid_time"][:], dtype="int64")
    days = vt // 86400
    hours = (vt % 86400) // 3600
    lat = np.asarray(d.variables["latitude"][:], dtype="float64")
    lon = np.asarray(d.variables["longitude"][:], dtype="float64")
    out = {}
    for sid, (lat0, lon0) in CELLS.items():
        jf = (lat[0] - lat0) / (lat[0] - lat[1])
        j0 = int(np.clip(np.floor(jf), 0, len(lat) - 2))
        wy = jf - j0
        if_ = (lon0 - lon[0]) / (lon[1] - lon[0])
        i0 = int(np.clip(np.floor(if_), 0, len(lon) - 2))
        wx = if_ - i0
        u = np.ma.filled(d.variables["u10"][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype("float64")
        v = np.ma.filled(d.variables["v10"][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype("float64")
        u00, u01 = u[:, 0, 0], u[:, 0, 1]
        u10v, u11 = u[:, 1, 0], u[:, 1, 1]
        v00, v01 = v[:, 0, 0], v[:, 0, 1]
        v10v, v11 = v[:, 1, 0], v[:, 1, 1]
        uint = (1 - wy) * ((1 - wx) * u00 + wx * u01) + wy * ((1 - wx) * u10v + wx * u11)
        vint = (1 - wy) * ((1 - wx) * v00 + wx * v01) + wy * ((1 - wx) * v10v + wx * v11)
        ws_comp = np.hypot(uint, vint)
        w00, w01 = np.hypot(u00, v00), np.hypot(u01, v01)
        w10, w11 = np.hypot(u10v, v10v), np.hypot(u11, v11)
        ws_spd = (1 - wy) * ((1 - wx) * w00 + wx * w01) + wy * ((1 - wx) * w10 + wx * w11)
        per, per_spd = {}, {}
        for k in range(vt.size):
            dd = per.setdefault(int(days[k]), {})
            dd[int(hours[k])] = float(ws_comp[k])
            per_spd.setdefault(int(days[k]), {})[int(hours[k])] = float(ws_spd[k])
        all24 = {}
        uniq, inv = np.unique(days, return_inverse=True)
        cnt = np.bincount(inv)
        s = np.bincount(inv, weights=ws_comp)
        for uu, cc, ss in zip(uniq, cnt, s):
            all24[int(uu)] = float(ss / cc)
        out[sid] = (per, per_spd, all24, (j0, i0, float(wy), float(wx)))
    d.close()
    return out


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    el = pd.read_csv(ELIG, dtype={"station_id": str})
    el = el[el["station_id"].isin(PILOTS)]

    global CELLS
    CELLS = {r.station_id: (r.lat, r.lon) for r in el.itertuples()}

    # --- 1) obs recompute + equivalence proof
    obs = {}
    hinfo = {}
    print("== M1b PILOT — equivalence check vs daily_obs_v2.parquet ==")
    for sid in PILOTS:
        r = el[el["station_id"] == sid].iloc[0]
        H = sorted(int(x) for x in str(r.habitual_slots).split(";"))
        mine, Hn, Hd = obs_daily(sid, H, float(r.lon))
        probs, ncmp = equivalence_check(mine, sid)
        status = "OK" if not probs else "FAIL: " + "; ".join(probs)
        print(f"  {sid}: {ncmp} days compared -> {status}")
        if probs:
            sys.exit(1)
        obs[sid] = mine
        hinfo[sid] = {"H": H, "H_night": Hn, "H_day": Hd,
                      "need": math.ceil(0.75 * len(H)),
                      "lat": float(r.lat), "lon": float(r.lon),
                      "elev": float(r.elevation)}

    # --- 2) CERRA columns (all years)
    print("== CERRA pairing (nearest great-circle cell, per-year assertion) ==")
    cell_cache = {}
    cerra = {sid: {} for sid in PILOTS}
    cerra_all8 = {sid: {} for sid in PILOTS}
    for y in YEARS:
        yr = read_cerra_year(y, cell_cache)
        for sid in PILOTS:
            cerra[sid].update(yr[sid])
            cerra_all8[sid].update(yr[sid + "_all8"])
        if (y - 1994) % 10 == 0:
            print(f"  ... CERRA {y} done")
    for sid in PILOTS:
        iy, ix, dist, cla, clo = cell_cache[sid]
        hinfo[sid]["cerra_cell"] = {"iy": iy, "ix": ix, "dist_km": round(dist, 3),
                                    "cell_lat": round(cla, 5), "cell_lon": round(clo, 5)}
        print(f"  {sid}: CERRA cell (iy,ix)=({iy},{ix}) d={dist:.2f} km "
              f"({cla:.4f},{clo:.4f})")

    # --- 3) ERA5 bilinear
    print("== ERA5 pairing (bilinear u,v -> ws; D3) ==")
    era5 = {sid: {} for sid in PILOTS}
    era5_spd = {sid: {} for sid in PILOTS}
    era5_all24 = {sid: {} for sid in PILOTS}
    for y in YEARS:
        yr = read_era5_year(y)
        for sid in PILOTS:
            per, per_spd, all24, cell = yr[sid]
            era5[sid].update(per)
            era5_spd[sid].update(per_spd)
            era5_all24[sid].update(all24)
            hinfo[sid]["era5_cell"] = {"j0": cell[0], "i0": cell[1],
                                       "wy": round(cell[2], 4), "wx": round(cell[3], 4)}
        if (y - 1994) % 10 == 0:
            print(f"  ... ERA5 {y} done")

    # --- 4) like-for-like merge (identical masking)
    print("== like-for-like merge ==")
    rows = []
    integrity_fail = 0
    for sid in PILOTS:
        H = hinfo[sid]["H"]
        mapped = {h: 3 * ((h + 1) // 3) for h in H}
        for r in obs[sid].itertuples():
            day70 = r.day_abs - OFF1970
            slots = [int(x) for x in r.slots.split(";")]
            cv, ev, es = [], [], []
            miss = 0
            for h in slots:
                mh = mapped[h]
                v = cerra[sid].get(day70, {}).get(mh)
                if v is None:
                    miss += 1
                else:
                    cv.append(v)
                e = era5[sid].get(day70, {}).get(h)
                if e is None:
                    miss += 1
                else:
                    ev.append(e)
                    es.append(era5_spd[sid][day70][h])
            if miss:
                integrity_fail += miss
                continue  # identical masking would kill the day for ALL series
            a8 = cerra_all8[sid].get(day70, (np.nan, 0))
            e24 = era5_all24[sid].get(day70, np.nan)
            rows.append({"station_id": sid, "date": r.date,
                         "n_slots": r.n_slots, "slots": r.slots,
                         "ws_obs": r.ws_daily,
                         "ws_cerra": float(np.mean(cv)),
                         "ws_era5": float(np.mean(ev)),
                         "ws_era5_spd": float(np.mean(es)),
                         "cerra_all8": a8[0], "cerra_n8": a8[1],
                         "era5_all24": e24, "valid": r.valid})
    if integrity_fail:
        print(f"  INTEGRITY: {integrity_fail} missing model slots (days dropped, "
              f"masking identical across series)")
    else:
        print("  INTEGRITY: zero missing model slots — masking fully identical")
    df = pd.DataFrame(rows).sort_values(["station_id", "date"]).reset_index(drop=True)
    df.to_parquet(os.path.join(OUT_DIR, "pilot_daily_paired.parquet"), index=False)
    print(f"  paired daily rows: {len(df)} "
          f"(valid: {int(df['valid'].sum())})")

    # --- 5) pilot diagnostics (calibration only)
    print("== pilot diagnostics (calibration, NOT confirmatory) ==")
    diag = {"params": {"window": "1995-2024", "pilots": PILOTS,
                       "rule": "prereg v1.1 sect.4-5", "executed": dt.datetime.utcnow().isoformat() + "Z"},
            "stations": {}}
    for sid in PILOTS:
        g = df[(df.station_id == sid) & (df.valid == 1)].copy()
        g["d"] = pd.to_datetime(g["date"])
        n = len(g)
        if n == 0:
            continue
        bc = float((g.ws_cerra - g.ws_obs).mean())
        be = float((g.ws_era5 - g.ws_obs).mean())
        rc = float(np.sqrt(((g.ws_cerra - g.ws_obs) ** 2).mean()))
        re_ = float(np.sqrt(((g.ws_era5 - g.ws_obs) ** 2).mean()))
        pc = float(np.corrcoef(g.ws_obs, g.ws_cerra)[0, 1])
        pe = float(np.corrcoef(g.ws_obs, g.ws_era5)[0, 1])
        sc, se = spearman(g.ws_obs, g.ws_cerra), spearman(g.ws_obs, g.ws_era5)
        dmax = float(np.max(np.abs(g.ws_era5 - g.ws_era5_spd)))
        dmean = float(np.mean(np.abs(g.ws_era5 - g.ws_era5_spd)))
        band_c = float(np.mean(np.abs(g.ws_cerra - g.cerra_all8)))
        band_e = float(np.mean(np.abs(g.ws_era5 - g.era5_all24)))
        # anomalies vs monthly climatology; ACF(1) on consecutive valid days
        anom = {}
        for col in ("ws_obs", "ws_cerra", "ws_era5"):
            clim = g.groupby(g.d.dt.month)[col].transform("mean")
            anom[col] = g[col] - clim
        consec = g.d.diff().dt.days == 1
        acf = {c: float(np.corrcoef(anom[c][consec], anom[c][consec.shift(-1, fill_value=False)])[0, 1])
               for c in anom}
        # low-wind day overlap (own P10, calibration only). '<=' so that a
        # calm-degenerate threshold (P10 = 0.0, e.g. 605180) still yields a
        # defined indicator; P10 values themselves are reported.
        thr_o = float(g.ws_obs.quantile(0.10))
        lo = g.ws_obs <= thr_o
        ovl = {}
        for col in ("ws_cerra", "ws_era5"):
            thr_m = float(g[col].quantile(0.10))
            ovl[col] = float(((g[col] <= thr_m) & lo).mean())
        # annual means P10/P50/P90
        yrmeans = g.groupby(g.d.dt.year)[["ws_obs", "ws_cerra", "ws_era5"]].mean()
        pxx = {c: [float(yrmeans[c].quantile(q)) for q in (0.1, 0.5, 0.9)]
               for c in yrmeans.columns}
        diag["stations"][sid] = {
            "n_valid_days": n, "first": g.date.iloc[0], "last": g.date.iloc[-1],
            "bias_cerra": round(bc, 3), "bias_era5": round(be, 3),
            "rmse_cerra": round(rc, 3), "rmse_era5": round(re_, 3),
            "pearson_cerra": round(pc, 3), "pearson_era5": round(pe, 3),
            "spearman_cerra": round(sc, 3), "spearman_era5": round(se, 3),
            "acf1_obs": round(acf["ws_obs"], 3),
            "acf1_cerra": round(acf["ws_cerra"], 3),
            "acf1_era5": round(acf["ws_era5"], 3),
            "lowwind_overlap_cerra": round(ovl["ws_cerra"], 3),
            "lowwind_overlap_era5": round(ovl["ws_era5"], 3),
            "annual_P10_P50_P90": {k: [round(x, 2) for x in v] for k, v in pxx.items()},
            "era5_interp_order_maxdiff": round(dmax, 4),
            "era5_interp_order_meandiff": round(dmean, 4),
            "masked_band_cerra": round(band_c, 3),
            "masked_band_era5": round(band_e, 3),
            "daily_P10": {"ws_obs": round(thr_o, 3),
                          "ws_cerra": round(float(g.ws_cerra.quantile(0.10)), 3),
                          "ws_era5": round(float(g.ws_era5.quantile(0.10)), 3)},
            "n_zero_days_obs": int((g.ws_obs == 0).sum()),
            **hinfo[sid]}
        s = diag["stations"][sid]
        print(f"  {sid}: n={n} bias C/E {bc:+.2f}/{be:+.2f}  RMSE {rc:.2f}/{re_:.2f}  "
              f"rho {sc:.3f}/{se:.3f}  acf1 {s['acf1_obs']:.2f}/{s['acf1_cerra']:.2f}/{s['acf1_era5']:.2f}  "
              f"loWindOv {ovl['ws_cerra']:.2f}/{ovl['ws_era5']:.2f}  interpΔmax {dmax:.3f}")
    with open(DIAG_JSON, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nPILOT DONE -> {OUT_DIR}/pilot_daily_paired.parquet")
    print(f"             {DIAG_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
