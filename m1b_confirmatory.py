#!/usr/bin/env python3
"""M1b CONFIRMATORY: like-for-like sampling + spatial pairing (32 confirmatory
+ 3 preregistered sensitivity stations).

Follows prereg v1.2 sect.4-5 + REVIEWER_PROTOCOL D3/D4 exactly, reusing the
calibrated pilot code (m1b_pilot.py, deviations #10/#11).

RESUMABLE design (sandbox kills background jobs):
  python3 m1b_confirmatory.py <i0> <i1>   -> processes ALL[i0:i1], one parquet
        part + one meta JSON per station
  python3 m1b_confirmatory.py merge       -> concatenates parts, runs obs
        equivalence checks vs daily_obs_v2.parquet, writes diagnostics JSON

Outputs:
  /tmp/my-project/M1b/parts/pair_<sid>.parquet, meta_<sid>.json
  /tmp/my-project/M1b/confirmatory_daily_paired.parquet
  /tmp/my-project/verification/m1b_confirmatory_diagnostics.json
"""
import datetime as dt
import glob
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

ELIG = '/tmp/my-project/verification/m1a_eligibility_v2.csv'
AUDIT = '/tmp/my-project/verification/m1a_station_audit.csv'
PILOT_JSON = '/tmp/my-project/verification/pilot_draw.json'
HAD_DIR = '/tmp/my-project/HadISD_subdaily'
CERRA_DIR = '/tmp/my-project/CERRA_data'
ERA5_DIR = '/tmp/my-project/ERA5_data'
PARQUET = '/tmp/my-project/M1a/daily_obs_v2.parquet'
OUT_DIR = '/tmp/my-project/M1b'
PARTS = os.path.join(OUT_DIR, 'parts')
DIAG_JSON = '/tmp/my-project/verification/m1b_confirmatory_diagnostics.json'
SENS3 = ['603400', '605220', '605570']
YEARS = list(range(1995, 2025))
CERRA_HOURS = [0, 3, 6, 9, 12, 15, 18, 21]


def circ_dist(h1, h2):
    return np.abs(((h1 - h2 + 12.0) % 24.0) - 12.0)


def lst_hour(utc_h, lon):
    return (utc_h + lon / 15.0) % 24.0


def is_diurnal(lst):
    return 6.0 <= lst < 18.0


# ---------------------------------------------------------------- observations
def read_station_nc(sid):
    p = os.path.join(HAD_DIR, f"{sid}.nc")
    ds = nc.Dataset(p)
    t = np.asarray(ds.variables["time"][:], dtype="float64")
    ws = np.asarray(ds.variables["windspeeds"][:], dtype="float64")
    qc = np.asarray(ds.variables["quality_control_flags"][:])
    ds.close()
    suspect = (np.nansum((qc == 2), axis=1) > 0)  # strict-QC sensitivity (R14)
    return t, ws, suspect


def slot_extract(t_h, w, hour):
    """closest obs to nominal hour h within +/-1h; ties -> earliest (lexsort)."""
    mask = circ_dist(t_h % 24.0, float(hour)) <= 1.0
    if not mask.any():
        return None, None
    d = np.floor(t_h[mask] / 24.0).astype("int64")
    dt_abs = np.abs(((t_h[mask] % 24.0) - hour + 12.0) % 24.0 - 12.0)
    order = np.lexsort((dt_abs, d))
    d_s, w_s = d[order], w[mask][order]
    uniq_days, first_idx = np.unique(d_s, return_index=True)
    return uniq_days, w_s[first_idx]


def obs_daily(sid, H, lon):
    """Exact M1a logic + strict-QC variant (suspect flags dropped)."""
    t, ws, susp = read_station_nc(sid)
    m = (t >= LO) & (t < HI) & np.isfinite(ws) & (ws >= 0)
    mq = m & (~susp)
    series = {"def": (t[m], ws[m]), "qc": (t[mq], ws[mq])}
    need = math.ceil(0.75 * len(H))
    lst = {h: lst_hour(h, lon) for h in range(24)}
    H_night = [h for h in H if not is_diurnal(lst[h])]
    H_day = [h for h in H if is_diurnal(lst[h])]

    def build(key):
        t_h, w = series[key]
        per_day = {}
        for h in H:
            days, vals = slot_extract(t_h, w, h)
            if days is None:
                continue
            for d, val in zip(days, vals):
                per_day.setdefault(int(d), []).append((h, val))
        rows = []
        for d, pairs in per_day.items():
            slots = [h for h, _ in pairs]
            night_ok = any(h in H_night for h in slots)
            day_ok = any(h in H_day for h in slots)
            ok = len(slots) >= need and night_ok and day_ok
            date = (EPOCH + dt.timedelta(days=int(d))).strftime("%Y-%m-%d")
            rows.append({"station_id": sid, "date": date, "day_abs": int(d),
                         "ws_daily": float(np.mean([v for _, v in pairs])),
                         "n_slots": len(slots), "valid": int(ok),
                         "slots": ";".join(map(str, sorted(slots)))})
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    return build("def"), build("qc"), H_night, H_day


def equivalence_check(mine, sid, cols=("n_slots", "valid")):
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
    for c in cols:
        nd = int((b[f"{c}_m"] != b[f"{c}_r"]).sum())
        if nd:
            problems.append(f"{c} mismatches: {nd}")
    return problems, len(b)


# ------------------------------------------------------------------- models
def haversine_two_nearest(la2d, lo2d, lat0, lon0):
    """Nearest and runner-up cells by great-circle distance to centre (D4)."""
    la2d = np.ma.filled(np.ma.asarray(la2d, dtype="float64"), 999.0)
    lo2d = np.ma.filled(np.ma.asarray(lo2d, dtype="float64"), 999.0)
    R = 6371.0088
    p1, l1 = np.radians(lat0), np.radians(lon0)
    p2, l2 = np.radians(la2d), np.radians(lo2d)
    dl = (l2 - l1 + np.pi) % (2 * np.pi) - np.pi
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * R * np.arcsin(np.sqrt(a))
    flat = dist.ravel()
    idx = np.argpartition(flat, 1)[:2]
    order = idx[np.argsort(flat[idx])]
    out = []
    for f in order:
        iy, ix = np.unravel_index(f, dist.shape)
        clo = float(lo2d[iy, ix])
        out.append((int(iy), int(ix), float(dist[iy, ix]),
                    float(la2d[iy, ix]), clo - 360.0 if clo > 180 else clo))
    return out


def group_by_day(days, hours, vals, n_slots):
    out = {}
    order = np.lexsort((hours, days))
    d_s, h_s, v_s = days[order], hours[order], vals[order]
    uniq, start = np.unique(d_s, return_index=True)
    ends = np.append(start[1:], d_s.size)
    for u, s, e in zip(uniq, start, ends):
        arr = np.full(n_slots, np.nan, dtype="float64")
        arr[h_s[s:e].astype("int64")] = v_s[s:e]
        out[int(u)] = arr
    return out


def read_cerra_year(year, cells_needed):
    """cells_needed: {label: (iy, ix)} -> {label: {day70: arr8}}."""
    path = os.path.join(CERRA_DIR, f"cerra_{year}_morocco.nc")
    d = nc.Dataset(path)
    vt = np.asarray(d.variables["valid_time"][:], dtype="int64")
    days = (vt // 86400).astype("int64")
    hours = ((vt % 86400) // 3600).astype("int64")
    keep = np.isin(hours, CERRA_HOURS)
    dk, hk = days[keep], (hours[keep] // 3).astype("int64")
    res = {}
    for label, (iy, ix) in cells_needed.items():
        col = d.variables["si10"][:, iy, ix]
        col = np.ma.filled(np.ma.asarray(col, dtype="float64"), np.nan)
        res[label] = group_by_day(dk, hk, col[keep], 8)
    d.close()
    return res


def read_era5_year(year, j0, i0, jn, inn, H, wy, wx):
    """Bilinear (D3) + nearest-cell + all-24h daily means."""
    path = os.path.join(ERA5_DIR, f"{year}.nc")
    d = nc.Dataset(path)
    vt = np.asarray(d.variables["valid_time"][:], dtype="int64")
    days = (vt // 86400).astype("int64")
    hours = ((vt % 86400) // 3600).astype("int64")
    u = np.ma.filled(d.variables["u10"][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype("float64")
    v = np.ma.filled(d.variables["v10"][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype("float64")
    d.close()
    uint = (1 - wy) * ((1 - wx) * u[:, 0, 0] + wx * u[:, 0, 1]) + \
        wy * ((1 - wx) * u[:, 1, 0] + wx * u[:, 1, 1])
    vint = (1 - wy) * ((1 - wx) * v[:, 0, 0] + wx * v[:, 0, 1]) + \
        wy * ((1 - wx) * v[:, 1, 0] + wx * v[:, 1, 1])
    comp = np.hypot(uint, vint)
    spd_cell = np.hypot(u, v)
    spd = (1 - wy) * ((1 - wx) * spd_cell[:, 0, 0] + wx * spd_cell[:, 0, 1]) + \
        wy * ((1 - wx) * spd_cell[:, 1, 0] + wx * spd_cell[:, 1, 1])
    near = np.hypot(u[:, jn - j0, inn - i0], v[:, jn - j0, inn - i0])
    keep = np.isin(hours, H)
    comp_d = group_by_day(days[keep], hours[keep], comp[keep], 24)
    spd_d = group_by_day(days[keep], hours[keep], spd[keep], 24)
    near_d = group_by_day(days[keep], hours[keep], near[keep], 24)
    uniq, inv = np.unique(days, return_inverse=True)
    ok = np.isfinite(comp)
    cnt_v = np.bincount(inv, weights=ok.astype("float64"))
    s_v = np.bincount(inv, weights=np.nan_to_num(comp))
    mean_v = np.where(cnt_v > 0, s_v / np.maximum(cnt_v, 1), np.nan)
    all24_d = {int(uu): float(mm) for uu, mm in zip(uniq, mean_v)}
    return comp_d, spd_d, near_d, all24_d


# ------------------------------------------------------------------ per-station
def process_station(sid, meta_in):
    H = meta_in["H"]
    lat0, lon0 = meta_in["lat"], meta_in["lon"]
    hinfo = dict(meta_in)

    obs, obs_qc, Hn, Hd = obs_daily(sid, H, lon0)
    hinfo["H_night"], hinfo["H_day"] = Hn, Hd
    hinfo["n_obs_rows"] = len(obs)

    edge23 = [h for h in H if 3 * ((h + 1) // 3) > 21]
    mapped = {h: (0 if h == 23 else 3 * ((h + 1) // 3)) for h in H}
    hinfo["cerra_slot_map"] = {str(h): mapped[h] for h in H}
    if edge23:
        hinfo["note_h23"] = "slot 23 mapped to CERRA analysis 00 (circular nearest)"

    cerra, cerra_alt = {}, {}
    first_key = None
    for y in YEARS:
        path = os.path.join(CERRA_DIR, f"cerra_{y}_morocco.nc")
        d = nc.Dataset(path)
        la2d, lo2d = d.variables["latitude"][:], d.variables["longitude"][:]
        d.close()
        two = haversine_two_nearest(la2d, lo2d, lat0, lon0)
        key = tuple((c[0], c[1]) for c in two)
        if first_key is None:
            first_key = key
            hinfo["cerra_cell"] = {"iy": two[0][0], "ix": two[0][1],
                                   "dist_km": round(two[0][2], 3),
                                   "cell_lat": round(two[0][3], 5),
                                   "cell_lon": round(two[0][4], 5)}
            hinfo["cerra_cell_alt"] = {"iy": two[1][0], "ix": two[1][1],
                                       "dist_km": round(two[1][2], 3),
                                       "cell_lat": round(two[1][3], 5),
                                       "cell_lon": round(two[1][4], 5)}
        elif key != first_key:
            raise RuntimeError(f"{sid}: CERRA cell moved in {y}: {key} vs {first_key}")
        res = read_cerra_year(y, {"near": key[0], "alt": key[1]})
        cerra.update(res["near"])
        cerra_alt.update(res["alt"])

    era5, era5_spd, era5_near, era5_all24 = {}, {}, {}, {}
    for y in YEARS:
        path = os.path.join(ERA5_DIR, f"{y}.nc")
        d = nc.Dataset(path)
        lat = np.asarray(d.variables["latitude"][:], dtype="float64")
        lon = np.asarray(d.variables["longitude"][:], dtype="float64")
        d.close()
        jf = (lat[0] - lat0) / (lat[0] - lat[1])
        j0 = int(np.clip(np.floor(jf), 0, len(lat) - 2))
        wy = jf - j0
        if_ = (lon0 - lon[0]) / (lon[1] - lon[0])
        i0 = int(np.clip(np.floor(if_), 0, len(lon) - 2))
        wx = if_ - i0
        jn = int(np.argmin(np.abs(lat - lat0)))
        inn = int(np.argmin(np.abs(lon - lon0)))
        if "era5_cell" not in hinfo:
            hinfo["era5_cell"] = {"j0": j0, "i0": i0, "wy": round(wy, 4),
                                  "wx": round(wx, 4), "jn": jn, "inn": inn}
        comp_d, spd_d, near_d, a24 = read_era5_year(y, j0, i0, jn, inn, H, wy, wx)
        era5.update(comp_d)
        era5_spd.update(spd_d)
        era5_near.update(near_d)
        era5_all24.update(a24)

    rows = []
    integrity_fail = 0
    dmax_list, dmean_list = [], []
    for r in obs.itertuples():
        day70 = int(r.day_abs) - OFF1970
        slots = [int(x) for x in r.slots.split(";")]
        cv, cv_a, ev, es, en = [], [], [], [], []
        miss = 0
        ca = cerra.get(day70)
        ca_a = cerra_alt.get(day70)
        e5 = era5.get(day70)
        e5s = era5_spd.get(day70)
        e5n = era5_near.get(day70)
        for h in slots:
            mh = mapped[h] // 3  # arr8 indexed by 3-hourly slot position
            if ca is not None and not np.isnan(ca[mh]):
                cv.append(float(ca[mh]))
            else:
                miss += 1
            if e5 is not None and not np.isnan(e5[h]):
                ev.append(float(e5[h]))
                es.append(float(e5s[h]))
                en.append(float(e5n[h]))
                if not np.isnan(e5s[h]):
                    dmax_list.append(abs(float(e5[h]) - float(e5s[h])))
                    dmean_list.append(abs(float(e5[h]) - float(e5s[h])))
            else:
                miss += 1
            if ca_a is not None and not np.isnan(ca_a[mh]):
                cv_a.append(float(ca_a[mh]))
        if miss:
            integrity_fail += miss
            continue
        rows.append({"station_id": sid, "date": r.date, "n_slots": r.n_slots,
                     "slots": r.slots, "ws_obs": r.ws_daily,
                     "ws_cerra": float(np.mean(cv)),
                     "ws_era5": float(np.mean(ev)),
                     "ws_era5_spd": float(np.mean(es)),
                     "ws_cerra_alt": float(np.mean(cv_a)) if cv_a else np.nan,
                     "ws_era5_near": float(np.mean(en)),
                     "cerra_all8": float(np.nanmean(ca)) if ca is not None else np.nan,
                     "era5_all24": era5_all24.get(day70, np.nan),
                     "valid": r.valid})
    qc_map = {r.date: r for r in obs_qc.itertuples()}
    for row in rows:
        qr = qc_map.get(row["date"])
        row["ws_obs_qc"] = float(qr.ws_daily) if (qr is not None and qr.valid) else np.nan
        row["valid_qc"] = int(qr.valid) if qr is not None else 0
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    hinfo["era5_interp_order_maxdiff"] = round(float(np.max(dmax_list)), 4) if dmax_list else 0.0
    hinfo["era5_interp_order_meandiff"] = round(float(np.mean(dmean_list)), 4) if dmean_list else 0.0
    return df, hinfo, integrity_fail


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "0"
    if mode == "merge":
        merge_parts()
        return 0
    i0, i1 = int(sys.argv[1]), int(sys.argv[2])
    draw = json.load(open(PILOT_JSON))
    CONF = draw["confirmatory_stations"]
    ALL = CONF + SENS3
    elig = pd.read_csv(ELIG, dtype={"station_id": str})
    audit = pd.read_csv(AUDIT, dtype={"station_id": str})
    os.makedirs(PARTS, exist_ok=True)
    for sid in ALL[i0:i1]:
        outp = os.path.join(PARTS, f"pair_{sid}.parquet")
        if os.path.exists(outp):
            print(f"{sid}: part exists, skip", flush=True)
            continue
        src = elig[elig.station_id == sid]
        if len(src):
            r = src.iloc[0]
            H = sorted(int(x) for x in str(r.habitual_slots).split(";"))
            meta = {"H": H, "lat": float(r.lat), "lon": float(r.lon),
                    "elev": float(r.elevation), "set": "confirmatory"}
        else:
            r = audit[audit.station_id == sid].iloc[0]
            H = sorted(int(x) for x in str(r.habitual_slots).split(";"))
            meta = {"H": H, "lat": float(r.lat), "lon": float(r.lon),
                    "elev": float(r.elevation), "set": "sensitivity",
                    "note": "preregistered N=38/N=39 sensitivity station"}
        df, hinfo, integrity = process_station(sid, meta)
        df.to_parquet(outp, index=False)
        hinfo["integrity_missing_slots"] = integrity
        hinfo["n_rows"] = len(df)
        hinfo["n_valid"] = int(df["valid"].sum())
        hinfo["n_valid_qc"] = int(df["valid_qc"].sum())
        with open(os.path.join(PARTS, f"meta_{sid}.json"), "w") as f:
            json.dump(hinfo, f, indent=2)
        print(f"{sid}: rows={len(df)} valid={hinfo['n_valid']} "
              f"valid_qc={hinfo['n_valid_qc']} miss={integrity}", flush=True)
    return 0


def merge_parts():
    parts = sorted(glob.glob(os.path.join(PARTS, "pair_*.parquet")))
    draw = json.load(open(PILOT_JSON))
    CONF = draw["confirmatory_stations"]
    dfs, metas = [], {}
    print("== equivalence check vs daily_obs_v2.parquet (confirmatory) ==")
    for p in parts:
        sid = os.path.basename(p)[5:-8]
        df = pd.read_parquet(p)
        if sid in CONF:
            mine = df[["date", "ws_obs", "n_slots", "valid"]].rename(
                columns={"ws_obs": "ws_daily"})
            probs, ncmp = equivalence_check(mine, sid)
            status = "OK" if not probs else "FAIL: " + "; ".join(probs)
            print(f"  {sid}: {ncmp} days -> {status}", flush=True)
            if probs:
                sys.exit(1)
        dfs.append(df)
        metas[sid] = json.load(open(os.path.join(PARTS, f"meta_{sid}.json")))
    all_df = pd.concat(dfs, ignore_index=True).sort_values(
        ["station_id", "date"]).reset_index(drop=True)
    all_df.to_parquet(os.path.join(OUT_DIR, "confirmatory_daily_paired.parquet"),
                      index=False)
    diag = {"params": {"window": "1995-2024", "n_confirmatory": len(CONF),
                       "n_sensitivity": len(SENS3), "sensitivity_stations": SENS3,
                       "rule": "prereg v1.2 sect.4-5 + D3/D4",
                       "executed": dt.datetime.now().isoformat()},
            "stations": metas,
            "totals": {"rows": len(all_df),
                       "valid_rows": int(all_df["valid"].sum()),
                       "valid_qc_rows": int(all_df["valid_qc"].sum()),
                       "integrity_missing_slots": int(sum(
                           m["integrity_missing_slots"] for m in metas.values()))}}
    with open(DIAG_JSON, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\nMERGED -> {OUT_DIR}/confirmatory_daily_paired.parquet")
    print(f"  rows={diag['totals']['rows']} valid={diag['totals']['valid_rows']} "
          f"valid_qc={diag['totals']['valid_qc_rows']} "
          f"integrity_missing={diag['totals']['integrity_missing_slots']}")


if __name__ == "__main__":
    sys.exit(main())
