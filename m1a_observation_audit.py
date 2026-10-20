#!/usr/bin/env python3
"""M1a: obs-only station audit and final eligibility (rule v2).

NO reanalysis data are opened here. The only reanalysis artefacts used are
grid METADATA (domain bbox), i.e. the sampling frame, never reanalysis values.

Rule v2 (supersedes the old 6/8-slot rule; parameters marked PROPOSED are to
be frozen at pre-registration):
  - Window: 1995-01-01 .. 2024-12-31 UTC (common reanalysis period).
  - Wind-valid obs: windspeeds finite and >= 0. Units: m/s (attribute verified
    for all 306 files; NO conversion applied).
  - Active day: >= 1 wind-valid obs.
  - freq(h): fraction of active days with a wind-valid obs within +/-1 h of
    nominal hour h (circular 24 h).
  - Habitual slots H: freq(h) >= 0.75 AND local maximum (ties kept).
    PROPOSED: |H| >= 4 for eligibility.
  - Day validity: matched habitual slots >= ceil(0.75*|H|), AND >=1 matched
    nocturnal AND >=1 matched diurnal slot (LST = UTC + lon/15; diurnal if
    LST hour in [6,18)). Per matched slot: obs closest to nominal hour.
  - Eligible year: >= 274 valid days (75% of 365.25). PROPOSED.
  - Eligible station: >= 20 eligible years in window (proposal rule 4),
    plus |H| >= 4 with night+day coverage in H.
  - Domain frame (grid metadata only): lat 20.2923..40.0 N, lon -20..0 E
    (CERRA cut bbox INTERSECT ERA5 box).
Outputs:
  verification/m1a_station_audit.csv   (all in-frame stations, full metrics)
  verification/m1a_eligibility_v2.csv  (eligible subset)
  M1a/daily_obs_v2.parquet             (obs-only daily series, valid days)
"""
import csv
import datetime as dt
import json
import math
import os
import sys
import numpy as np
import netCDF4 as nc

DATA = "/tmp/my-project/HadISD_subdaily"
CATALOG = "/tmp/my-project/verification/hadisd_catalog.csv"
QUANT = "/tmp/my-project/verification/ws_quantization.csv"
OUT_DIR = "/tmp/my-project/verification"
DAILY_DIR = "/tmp/my-project/M1a"
os.makedirs(DAILY_DIR, exist_ok=True)

EPOCH = dt.datetime(1931, 1, 1)
W0 = dt.datetime(1995, 1, 1)
W1 = dt.datetime(2025, 1, 1)
LO = (W0 - EPOCH).total_seconds() / 3600.0
HI = (W1 - EPOCH).total_seconds() / 3600.0
LAT_MIN, LAT_MAX = 20.2923, 40.0
LON_MIN, LON_MAX = -20.0, 0.0
FREQ_THR = 0.75
MIN_SLOTS = 4
DAY_FRAC = 0.75
MIN_VALID_DAYS = 274
MIN_ELIG_YEARS = 20

DAILY_FIELDS = ["station_id", "date", "ws_daily", "n_slots", "night_ok", "day_ok", "valid"]
AUDIT_FIELDS = ["station_id", "lat", "lon", "elevation", "in_user42",
                "n_obs_total", "n_wind_valid", "n_active_days",
                "mean_obs_per_active_day", "frac_calm_eq0", "frac_calm_lt05",
                "frac_on_01_grid", "habitual_slots", "n_habitual",
                "freq_table", "n_valid_days", "coverage_days",
                "eligible_years", "n_eligible_years", "eligible_v2",
                "exclude_reason"]
ELIG_FIELDS = ["station_id", "lat", "lon", "elevation", "in_user42",
               "n_valid_days", "coverage_days", "n_eligible_years",
               "eligible_years_list", "habitual_slots", "mean_obs_per_active_day",
               "frac_calm_eq0", "frac_on_01_grid"]


def circ_dist(h1, h2):
    return np.abs(((h1 - h2 + 12.0) % 24.0) - 12.0)


def lst_hour(utc_h, lon):
    return (utc_h + lon / 15.0) % 24.0


def is_diurnal(lst):
    return 6.0 <= lst < 18.0


def main():
    # --- inputs ---
    cat = {r["station_id"]: r for r in csv.DictReader(open(CATALOG))}
    quant = {r["station_id"]: r for r in csv.DictReader(open(QUANT))}
    user42 = set()
    with open("/tmp/my-project/HadISD_data/stations_meta_qc.csv") as f:
        for r in csv.DictReader(f):
            user42.add(r["station_id"].split("-")[0])

    audit_rows, elig_rows = [], []
    daily_all = []
    excluded = []

    files = sorted(os.path.join(DATA, f) for f in os.listdir(DATA) if f.endswith(".nc"))
    for p in files:
        sid = os.path.basename(p)[:-3]
        c = cat.get(sid)
        if c is None:
            continue
        lat, lon, elev = float(c["lat"]), float(c["lon"]), float(c["elevation"])
        base = {"station_id": sid, "lat": lat, "lon": lon, "elevation": elev,
                "in_user42": sid in user42}

        # sampling frame (grid metadata only)
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            excluded.append({**base, "reason": "outside_common_domain"})
            continue

        ds = nc.Dataset(p)
        t = np.asarray(ds.variables["time"][:], dtype="float64")
        ws = np.asarray(ds.variables["windspeeds"][:], dtype="float64")
        ds.close()

        n_obs_total = int(t.size)
        m = (t >= LO) & (t < HI) & np.isfinite(ws) & (ws >= 0)
        t_h = t[m]
        w = ws[m]
        n_wind_valid = int(w.size)

        row = {**base, "n_obs_total": n_obs_total, "n_wind_valid": n_wind_valid,
               "n_active_days": 0, "mean_obs_per_active_day": 0.0,
               "frac_calm_eq0": "", "frac_calm_lt05": "",
               "frac_on_01_grid": quant.get(sid, {}).get("frac_on_01_grid", ""),
               "habitual_slots": "", "n_habitual": 0, "freq_table": "",
               "n_valid_days": 0, "coverage_days": 0.0,
               "eligible_years": "", "n_eligible_years": 0, "eligible_v2": False,
               "exclude_reason": ""}

        if n_wind_valid == 0:
            row["exclude_reason"] = "no_wind_valid_obs_in_window"
            audit_rows.append(row)
            excluded.append({**base, "reason": "no_wind_valid_obs_in_window"})
            continue

        hour = t_h % 24.0
        day_idx = np.floor(t_h / 24.0).astype("int64")
        active_days = np.unique(day_idx)
        n_active = active_days.size
        row["n_active_days"] = int(n_active)
        row["mean_obs_per_active_day"] = round(n_wind_valid / n_active, 3)
        row["frac_calm_eq0"] = round(float((w == 0.0).mean()), 5)
        row["frac_calm_lt05"] = round(float((w < 0.5).mean()), 5)

        # freq per nominal hour
        freqs = np.zeros(24)
        for h in range(24):
            mask = circ_dist(hour, float(h)) <= 1.0
            if mask.any():
                freqs[h] = np.unique(day_idx[mask]).size / n_active
        row["freq_table"] = ";".join(f"{h}:{freqs[h]:.2f}" for h in range(24)
                                     if freqs[h] > 0)

        # habitual slots: freq>=thr AND local max (ties kept, circular)
        H = []
        for h in range(24):
            if freqs[h] >= FREQ_THR:
                prev = freqs[(h - 1) % 24]
                nxt = freqs[(h + 1) % 24]
                if freqs[h] >= prev and freqs[h] >= nxt:
                    H.append(h)
        row["habitual_slots"] = ";".join(str(h) for h in H)
        row["n_habitual"] = len(H)

        # night/day split of habitual slots (LST at station lon)
        lst = {h: lst_hour(h, lon) for h in range(24)}
        H_night = [h for h in H if not is_diurnal(lst[h])]
        H_day = [h for h in H if is_diurnal(lst[h])]

        # per-slot matching -> daily series
        # BUGFIX 2026-10-17 (found by M1b pilot debugging): w[order] indexed
        # the GLOBAL wind array with subset indices, silently taking the first
        # n obs of the file instead of the slot-matched ones. w_sub = w[mask]
        # is the correct source array. Day masks / eligibility were
        # value-independent and are unaffected; only ws_daily changes.
        need = math.ceil(DAY_FRAC * len(H)) if H else 0
        slot_days = {}
        slot_vals = {}
        for h in H:
            mask = circ_dist(hour, float(h)) <= 1.0
            if not mask.any():
                continue
            d = day_idx[mask]
            dt_abs = np.abs(((t_h[mask] % 24.0) - h + 12.0) % 24.0 - 12.0)
            w_sub = w[mask]
            order = np.lexsort((dt_abs, d))
            d_s, dt_s, w_s = d[order], dt_abs[order], w_sub[order]
            uniq_days, first_idx = np.unique(d_s, return_index=True)
            slot_days[h] = uniq_days
            slot_vals[h] = w_s[first_idx]

        # aggregate matched slots per day
        per_day_slots = {}
        for h in H:
            for d, val in zip(slot_days.get(h, []), slot_vals.get(h, [])):
                per_day_slots.setdefault(d, []).append((h, val))

        valid_days_list = []
        for d, pairs in per_day_slots.items():
            slots_matched = [h for h, _ in pairs]
            night_ok = any(h in H_night for h in slots_matched)
            day_ok = any(h in H_day for h in slots_matched)
            ok = len(slots_matched) >= need and night_ok and day_ok
            vals = np.array([v for _, v in pairs])
            date = (EPOCH + dt.timedelta(days=int(d))).strftime("%Y-%m-%d")
            daily_all.append((sid, date, float(vals.mean()), len(slots_matched),
                              int(night_ok), int(day_ok), int(ok)))
            if ok:
                valid_days_list.append(d)

        valid_days_arr = np.array(sorted(valid_days_list), dtype="int64")
        row["n_valid_days"] = int(valid_days_arr.size)
        row["coverage_days"] = round(valid_days_arr.size / (10957), 4)  # 30y window days

        # eligible years
        if valid_days_arr.size:
            years = np.array([(EPOCH + dt.timedelta(days=int(d))).year
                              for d in valid_days_arr])
            yc = {}
            for y in years:
                yc[y] = yc.get(y, 0) + 1
            elig_years = sorted(y for y, cnt in yc.items() if cnt >= MIN_VALID_DAYS)
            row["eligible_years"] = ";".join(map(str, elig_years))
            row["n_eligible_years"] = len(elig_years)
            ok_station = (len(elig_years) >= MIN_ELIG_YEARS and len(H) >= MIN_SLOTS
                          and H_night and H_day)
            row["eligible_v2"] = bool(ok_station)
            if ok_station:
                elig_rows.append({
                    "station_id": sid, "lat": lat, "lon": lon, "elevation": elev,
                    "in_user42": base["in_user42"],
                    "n_valid_days": row["n_valid_days"],
                    "coverage_days": row["coverage_days"],
                    "n_eligible_years": len(elig_years),
                    "eligible_years_list": row["eligible_years"],
                    "habitual_slots": row["habitual_slots"],
                    "mean_obs_per_active_day": row["mean_obs_per_active_day"],
                    "frac_calm_eq0": row["frac_calm_eq0"],
                    "frac_on_01_grid": row["frac_on_01_grid"]})
        else:
            row["exclude_reason"] = "no_valid_days_under_v2_rule"
        audit_rows.append(row)
        if (len(audit_rows)) % 25 == 0:
            print(f"  ... {len(audit_rows)} in-frame stations processed", flush=True)

    # --- write outputs ---
    with open(os.path.join(OUT_DIR, "m1a_station_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(audit_rows)
    with open(os.path.join(OUT_DIR, "m1a_eligibility_v2.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ELIG_FIELDS)
        w.writeheader()
        w.writerows(elig_rows)
    import pandas as pd
    pd.DataFrame(daily_all, columns=DAILY_FIELDS).to_parquet(
        os.path.join(DAILY_DIR, "daily_obs_v2.parquet"), index=False)

    # --- summary ---
    elig_ids = {r["station_id"] for r in elig_rows}
    in42 = {r["station_id"] for r in audit_rows if r["in_user42"]}
    print("\n== M1a SUMMARY (obs-only, rule v2) ==")
    print(f"in-frame stations audited : {len(audit_rows)}")
    print(f"excluded (outside frame)  : {len(excluded)}")
    print(f"ELIGIBLE stations v2      : {len(elig_ids)}")
    print(f"user-42 inside frame      : {len(in42)}")
    print(f"user-42 eligible          : {len(in42 & elig_ids)}")
    print(f"user-42 NOT eligible      : {sorted(in42 - elig_ids)}")
    print(f"new eligible outside 42   : {sorted(elig_ids - in42)}")
    summary = {
        "in_frame": len(audit_rows), "eligible": len(elig_ids),
        "user42_in_frame": len(in42),
        "user42_eligible": sorted(in42 & elig_ids),
        "user42_not_eligible": sorted(in42 - elig_ids),
        "eligible_beyond_42": sorted(elig_ids - in42),
        "params": {"window": "1995-2024", "freq_thr": FREQ_THR,
                   "min_slots": MIN_SLOTS, "day_frac": DAY_FRAC,
                   "min_valid_days": MIN_VALID_DAYS,
                   "min_elig_years": MIN_ELIG_YEARS,
                   "frame": "lat 20.2923-40, lon -20..0 (grid metadata only)"},
    }
    with open(os.path.join(OUT_DIR, "m1a_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
