#!/usr/bin/env python3
"""R9 closure: ancillary DEM covariates (Copernicus DEM GLO-90 via the
Open-Meteo elevation API, free, no key).

For every station in the M1b confirmatory run:
  - station lat/lon/elev (HadISD catalog)
  - CERRA nearest-cell centre + the 3x3 index neighbourhood centres
    (elev mean/min/max = within-cell-region relief at ~5.5 km spacing)
  - ERA5 bilinear 4 corner centres + nearest cell centre (relief over
    ~0.25 deg footprint)
  - delta-z = station elevation - DEM elevation at each product's location

Outputs:
  /tmp/my-project/verification/dem_covariates.json
  /tmp/my-project/verification/dem_covariates.csv
"""
import json
import os
import time

import numpy as np
import pandas as pd
import netCDF4 as nc

PARTS = '/tmp/my-project/M1b/parts'
CERRA_FILE = '/tmp/my-project/CERRA_data/cerra_1995_morocco.nc'
ERA5_FILE = '/tmp/my-project/ERA5_data/1995.nc'
OUT_JSON = '/tmp/my-project/verification/dem_covariates.json'
OUT_CSV = '/tmp/my-project/verification/dem_covariates.csv'
API = "https://api.open-meteo.com/v1/elevation"


def fetch_elevations(coords):
    """coords: list of (lat, lon) -> list of elevations (m)."""
    out = []
    for s in range(0, len(coords), 100):
        chunk = coords[s:s + 100]
        lats = ",".join(f"{c[0]:.5f}" for c in chunk)
        lons = ",".join(f"{c[1]:.5f}" for c in chunk)
        url = f"{API}?latitude={lats}&longitude={lons}"
        for attempt in range(5):
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode())
                out.extend(float(x) for x in data["elevation"])
                break
            except Exception as e:
                print(f"  chunk {s} attempt {attempt + 1} failed: {e}")
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"elevation API failed for chunk at {s}")
        time.sleep(0.6)
    return out


def main():
    metas = {}
    for f in sorted(os.listdir(PARTS)):
        if f.startswith("meta_") and f.endswith(".json"):
            sid = f[5:-5]
            metas[sid] = json.load(open(os.path.join(PARTS, f)))

    d = nc.Dataset(CERRA_FILE)
    la2d = np.ma.filled(np.ma.asarray(d.variables["latitude"][:], dtype="float64"), np.nan)
    lo2d = np.ma.filled(np.ma.asarray(d.variables["longitude"][:], dtype="float64"), np.nan)
    d.close()
    d = nc.Dataset(ERA5_FILE)
    lat1 = np.asarray(d.variables["latitude"][:], dtype="float64")
    lon1 = np.asarray(d.variables["longitude"][:], dtype="float64")
    d.close()

    def norm(lon):
        return lon - 360.0 if lon > 180 else lon

    coords, index = [], []

    def add(lat, lon, tag):
        coords.append((float(lat), float(norm(lon))))
        index.append(tag)

    per_station = {}
    for sid, m in metas.items():
        iy, ix = m["cerra_cell"]["iy"], m["cerra_cell"]["ix"]
        j0, i0 = m["era5_cell"]["j0"], m["era5_cell"]["i0"]
        jn, inn = m["era5_cell"]["jn"], m["era5_cell"]["inn"]
        tags = {}
        add(m["lat"], m["lon"], f"{sid}|station")
        tags["station"] = len(coords) - 1
        nbr = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = iy + dy, ix + dx
                add(la2d[yy, xx], lo2d[yy, xx], f"{sid}|cerra{dy}{dx}")
                nbr.append(len(coords) - 1)
        tags["cerra3x3"] = nbr
        tags["cerra_center"] = nbr[4]
        corners = []
        for (jj, ii) in ((j0, i0), (j0, i0 + 1), (j0 + 1, i0), (j0 + 1, i0 + 1)):
            add(lat1[jj], lon1[ii], f"{sid}|era5corner{jj}_{ii}")
            corners.append(len(coords) - 1)
        add(lat1[jn], lon1[inn], f"{sid}|era5near")
        tags["era5_corners"] = corners
        tags["era5_near"] = len(coords) - 1
        per_station[sid] = tags

    print(f"fetching {len(coords)} DEM points ...")
    elev = fetch_elevations(coords)
    print(f"  got {len(elev)} elevations")

    rows = []
    for sid, tg in per_station.items():
        m = metas[sid]
        e_st = elev[tg["station"]]
        c3 = np.array([elev[k] for k in tg["cerra3x3"]])
        e_cerra = elev[tg["cerra_center"]]
        ec = [elev[k] for k in tg["era5_corners"]]
        e_near = elev[tg["era5_near"]]
        rows.append({
            "station_id": sid, "lat": m["lat"], "lon": m["lon"],
            "elev_station": m["elev"], "dem_station": e_st,
            "dem_cerra_cell": e_cerra, "dz_cerra": round(m["elev"] - e_cerra, 1),
            "dem_cerra3x3_mean": round(float(c3.mean()), 1),
            "relief_cerra3x3": round(float(c3.max() - c3.min()), 1),
            "dem_era5_corners_mean": round(float(np.mean(ec)), 1),
            "relief_era5_corners": round(float(max(ec) - min(ec)), 1),
            "dem_era5_near": e_near, "dz_era5": round(m["elev"] - e_near, 1),
            "set": m["set"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    payload = {"source": "Copernicus DEM GLO-90 via Open-Meteo elevation API",
               "fetched": time.strftime("%Y-%m-%d"),
               "n_points": len(coords),
               "stations": {sid: {k: (elev[v] if isinstance(v, int) else
                                      [elev[k2] for k2 in v] if isinstance(v, list)
                                      else elev[v])
                                  for k, v in tg.items()}
                            for sid, tg in per_station.items()},
               "summary": df.to_dict(orient="records")}
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(df[["station_id", "elev_station", "dem_cerra_cell", "dz_cerra",
              "relief_cerra3x3", "dz_era5", "relief_era5_corners"]].to_string(
                  index=False))
    print(f"\n-> {OUT_CSV}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
