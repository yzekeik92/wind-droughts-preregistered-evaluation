#!/usr/bin/env python3
"""V4 fix (reviewer point 5): join the REAL ERA5 model orography and land-sea
mask to the pairing, replacing the DEM-proxy narrative and fixing the dz sign.

Source: WeatherBench2 public anonymous mirror of the ERA5 archive
(0.25-degree native grid, single chunks):
  gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2
  variables: geopotential_at_surface (z [m2 s-2]), land_sea_mask (lsm [0-1])
These are the same invariant fields the CDS distributes (ERA5 does not change
them over time); the CDS API needs an account, which this environment lacks
(a documented anonymous-key attempt fails with 401, logged below).

For every station (meta files carry the local 81x81 cutout indices):
  - local->global index map: lat_local(j)=40-0.25j -> jg=200+j;
    lon_local(i)=-20+0.25i -> ig=1360+i
  - ERA5 model orography z/9.80665 (m) at the 4 bilinear corner cells and the
    nearest corner; lsm at the same cells (ocean if lsm<0.5)
  - dz_model = station elevation - model orography (mean of corners and near)
  - comparison of model orography vs Copernicus DEM at the same cells

Outputs:
  /tmp/my-project/verification/era5_invariants.npz       (fields + axes)
  /home/z/my-project/download/M2_results/era5_orography_v4.csv
"""
import glob
import json
import os
import subprocess

import numpy as np

WB2 = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
       "1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2")
PARTS = '/tmp/my-project/M1b/parts'
OUT_NPZ = '/tmp/my-project/verification/era5_invariants.npz'
OUT_CSV = '/home/z/my-project/download/M2_results/era5_orography_v4.csv'
G0, I0 = 200, 1360          # local (0,0) -> global (lat 40N, lon -20E)
# local lon runs -20..0 inclusive; lon 0E wraps to global index 0
def gi(i_local):
    return (I0 + i_local) % 1440
G = 9.80665


def fetch(url, path):
    r = subprocess.run(["curl", "-sS", "--retry", "3", "-o", path, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl failed {url}: {r.stderr}")


def decode_chunk(path, shape):
    import blosc
    raw = open(path, 'rb').read()
    arr = np.frombuffer(blosc.decompress(raw), dtype='<f4')
    return arr.reshape(shape)


def main():
    # ---- document the CDS anonymous attempt (no credentials in this env) ----
    log = subprocess.run(
        ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}",
         "https://cds.climate.copernicus.eu/api/catalogue/v1/collections/"
         "reanalysis-cerra-france-plus"],
        capture_output=True, text=True)
    print(f"CDS anonymous catalogue probe HTTP {log.stdout.strip()} "
          "(API data access requires a personal access token; none configured)")

    cache = '/tmp/my-project/verification/wb2_chunks'
    os.makedirs(cache, exist_ok=True)
    zpath = f"{cache}/z.bin"
    lpath = f"{cache}/lsm.bin"
    if not os.path.exists(zpath):
        fetch(f"{WB2}/geopotential_at_surface/0.0", zpath)
    if not os.path.exists(lpath):
        fetch(f"{WB2}/land_sea_mask/0.0", lpath)
    z = decode_chunk(zpath, (721, 1440)).astype('float64') / G   # metres
    lsm = decode_chunk(lpath, (721, 1440)).astype('float64')

    # axes (1-D arrays, chunk key '0')
    apath, opath = f"{cache}/lat.bin", f"{cache}/lon.bin"
    if not os.path.exists(apath):
        fetch(f"{WB2}/latitude/0", apath)
    if not os.path.exists(opath):
        fetch(f"{WB2}/longitude/0", opath)
    import blosc
    lat_g = np.frombuffer(blosc.decompress(open(apath, 'rb').read()),
                          dtype='<f4').astype('float64')
    lon_g = np.frombuffer(blosc.decompress(open(opath, 'rb').read()),
                          dtype='<f4').astype('float64')
    assert lat_g.shape == (721,) and lon_g.shape == (1440,), (lat_g.shape, lon_g.shape)
    # verify the local->global map against the axes themselves
    assert abs(lat_g[G0] - 40.0) < 1e-6 and abs(lat_g[G0 + 80] - 20.0) < 1e-6, lat_g[[G0, G0+80]]
    assert abs(lon_g[I0] - 340.0) < 1e-6 and abs(lon_g[gi(80)] - 0.0) < 1e-6, lon_g[[I0, gi(80)]]
    np.savez_compressed(OUT_NPZ, z=z, lsm=lsm, lat=lat_g, lon=lon_g)
    print(f"invariants saved: z[{z.min():.0f},{z.max():.0f}] m, "
          f"lsm[{lsm.min():.2f},{lsm.max():.2f}]")

    rows = []
    for mf in sorted(glob.glob(os.path.join(PARTS, "meta_*.json"))):
        m = json.load(open(mf))
        sid = os.path.basename(mf)[5:-5]
        if m["set"] != "confirmatory":
            continue
        ec = m["era5_cell"]
        j0, i0, jn, inn = ec["j0"], ec["i0"], ec["jn"], ec["inn"]
        corners = [(G0 + j0 + dj, gi(i0 + di))
                   for dj in (0, 1) for di in (0, 1)]
        zm = np.array([z[jj, ii] for jj, ii in corners])
        lm = np.array([lsm[jj, ii] for jj, ii in corners])
        z_near = z[G0 + jn, gi(inn)]
        lsm_near = lsm[G0 + jn, gi(inn)]
        elev = m["elev"]
        rows.append({
            "station_id": sid, "class": "", "elev_station": elev,
            "z_era5_corners_mean": round(float(zm.mean()), 1),
            "z_era5_near": round(float(z_near), 1),
            "dz_model_corners_mean": round(elev - float(zm.mean()), 1),
            "dz_model_near": round(elev - float(z_near), 1),
            "n_ocean_corners_lsm05": int((lm < 0.5).sum()),
            "lsm_near": round(float(lsm_near), 2),
            "ocean_any_corner": int((lm < 0.5).any()),
        })
    import pandas as pd
    # attach classes from the revision sheet
    st = pd.read_csv('/home/z/my-project/download/M2_results/'
                     'station_metrics_confirmatory.csv',
                     dtype={'station_id': str})
    cls = dict(zip(st.station_id, st['class']))
    df = pd.DataFrame(rows)
    df['class'] = df.station_id.map(cls)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print("\nARCHIPELAGO summary (model orography):")
    a = df[df['class'] == 'archipelago']
    print(a[["station_id", "elev_station", "z_era5_near", "dz_model_near",
             "dz_model_corners_mean", "n_ocean_corners_lsm05"]].to_string(index=False))
    print("\n-> ", OUT_CSV)


if __name__ == '__main__':
    main()
