#!/usr/bin/env python3
"""V5 fix (reviewer point 10): direct download of the Copernicus DEM GLO-90
(replacing the Open-Meteo elevation API as the access route) and re-derivation
of the DEM covariates used for the dz/terrain-context diagnostics.

Source: Microsoft Planetary Computer public mirror of Copernicus DEM GLO-90
(collection cop-dem-glo-90; anonymous SAS token, no registration), which
redistributes the same ESA product (dataset DOI 10.5270/ESA-c5d3b65) that the
paper cites. Tiles are 1x1-degree COGs at 90 m.

For every station (dem_covariates.csv, all sets):
  - DEM bilinear at the station point            -> dem_station_v5
  - DEM bilinear at the CERRA paired-cell centre -> dem_cerra_cell_v5, dz_cerra_v5
  - 3x3 90-m window around the CERRA cell centre -> dem_cerra3x3_mean_v5, relief_v5
Validation against the v3/Open-Meteo sheet (dem_covariates.csv): per-field
max |diff| is reported; agreement within a few metres is expected (same
underlying product, independent access route and resampling).

Outputs:
  /tmp/my-project/verification/dem_glo90/            (tile cache)
  /home/z/my-project/download/M2_results/dem_covariates_v5.csv
  /home/z/my-project/download/M2_results/dem_direct_validation_v5.json
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

DEM_CSV = '/tmp/my-project/verification/dem_covariates.csv'
PARTS = '/tmp/my-project/M1b/parts'
CACHE = '/tmp/my-project/verification/dem_glo90'
RES = '/home/z/my-project/download/M2_results'
PC_TOKEN = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/cop-dem-glo-90'
PC_ASSET = ('https://elevationeuwest.blob.core.windows.net/copernicus-dem/'
            'COP90_hh/Copernicus_DSM_COG_30_{tile}_DEM.tif{sas}')


def tile_name(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f'{ns}{abs(int(np.floor(lat))):02d}_00_{ew}{abs(int(np.floor(lon))):03d}_00'


def fetch(url, path):
    r = subprocess.run(['curl', '-sS', '--retry', '3', '--max-time', '300',
                        '-o', path, url], capture_output=True, text=True)
    ok = r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 10000
    if not ok and os.path.exists(path):
        os.remove(path)
    return ok


def sample_bilinear(src, x, y):
    v = next(src.sample([(x, y)]))
    return float(v[0])


def main():
    os.makedirs(CACHE, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    dem = pd.read_csv(DEM_CSV, dtype={'station_id': str})
    metas = {}
    for mf in sorted(os.listdir(PARTS)):
        if mf.startswith('meta_') and mf.endswith('.json'):
            sid = mf[5:-5]
            m = json.load(open(os.path.join(PARTS, mf)))
            if m.get('set') in ('confirmatory', 'sensitivity', 'pilot'):
                metas[sid] = m
    dem = dem[dem.station_id.isin(metas)].copy()

    # tile set covering stations and CERRA cell centres
    pts = [(r.lat, r.lon) for r in dem.itertuples()]
    pts += [(metas[r.station_id]['cerra_cell']['cell_lat'],
             metas[r.station_id]['cerra_cell']['cell_lon']) for r in dem.itertuples()]
    tiles = sorted({tile_name(la, lo) for la, lo in pts})
    print(f'{len(pts)} points -> {len(tiles)} tiles: {tiles}')

    tok = json.loads(subprocess.run(['curl', '-sS', '--max-time', '60', PC_TOKEN],
                                    capture_output=True, text=True).stdout)['token']
    sas = f'?{tok}'
    paths, missing = {}, []
    for t in tiles:
        p = os.path.join(CACHE, t + '.tif')
        if not os.path.exists(p):
            ok = fetch(PC_ASSET.format(tile=t, sas=sas), p)
            print(' ', t, 'OK' if ok else 'MISSING (ocean-only tile or gap)')
            if not ok:
                missing.append(t)
                paths[t] = None
                continue
        paths[t] = p

    open_tiles = {t: (rasterio.open(p) if p else None) for t, p in paths.items()}

    def pick(lat, lon):
        t = tile_name(lat, lon)
        if open_tiles.get(t) is None:
            return None
        return open_tiles[t]

    rows = []
    for r in dem.itertuples():
        m = metas[r.station_id]
        cc = m['cerra_cell']
        lat, lon = r.lat, r.lon
        src = pick(lat, lon)
        dem_station = sample_bilinear(src, lon, lat) if src is not None else np.nan
        srcc = pick(cc['cell_lat'], cc['cell_lon'])
        if srcc is not None:
            dem_cerra = sample_bilinear(srcc, cc['cell_lon'], cc['cell_lat'])
            row, col = rasterio.transform.rowcol(srcc.transform, cc['cell_lon'],
                                                 cc['cell_lat'])
            win = Window(col - 1, row - 1, 3, 3).round_offsets().round_lengths()
            arr = srcc.read(1, window=win).astype('float64')
            m3, rel = float(arr.mean()), float(arr.max() - arr.min())
        else:
            dem_cerra, m3, rel = np.nan, np.nan, np.nan
        rows.append({
            'station_id': r.station_id, 'lat': lat, 'lon': lon,
            'elev_station': r.elev_station, 'set': r.set,
            'dem_station_v5': float(dem_station),
            'dem_cerra_cell_v5': float(dem_cerra),
            'dz_cerra_v5': float(r.elev_station - dem_cerra),
            'dem_cerra3x3_mean_v5': m3,
            'relief_cerra3x3_v5': rel,
        })
    v5 = pd.DataFrame(rows)
    v5.to_csv(os.path.join(RES, 'dem_covariates_v5.csv'), index=False)

    val = {}
    for a, b, name in (('dem_station_v5', 'dem_station', 'dem_station'),
                       ('dem_cerra_cell_v5', 'dem_cerra_cell', 'dem_cerra_cell'),
                       ('dz_cerra_v5', 'dz_cerra', 'dz_cerra'),
                       ('dem_cerra3x3_mean_v5', 'dem_cerra3x3_mean', 'dem3x3_mean'),
                       ('relief_cerra3x3_v5', 'relief_cerra3x3', 'relief')):
        pair = v5[['station_id', a]].dropna()
        ref = dem.set_index('station_id').loc[pair.station_id, b]
        d = (pair[a].to_numpy() - ref.to_numpy()).__abs__()
        val[name] = {'max_abs_diff_m': float(d.max()),
                     'median_abs_diff_m': float(d.median()),
                     'n_compared': int(len(d))}
    conf = v5[(v5.set == 'confirmatory') & v5.dz_cerra_v5.notna()]
    summary = {
        'source': 'Copernicus DEM GLO-90 via Microsoft Planetary Computer public '
                  'mirror (collection cop-dem-glo-90, anonymous SAS), direct COG '
                  'download; dataset DOI 10.5270/ESA-c5d3b65',
        'n_tiles': len(tiles), 'n_tiles_missing': len(missing),
        'missing_tiles': missing,
        'n_stations': len(v5),
        'stations_without_tile': v5[v5.dem_cerra_cell_v5.isna()].station_id.tolist(),
        'validation_vs_openmeteo_v3': val,
        'confirmatory_median_dz_cerra_v5': float(conf.dz_cerra_v5.median()),
        'confirmatory_median_dz_cerra_v3': float(
            dem[dem.set == 'confirmatory'].set_index('station_id')
            .loc[conf.station_id, 'dz_cerra'].median()),
    }
    with open(os.path.join(RES, 'dem_direct_validation_v5.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
