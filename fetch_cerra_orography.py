#!/usr/bin/env python3
"""Ready-to-run CDS join for the CERRA surface-height invariant (v5, point 10).

The manuscript's CERRA terrain context is DEM-based because the analysis
environment had no CDS credentials (an anonymous-key attempt is logged in
fetch_era5_invariants.py). The CDS account is free; with a personal access
token (https://cds.climate.copernicus.eu/how-to-api) this script completes
the join in one command: it downloads the CERRA invariants (surface
geopotential, land--sea mask) over the study domain and writes per-station
CERRA model orography and land--sea fraction at the paired cell centres, in
the same shape as era5_orography_v4.csv so the class medians of Table 1 can
be recomputed for CERRA exactly as they are computed for ERA5.

Usage:
    export CDSAPI_TOKEN=<your-personal-access-token>
    python3 fetch_cerra_orography.py [stations.csv]

stations.csv (optional): columns station_id,lat,lon,elev_station (defaults
to the confirmatory stations of station_metrics_confirmatory.csv).

Dataset notes (checked 2026-10-18):
  - CERRA invariants: dataset "reanalysis-cerra-single-levels" (data_type
    "invariants"; if your catalogue view exposes the CERRA-France production
    instead, try "reanalysis-cerra-france-plus"), variables "geopotential"
    (m2 s-2 -> divide by 9.80665) and "land_sea_mask". The CERRA UERRA-region
    grid (0.05 deg) covers the Moroccan/Canary domain of the study.
  - Adjust DATASET/VARIABLE names if the CDS migration renames them.
Output: cerra_orography_v5.csv in the working directory.
"""
import os
import sys

import pandas as pd

DATASET = 'reanalysis-cerra-single-levels'
VARIABLES = ['geopotential', 'land_sea_mask']
YEAR = '2024'          # invariants are static; any year works
OUT = 'cerra_orography_v5.csv'
G = 9.80665


def main():
    try:
        import cdsapi
    except ImportError:
        sys.exit('pip install cdsapi')
    token = (os.environ.get('CDSAPI_TOKEN') or os.environ.get('CDSEA_API_TOKEN')
             or os.environ.get('CDSAPI_KEY'))
    if not token:
        sys.exit('Set CDSAPI_TOKEN=<personal access token> (free at '
                 'https://cds.climate.copernicus.eu/how-to-api).')
    c = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=token,
                      quiet=True)
    target = 'cerra_invariants.nc'
    print('Requesting', DATASET, '...')
    c.retrieve(DATASET, {
        'data_type': 'invariants',
        'variable': VARIABLES,
        'year': [YEAR],
        'month': ['01'],
        'day': ['01'],
        'time': ['00:00'],
        'area': [36.5, -19.0, 27.0, 1.0],   # N, W, S, E over the domain
        'format': 'netcdf',
    }, target)

    import xarray as xr
    ds = xr.open_dataset(target)
    print('Fields:', list(ds.data_vars))
    stations = sys.argv[1] if len(sys.argv) > 1 else (
        '/home/z/my-project/download/M2_results/station_metrics_confirmatory.csv')
    st = pd.read_csv(stations, dtype={'station_id': str})
    rows = []
    zname = 'z' if 'z' in ds.data_vars else ('geopotential' if 'geopotential' in ds.data_vars
                                             else list(ds.data_vars)[0])
    lname = 'lsm' if 'lsm' in ds.data_vars else None
    for r in st.itertuples():
        p = ds.sel(latitude=r.lat, longitude=r.lon, method='nearest')
        p0 = p.isel(time=0) if 'time' in p.dims else p
        zz = float(p0[zname].values) / G
        lm = float(p0[lname].values) if lname else float('nan')
        rows.append({'station_id': r.station_id,
                     'elev_station': getattr(r, 'elev_station', float('nan')),
                     'z_cerra_model_m': zz,
                     'dz_cerra_model': getattr(r, 'elev_station', zz) - zz,
                     'lsm_cerra': lm,
                     'cell_lat': float(p0.latitude),
                     'cell_lon': float(p0.longitude)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print('wrote', OUT)
    if 'class' in st.columns:
        out = out.merge(st[['station_id', 'class']], on='station_id')
        print(out.groupby('class').dz_cerra_model.median())
    return 0


if __name__ == '__main__':
    sys.exit(main())
