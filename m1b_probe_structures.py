"""M1b probe: dump the exact structures M1b needs (metadata + schema only).
No reanalysis VALUES are used for any decision here - structure inspection for code writing.
"""
import pandas as pd
import numpy as np
import netCDF4 as nc

print("=== eligibility v2 ===")
el = pd.read_csv('/tmp/my-project/verification/m1a_eligibility_v2.csv', dtype={'station_id': str})
print("columns:", list(el.columns))
pilots = ['601850', '601900', '605170', '605180', '605600']
sub = el[el['station_id'].isin(pilots)]
print(sub[['station_id', 'lat', 'lon', 'elevation', 'n_valid_days', 'n_eligible_years', 'habitual_slots']].to_string())
print("n_rows:", len(el))

print("\n=== daily_obs_v2.parquet ===")
d = pd.read_parquet('/tmp/my-project/M1a/daily_obs_v2.parquet')
print("columns:", list(d.columns))
print("dtypes:", {k: str(v) for k, v in d.dtypes.items()})
print(d.head(4).to_string())
print("rows:", len(d))
if 'station_id' in d.columns:
    print("n stations in parquet:", d['station_id'].astype(str).nunique())

print("\n=== CERRA 2020 structure (metadata) ===")
c = nc.Dataset('/tmp/my-project/CERRA_data/cerra_2020_morocco.nc')
print("dims:", {k: len(v) for k, v in c.dimensions.items()})
for name, v in c.variables.items():
    keep = {}
    for a in v.ncattrs():
        if a in ('units', 'standard_name', 'long_name', 'grid_mapping_name'):
            keep[a] = str(v.getncattr(a))[:60]
    print(f"  {name}: {v.dtype} {v.dimensions} {keep}")
ga = {a: str(c.getncattr(a))[:90] for a in c.ncattrs() if a.lower() not in ('history', 'comment')}
print("global attrs:", ga)
c.close()

print("\n=== ERA5 2020 structure (metadata) ===")
e = nc.Dataset('/tmp/my-project/ERA5_data/2020.nc')
print("dims:", {k: len(v) for k, v in e.dimensions.items()})
for name, v in e.variables.items():
    u = v.getncattr('units') if 'units' in v.ncattrs() else ''
    print(f"  {name}: {v.dtype} {v.dimensions} units={u}")
la = e.variables['latitude'][:]
lo = e.variables['longitude'][:]
print("lat:", float(la.min()), float(la.max()), "asc?", bool((np.diff(la) > 0).all()))
print("lon:", float(lo.min()), float(lo.max()), "asc?", bool((np.diff(lo) > 0).all()))
e.close()

print("\n=== HadISD pilot 601850 structure ===")
h = nc.Dataset('/tmp/my-project/HadISD_subdaily/601850.nc')
print("dims:", {k: len(v) for k, v in h.dimensions.items()})
for name, v in h.variables.items():
    keep = {}
    for a in v.ncattrs():
        if a in ('units', 'standard_name', 'long_name', '_FillValue'):
            keep[a] = str(v.getncattr(a))[:50]
    print(f"  {name}: {v.dtype} {v.dimensions} {keep}")
for tname in ('epoch', 'timestamp', 'report_timestamp'):
    if tname in h.variables:
        t = h.variables[tname]
        print(f"{tname} first/last raw:", float(t[0]), float(t[-1]), "units:", t.getncattr('units'))
h.close()
print("\nPROBE DONE")
