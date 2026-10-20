"""Timing probe: cost of reading one station column from a CERRA year file."""
import time
import numpy as np
import netCDF4 as nc

P = '/tmp/my-project/CERRA_data/cerra_2020_morocco.nc'
d = nc.Dataset(P)
t0 = time.time()
la = d.variables['latitude'][:]
lo = d.variables['longitude'][:]
print(f"lat/lon 2D read: {time.time()-t0:.2f} s, shape {la.shape}")

lat0, lon0 = 32.283, -9.233  # pilot 601850
R = 6371.0088
p1, l1 = np.radians(lat0), np.radians(lon0)
p2, l2 = np.radians(la), np.radians(lo)
a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
dist = 2 * R * np.arcsin(np.sqrt(a))
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
print(f"nearest cell (iy,ix)=({iy},{ix})  d={float(dist[iy, ix]):.3f} km  "
      f"cell lat/lon=({float(la[iy, ix]):.4f},{float(lo[iy, ix]):.4f})")

v = d.variables['si10']
print("si10 attrs:", {a: str(v.getncattr(a))[:40] for a in v.ncattrs()})
try:
    print("chunking:", v.chunking())
except Exception as e:
    print("chunking n/a:", e)

t0 = time.time()
col = v[:, iy, ix]
dt1 = time.time() - t0
print(f"column read (auto mask/scale ON): {dt1:.2f} s, {col.shape}, "
      f"min={float(np.min(col)):.2f}, max={float(np.max(col)):.2f}")

t0 = time.time()
for k in range(4):  # simulate 5 stations on the same file
    iy2 = (iy + 7 * k) % 412
    ix2 = (ix + 11 * k) % 409
    _ = v[:, iy2, ix2]
print(f"4 more columns: {time.time()-t0:.2f} s total")
vt = d.variables['valid_time'][:]
ts = vt[:6].astype('datetime64[s]')
print("valid_time head:", ts, "... n =", len(vt))
d.close()
print("TIMING PROBE DONE")
