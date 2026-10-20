"""Debug: why is obs-vs-model daily correlation ~0 in the pilot merge?
1) anchor check of HadISD 'time' decode (num2date vs floor(t/24) formula)
2) cross-correlation of paired series at day offsets -2..+2
3) raw spot-check of one date against CERRA / ERA5 files
"""
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc

SID = '601850'
LAT, LON = 32.283, -9.233

df = pd.read_parquet('/tmp/my-project/M1b/pilot_daily_paired.parquet')
g = df[df.station_id == SID].copy()
print("=== paired head ===")
print(g.head(8).to_string())
print("\n=== describe ===")
print(g[['ws_obs', 'ws_cerra', 'ws_era5', 'cerra_all8', 'era5_all24']].describe().to_string())

print("\n=== 1) HadISD time anchor check ===")
ds = nc.Dataset(f'/tmp/my-project/HadISD_subdaily/{SID}.nc')
t = np.asarray(ds.variables['time'][:], dtype='float64')
units = ds.variables['time'].units
print("units:", units)
d0 = nc.num2date(t[:3], units, only_use_cftime_datetimes=False)
print("num2date first 3:", d0)
EPOCH = dt.datetime(1931, 1, 1)
mine = [EPOCH + dt.timedelta(hours=float(x)) for x in t[:3]]
print("formula first 3:", mine)
twin = t[(t >= (dt.datetime(1995, 1, 1) - EPOCH).total_seconds() / 3600)]
first_w = EPOCH + dt.timedelta(hours=float(twin[0]))
print("first window obs:", first_w)
ds.close()

print("\n=== 2) cross-correlation at day offsets ===")
g['date'] = pd.to_datetime(g['date'])
base = g.set_index('date')['ws_obs']
for prod in ('ws_cerra', 'ws_era5'):
    s = g.set_index('date')[prod]
    row = []
    for off in range(-2, 3):
        a = base.shift(0, freq='D')
        b = s.shift(off, freq='D')
        j = pd.concat([a, b], axis=1, join='inner').dropna()
        row.append(f"{off:+d}:{np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]:+.3f}")
    print(f"  {SID} {prod}: " + "  ".join(row))

print("\n=== 3) raw spot-check 2020-06-15 (601850) ===")
day70 = (dt.date(2020, 6, 15) - dt.date(1970, 1, 1)).days
H = [5, 7, 11, 12, 17, 18]
mapped = {h: 3 * ((h + 1) // 3) for h in H}
c = nc.Dataset('/tmp/my-project/CERRA_data/cerra_2020_morocco.nc')
vt = np.asarray(c.variables['valid_time'][:], dtype='int64')
la = c.variables['latitude'][:]
lo = c.variables['longitude'][:]
R = 6371.0088
p1, l1 = np.radians(LAT), np.radians(LON)
p2, l2 = np.radians(la), np.radians(lo)
dl = (l2 - l1 + np.pi) % (2 * np.pi) - np.pi
a_ = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
dist = 2 * R * np.arcsin(np.sqrt(a_))
iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
col = np.ma.filled(c.variables['si10'][:, iy, ix], np.nan)
sel = np.where(vt // 86400 == day70)[0]
print("CERRA that day:", {int((vt[k] % 86400) // 3600): round(float(col[k]), 2) for k in sel})
print("mapped slots needed:", {h: mapped[h] for h in H})
c.close()
e = nc.Dataset('/tmp/my-project/ERA5_data/2020.nc')
vt2 = np.asarray(e.variables['valid_time'][:], dtype='int64')
lat = np.asarray(e.variables['latitude'][:], dtype='float64')
lon = np.asarray(e.variables['longitude'][:], dtype='float64')
jf = (lat[0] - LAT) / (lat[0] - lat[1]); j0 = int(np.floor(jf)); wy = jf - j0
if_ = (LON - lon[0]) / (lon[1] - lon[0]); i0 = int(np.floor(if_)); wx = if_ - i0
u = np.ma.filled(e.variables['u10'][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype('float64')
v = np.ma.filled(e.variables['v10'][:, j0:j0 + 2, i0:i0 + 2], np.nan).astype('float64')
uint = (1 - wy) * ((1 - wx) * u[:, 0, 0] + wx * u[:, 0, 1]) + wy * ((1 - wx) * u[:, 1, 0] + wx * u[:, 1, 1])
vint = (1 - wy) * ((1 - wx) * v[:, 0, 0] + wx * v[:, 0, 1]) + wy * ((1 - wx) * v[:, 1, 0] + wx * v[:, 1, 1])
ws = np.hypot(uint, vint)
sel2 = np.where(vt2 // 86400 == day70)[0]
print("ERA5 that day (hour:ws):", {int((vt2[k] % 86400) // 3600): round(float(ws[k]), 2) for k in sel2})
print("ERA5 j0,i0,wy,wx:", j0, i0, round(wy, 3), round(wx, 3), "cell lats:", lat[j0], lat[j0 + 1])
e.close()
row = g[g.date == '2020-06-15']
print("paired row:", row[['date', 'ws_obs', 'ws_cerra', 'ws_era5', 'slots']].to_dict('records'))
