"""Debug 2: (a) CERRA vs ERA5 mutual correlation in the paired table;
(b) raw HadISD obs dump for 601850 on sample days to see what the station
actually reported (timestamps + speeds).
"""
import datetime as dt
import numpy as np
import pandas as pd
import netCDF4 as nc

df = pd.read_parquet('/tmp/my-project/M1b/pilot_daily_paired.parquet')
print("=== mutual correlation (all pilots) ===")
for sid, gg in df.groupby('station_id'):
    gg = gg[gg.valid == 1]
    print(f"  {sid}: corr(cerra,era5)={np.corrcoef(gg.ws_cerra, gg.ws_era5)[0, 1]:.3f}  "
          f"corr(cerra8,era5_24)={np.corrcoef(gg.cerra_all8, gg.era5_all24)[0, 1]:.3f}  "
          f"n={len(gg)}")

SID = '601850'
EPOCH = dt.datetime(1931, 1, 1)
ds = nc.Dataset(f'/tmp/my-project/HadISD_subdaily/{SID}.nc')
t = np.asarray(ds.variables['time'][:], dtype='float64')
w = np.asarray(ds.variables['windspeeds'][:], dtype='float64')
ds.close()

for day in ('1995-01-03', '2020-06-15'):
    d0 = dt.datetime.strptime(day, '%Y-%m-%d')
    lo = (d0 - EPOCH).total_seconds() / 3600.0
    hi = lo + 24.0
    m = (t >= lo) & (t < hi)
    print(f"\n=== raw obs {SID} {day} ({int(m.sum())} reports) ===")
    for tt, ww in zip(t[m], w[m]):
        ts = EPOCH + dt.timedelta(hours=float(tt))
        print(f"  {ts:%Y-%m-%d %H:%M}  {ww:6.2f}")

# day before/after sanity for 2020-06-15 window and slot stats
print("\n=== slot availability stats 601850, 2020-06-01..30 ===")
lo0 = (dt.datetime(2020, 6, 1) - EPOCH).total_seconds() / 3600.0
m = (t >= lo0) & (t < lo0 + 30 * 24)
hh = t[m] % 24
cnt = np.histogram(hh, bins=np.arange(25))[0]
print("reports per UTC hour:", {h: int(c) for h, c in enumerate(cnt) if c})
