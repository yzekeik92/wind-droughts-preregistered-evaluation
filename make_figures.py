#!/usr/bin/env python3
"""Paper figures F1-F9 (English, publication style).

Outputs vector PDF (+PNG preview) to /home/z/my-project/download/M2_results/figures/
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

RES = '/home/z/my-project/download/M2_results'
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)
PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
CERRA_FILE = '/tmp/my-project/CERRA_data/cerra_1995_morocco.nc'
ERA5_FILE = '/tmp/my-project/ERA5_data/1995.nc'

C_CERRA, C_ERA5, C_OBS = "#0072B2", "#D55E00", "#009E73"
CLASSES = ['archipelago', 'atlantic', 'strait_med', 'interior']
C_CLS = {'archipelago': '#4477AA', 'atlantic': '#66CCEE',
         'strait_med': '#228833', 'interior': '#CCBB44'}

plt.rcParams.update({
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'))
    plt.close(fig)
    print('saved', name)


def runs(low, day_int, kmin=5):
    if len(low) == 0:
        return []
    brk = np.r_[True, np.diff(day_int) != 1, True]
    ss = np.flatnonzero(brk)[:-1]
    ee = np.flatnonzero(brk)[1:]
    out = []
    for s, e in zip(ss, ee):
        seg = low[s:e]
        if not seg.any():
            continue
        d = np.diff(np.r_[0, seg.astype('int8'), 0])
        st = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        out.extend([(int(a), int(b)) for a, b in zip(st + s, en + s)])
    return out


# ------------------------------------------------------------------ load data
st = pd.read_csv(os.path.join(RES, 'station_metrics_confirmatory.csv'),
                 dtype={'station_id': str})
_dem = pd.read_csv('/tmp/my-project/verification/dem_covariates.csv',
                   dtype={'station_id': str})[['station_id', 'lat', 'lon']]
st = st.merge(_dem, on='station_id', how='left')
h2 = pd.read_csv(os.path.join(RES, 'h2_spell_duration.csv'), dtype={'station_id': str})
h3 = pd.read_csv(os.path.join(RES, 'h3_annual_ratios.csv'), dtype={'station_id': str})
h4 = pd.read_csv(os.path.join(RES, 'h4_pettitt.csv'), dtype={'station_id': str})
ev = pd.read_csv(os.path.join(RES, 'event_hit_rates.csv'), dtype={'station_id': str})
grid = pd.read_csv(os.path.join(RES, 'spell_threshold_grid.csv'), dtype={'station_id': str})
o5ps = pd.read_csv(os.path.join(RES, 'o5_per_station.csv'), dtype={'station_id': str})
m2 = json.load(open('/tmp/my-project/verification/m2_results.json'))
npz = np.load(os.path.join(RES, 'fig_data.npz'), allow_pickle=True)

# ------------------------------------------------------------------ F1 map
import netCDF4 as nc

def draw_countries(ax, gj, names):
    for f in gj['features']:
        nm = f['properties'].get('NAME', '')
        if nm not in names:
            continue
        geom = f['geometry']
        polys = geom['coordinates'] if geom['type'] == 'MultiPolygon' else [geom['coordinates']]
        for poly in polys:
            arr = np.asarray(poly[0])
            ax.fill(arr[:, 0], arr[:, 1], color='#f2f0e9', zorder=1)
            ax.plot(arr[:, 0], arr[:, 1], color='#9a9a92', lw=0.5, zorder=2)


def fig1():
    gj = json.load(open('/tmp/my-project/verification/ne_50m_countries.geojson'))
    names = {'Morocco', 'Algeria', 'Spain', 'Mauritania', 'Portugal', 'Mali',
             'Western Sahara', 'W. Sahara'}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    draw_countries(ax, gj, names)
    ax.axhline(0, color='none')
    for _, r in st.iterrows():
        ax.scatter(r.lon, r.lat, s=26, c=C_CLS[r['class']], edgecolor='k',
                   linewidth=0.4, zorder=5)
    lab = {'601010': 'Tangier', '601270': 'Taza', '600010': 'El Hierro',
           '614150': 'Nouadhibou', '604900': 'Oran', '600330': 'Laayoune',
           '601950': 'Atlas'}
    for sid, t in lab.items():
        r = st[st.station_id == sid].iloc[0]
        off = (4, 4) if sid != '614150' else (6, -9)
        ax.annotate(t, (r.lon, r.lat), textcoords='offset points', xytext=off,
                    fontsize=6.5, color='#333')
    ax.add_patch(Rectangle((-20, 20.29), 20, 19.71, fill=False, ls='--',
                           lw=1.0, edgecolor='#555', zorder=4))
    ax.text(-12.5, 19.6, 'common domain 20.3\u201340\u00b0N, 20\u20130\u00b0E',
            fontsize=6.5, color='#555')
    ax.set_xlim(-19.5, 0.5)
    ax.set_ylim(19.5, 37.5)
    ax.set_xlabel('Longitude (\u00b0E)')
    ax.set_ylabel('Latitude (\u00b0N)')
    ax.set_title('(a) Stations and classes (N = 32 confirmatory)')
    handles = [Line2D([], [], marker='o', ls='', markerfacecolor=C_CLS[c],
                      markeredgecolor='k', markeredgewidth=0.4, markersize=5,
                      label=c.replace('_', '-'))
               for c in CLASSES]
    ax.legend(handles=handles, loc='lower right', frameon=False)
    ax.set_aspect(1.0 / np.cos(np.radians(28)))

    ax = axes[1]
    d = nc.Dataset(CERRA_FILE)
    la2 = np.ma.filled(np.ma.asarray(d.variables['latitude'][:], dtype='float64'), np.nan)
    lo2 = np.ma.filled(np.ma.asarray(d.variables['longitude'][:], dtype='float64'), np.nan)
    d.close()
    lo2n = np.where(lo2 > 180, lo2 - 360, lo2)
    box = (lo2n > -7.5) & (lo2n < -3.5) & (la2 > 33.5) & (la2 < 36.5)
    ax.scatter(lo2n[box], la2[box], s=1.2, c=C_CERRA, marker='o', zorder=4,
               label='CERRA 5.5 km')
    d = nc.Dataset(ERA5_FILE)
    la1 = np.asarray(d.variables['latitude'][:], dtype='float64')
    lo1 = np.asarray(d.variables['longitude'][:], dtype='float64')
    d.close()
    lat_m = (la1 >= 33.5) & (la1 <= 36.5)
    lon_m = (lo1 >= -7.5) & (lo1 <= -3.5)
    LA, LO = np.meshgrid(la1[lat_m], lo1[lon_m], indexing='ij')
    ax.scatter(LO, LA, s=22, facecolor='none', edgecolor=C_ERA5, linewidth=0.9,
               zorder=5, label='ERA5 0.25\u00b0')
    draw_countries(ax, gj, names)
    ax.set_xlim(-7.5, -3.5)
    ax.set_ylim(33.5, 36.5)
    ax.set_xlabel('Longitude (\u00b0E)')
    ax.set_title('(b) Grid density, Strait of Gibraltar')
    ax.legend(loc='lower left', frameon=False, markerscale=3)
    ax.set_aspect(1.0 / np.cos(np.radians(35)))
    save(fig, 'f1_map')


# ------------------------------------------------------------------ F3 H1
def fig3():
    G_rmse, G_pers = npz['G_rmse'], npz['G_pers']
    rho, ci = float(npz['rho']), npz['ci']
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    for c in CLASSES:
        m = (st['class'] == c).to_numpy()
        ax.scatter(G_rmse[m], G_pers[m], s=30, c=C_CLS[c], edgecolor='k',
                   linewidth=0.4, zorder=3, label=c.replace('_', '-'))
    ax.axhline(0, color='#888', lw=0.8)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_xlabel('G$_{\\mathrm{RMSE}}$ = RMSE$_{\\mathrm{ERA5}}$ \u2212 RMSE$_{\\mathrm{CERRA}}$ (m s$^{-1}$)')
    ax.set_ylabel('G$_{\\mathrm{pers}}$ = E$_{\\mathrm{ERA5}}$ \u2212 E$_{\\mathrm{CERRA}}$ (m s$^{-1}$)')
    ax.set_title(f'Spearman $\\rho$ = {rho:.2f}  (95% CI {ci[0]:.2f} to {ci[1]:.2f})')
    ax.legend(loc='upper left', frameon=False, fontsize=6.5)
    lim = max(np.abs(np.r_[G_rmse, G_pers])) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.12, 0.12)
    ax.annotate('right: CERRA better RMSE', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', fontsize=6.5, color='#555')
    save(fig, 'f3_h1_scatter')


# ------------------------------------------------------------------ F4 spells
def fig4():
    df = pd.read_parquet(PARQUET)
    df['date'] = pd.to_datetime(df['date'])
    CONF = st.station_id.tolist()
    res = {k: {'obs': [], 'cerra': [], 'era5': []} for k in (3, 5, 10)}
    for sid in CONF:
        g = df[(df.station_id == sid) & (df.valid == 1)].sort_values('date')
        dint = g.date.map(pd.Timestamp.toordinal).to_numpy().astype('int64')
        yr = g.date.dt.year.to_numpy()
        thr = {}
        trig = False
        for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5')):
            t = g.groupby(g.date.dt.year)[c].quantile(0.10)
            thr[p] = t
            if (t == 0).any():
                trig = True
        tmap = {p: np.array([max(thr[p].get(y, np.nan), 0.5) if trig else thr[p].get(y, np.nan)
                             for y in yr]) for p in thr}
        for k in (3, 5, 10):
            for p, c in (('obs', 'ws_obs'), ('cerra', 'ws_cerra'), ('era5', 'ws_era5')):
                low = g[c].to_numpy() <= tmap[p]
                lens = [b - a for a, b in runs(low, dint) if b - a >= k]
                if lens:
                    arr = np.array(lens)
                    yrs = len(set(yr))
                    res[k][p].append((len(lens) / yrs, arr.mean(), arr.max(),
                                      100.0 * arr.sum() / len(low)))
    ks = (3, 5, 10)
    panels = [('Events per year', 0), ('Mean duration (d)', 1), ('Longest event (d)', 2),
              ('Days in events (%)', 3)]
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.6), constrained_layout=True)
    x = np.arange(3)
    w = 0.26
    for ax, (title, idx) in zip(axes, panels):
        for i, (p, c) in enumerate((('obs', C_OBS), ('cerra', C_CERRA), ('era5', C_ERA5))):
            med = [np.median([r[idx] for r in res[k][p]]) for k in ks]
            ax.bar(x + (i - 1) * w, med, w, color=c, edgecolor='k', linewidth=0.4,
                   label=p.upper() if p != 'obs' else 'HadISD')
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in ks])
        ax.set_title(title, fontsize=8)
        if idx == 3:
            ax.set_ylabel('%')
    axes[0].legend(frameon=False, loc='upper right')
    save(fig, 'f4_spell_stats')


# ------------------------------------------------------------------ F5 T1/T2
def fig5():
    gg = grid[(grid.k == 5) & (grid.pct == 10)].copy()
    gg = gg.pivot_table(index='station_id', columns='regime',
                        values=['cerra', 'era5', 'obs'])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), constrained_layout=True,
                             sharey=True)
    for ax, prod, c in zip(axes, ('cerra', 'era5'), (C_CERRA, C_ERA5)):
        r1 = gg[(prod, 'T1_own')] / gg[('obs', 'T1_own')]
        r2 = gg[(prod, 'T2_obsthreshold')] / gg[('obs', 'T1_own')]
        bp = ax.boxplot([r1.dropna(), r2.dropna()], widths=0.5, patch_artist=True,
                        medianprops=dict(color='k', lw=1.2),
                        flierprops=dict(marker='o', markersize=3, alpha=0.6))
        for b, cc in zip(bp['boxes'], ('#b8d4ea', '#f2c8b0')):
            b.set_facecolor(cc)
            b.set_edgecolor('#555')
        ax.axhline(1.0, color='#888', lw=0.8, ls='--')
        ax.set_xticklabels(['T1 own\npercentile', 'T2 observed\nthreshold'])
        ax.set_title({'cerra': 'CERRA', 'era5': 'ERA5'}[prod], color=c)
        ax.set_ylabel('mean spell duration ratio\n(model / HadISD)' if prod == 'cerra' else '')
    save(fig, 'f5_t1t2')


# ------------------------------------------------------------------ F6 POD/FAR
def fig6():
    med = ev.groupby(['product', 'regime'])[['POD', 'FAR', 'CSI']].median()
    fig, ax = plt.subplots(figsize=(5.4, 3.0), constrained_layout=True)
    metrics = ['POD', 'FAR', 'CSI']
    x = np.arange(3)
    w = 0.2
    combos = [('cerra', 'T1_own', C_CERRA, 'CERRA (T1)'),
              ('cerra', 'T2_obsthreshold', '#7fb3d7', 'CERRA (T2)'),
              ('era5', 'T1_own', C_ERA5, 'ERA5 (T1)'),
              ('era5', 'T2_obsthreshold', '#eda67f', 'ERA5 (T2)')]
    for i, (p, rg, c, lab) in enumerate(combos):
        vals = [med.loc[(p, rg), m] for m in metrics]
        ax.bar(x + (i - 1.5) * w, vals, w, color=c, edgecolor='k', linewidth=0.4,
               label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('median across stations')
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=6.8)
    ax.set_title('Detection of observed low-wind events (spells \u2265 5 d, \u226550% overlap)')
    save(fig, 'f6_podfarcsi')


# ------------------------------------------------------------------ F7 H3
def fig7():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), constrained_layout=True,
                             sharex=True, sharey=True)
    for ax, prod, c in zip(axes, ('cerra', 'era5'), (C_CERRA, C_ERA5)):
        ins = h3[f'{prod}_in'] == 1
        ax.scatter(h3.ratio_obs[ins], h3[f'ratio_{prod}'][ins], s=26, c=c,
                   edgecolor='k', linewidth=0.4, label=f'in CI ({ins.sum()})')
        ax.scatter(h3.ratio_obs[~ins], h3[f'ratio_{prod}'][~ins], s=34, marker='x',
                   c='#b2182b', label=f'outside CI ({(~ins).sum()})')
        lims = [0.4, 1.05]
        ax.plot(lims, lims, color='#888', lw=0.8, ls='--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('P90/P50 ratio, HadISD')
        ax.set_title({'cerra': 'CERRA', 'era5': 'ERA5'}[prod], color=c)
        ax.legend(frameon=False, loc='upper left', fontsize=6.8)
    axes[0].set_ylabel('P90/P50 ratio, reanalysis')
    save(fig, 'f7_h3_ratios')


# ------------------------------------------------------------------ F8 H4
def fig8():
    ann = pd.read_csv(os.path.join(RES, 'annual_series_H4.csv'), dtype={'station_id': str})
    brk = h4[h4.pettitt_q_cerra < 0.05].sort_values('pettitt_p_cerra')
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.6), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, r) in zip(axes, brk.iterrows()):
        sid = r.station_id
        s = ann[ann.station_id == sid].sort_values('year')
        yrs = s.year.to_numpy()
        t = int(r.pettitt_t_cerra)
        ybreak = yrs[min(t, len(yrs)) - 1]
        ax.plot(s.year, s.d_cerra, '-o', ms=2.5, color=C_CERRA, lw=0.9,
                label='CERRA \u2212 obs')
        ax.plot(s.year, s.d_era5, '-o', ms=2.0, color=C_ERA5, lw=0.7, alpha=0.55,
                label='ERA5 \u2212 obs')
        ax.axvline(ybreak, color='#444', ls='--', lw=0.8)
        cls = s['class'].iloc[0].replace('_', '-')
        ax.set_title(f'{sid} ({cls})  q={r.pettitt_q_cerra:.3f}', fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xticks([1995, 2010, 2024])
    axes[0].legend(frameon=False, fontsize=6.2, loc='lower left')
    for i in range(7, len(axes)):
        axes[i].axis('off')
    fig.supxlabel('Year')
    fig.supylabel('Annual mean difference (m s$^{-1}$)')
    save(fig, 'f8_h4_breaks')


# ------------------------------------------------------------------ F9 O5
def fig9():
    sel = o5ps[(o5ps.alpha == 0.14) & (o5ps.q == 0.7)]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1.15]})
    ax = axes[0]
    data = [sel[sel['product'] == p].S_median.dropna() for p in ('obs', 'cerra', 'era5')]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                    medianprops=dict(color='k', lw=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.6))
    for b, c in zip(bp['boxes'], (C_OBS, C_CERRA, C_ERA5)):
        b.set_facecolor(c)
        b.set_alpha(0.55)
        b.set_edgecolor('#555')
    ax.set_xticklabels(['HadISD', 'CERRA', 'ERA5'])
    ax.set_ylabel('storage depth (days of mean production)')
    ax.set_title(r'(a) $\alpha$=0.14, $q$=0.7', fontsize=8)
    ax = axes[1]
    alphas = sorted(o5ps.alpha.unique())
    qs = sorted(o5ps.q.unique())
    M = np.full((len(alphas), len(qs), 2), np.nan)
    for i, a in enumerate(alphas):
        for j, q in enumerate(qs):
            s = o5ps[(o5ps.alpha == a) & (o5ps.q == q)]
            piv = s.pivot_table(index='station_id', columns='product', values='S_median')
            piv = piv.dropna()
            if len(piv):
                M[i, j, 0] = (piv.cerra / piv.obs).median()
                M[i, j, 1] = (piv.era5 / piv.obs).median()
    for k, (c, name) in enumerate(zip((C_CERRA, C_ERA5), ('CERRA', 'ERA5'))):
        MM = M[:, :, k]
        for i in range(len(alphas)):
            for j in range(len(qs)):
                if np.isfinite(MM[i, j]):
                    ax.text(j + (k - 0.5) * 0.5, i, f'{MM[i, j]:.2f}',
                            ha='center', va='center', fontsize=6.6, color=c,
                            fontweight='bold' if k == 0 else 'normal')
    ax.set_xlim(-0.5, len(qs) - 0.5)
    ax.set_ylim(len(alphas) - 0.5, -0.5)
    ax.set_xticks([j - 0.25 for j in range(len(qs))])
    ax.set_xticklabels([f'$q$={q}' for q in qs])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'$\\alpha$={a}' for a in alphas])
    ax.set_title('(b) median storage ratio vs HadISD\n(blue: CERRA, orange: ERA5)',
                 fontsize=8)
    ax.grid(False)
    handles = [Line2D([], [], marker='s', ls='', color=C_CERRA, label='CERRA'),
               Line2D([], [], marker='s', ls='', color=C_ERA5, label='ERA5')]
    ax.legend(handles=handles, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=6.8)
    save(fig, 'f9_storage')


# ------------------------------------------------------------------ F2 (data audit figure: slot masking + units)
def fig2():
    """Distribution-error vs structure-error summary + like-for-like effect."""
    senstab = pd.read_csv(os.path.join(RES, 'sensitivity_pairings.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)
    ax = axes[0]
    ax.hist(st.bias_cerra, bins=np.arange(-1.5, 1.6, 0.25), alpha=0.65, color=C_CERRA,
            edgecolor='k', lw=0.4, label='CERRA')
    ax.hist(st.bias_era5, bins=np.arange(-1.5, 1.6, 0.25), alpha=0.65, color=C_ERA5,
            edgecolor='k', lw=0.4, label='ERA5')
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('daily-mean bias (m s$^{-1}$)')
    ax.set_ylabel('stations')
    ax.legend(frameon=False)
    ax.set_title('(a) Distribution error (bias)', fontsize=8)
    ax = axes[1]
    deltas = [st.rmse_era5 - st.rmse_cerra]
    ax.hist(deltas[0], bins=np.arange(-1.0, 1.3, 0.2), color='#7fb3d7',
            edgecolor='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(deltas[0].median(), color=C_ERA5, lw=1.2, ls='--',
               label=f'median {deltas[0].median():+.2f}')
    ax.set_xlabel('RMSE$_{\\mathrm{ERA5}}$ \u2212 RMSE$_{\\mathrm{CERRA}}$ (m s$^{-1}$)')
    ax.set_ylabel('stations')
    ax.legend(frameon=False)
    ax.set_title('(b) RMSE gain of CERRA', fontsize=8)
    save(fig, 'f2_errorstructure')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    print('ALL FIGURES DONE')
