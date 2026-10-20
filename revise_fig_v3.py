#!/usr/bin/env python3
"""REVISION v3 FIGURES — fixes the Q1 audit figure issues.

f2_errorstructure: panel (b) bins extended so all 32 stations are visible
   (two values were clipped beyond the axis).
f3_h1_scatter: re-expressed in relative (log-ratio) gains with per-station
   year-block bootstrap CIs on the persistence gain; preregistered
   difference-based rho reported in the annotation.
f7_h3_ratios: the seven H4 rupture stations are highlighted so readers can
   see which stations drive the observed low-side outliers.
f9_storage: (b) now shows paired per-station log-ratios with block-bootstrap
   CIs of the median (the statistic defined in the text); grid ranges in the
   annotation.
f4 spell-stat medians dumped to f4_stats_v3.json for exact text quoting.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = '/home/z/my-project/download/M2_results'
FIG = os.path.join(RES, 'figures')
PARQUET = '/tmp/my-project/M1b/confirmatory_daily_paired.parquet'
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

st = pd.read_csv(os.path.join(RES, 'station_metrics_confirmatory.csv'),
                 dtype={'station_id': str})
h3 = pd.read_csv(os.path.join(RES, 'h3_annual_ratios.csv'), dtype={'station_id': str})
h4 = pd.read_csv(os.path.join(RES, 'h4_pettitt.csv'), dtype={'station_id': str})
o5ps = pd.read_csv(os.path.join(RES, 'o5_per_station.csv'), dtype={'station_id': str})
o5v3 = pd.read_csv(os.path.join(RES, 'o5_logratio_v3.csv'))
h1ci = pd.read_csv(os.path.join(RES, 'h1_pers_logratio_ci_v3.csv'), dtype={'station_id': str})
rupt7 = h4[h4.pettitt_q_cerra < 0.05].station_id.tolist()
json.dump(rupt7, open(os.path.join(RES, 'rupture_stations_v3.json'), 'w'))


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
        stt = np.flatnonzero(d == 1)
        en = np.flatnonzero(d == -1)
        out.extend([(int(a), int(b)) for a, b in zip(stt + s, en + s)])
    return out


# ------------------------------------------------------------------ F2
def fig2():
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
    ax.set_title('(a) Distribution error (bias; n = 32)', fontsize=8)
    ax = axes[1]
    d = (st.rmse_era5 - st.rmse_cerra).to_numpy()
    lo = np.floor(d.min() * 5) / 5 - 0.05
    hi = np.ceil(d.max() * 5) / 5 + 0.05
    ax.hist(d, bins=np.arange(lo, hi + 0.001, 0.2), color='#7fb3d7',
            edgecolor='k', lw=0.4)
    ax.axvline(0, color='k', lw=0.8)
    ax.axvline(np.median(d), color=C_ERA5, lw=1.2, ls='--',
               label=f'median {np.median(d):+.2f} (n = {len(d)})')
    ax.set_xlabel('RMSE$_{\\mathrm{ERA5}}$ \u2212 RMSE$_{\\mathrm{CERRA}}$ (m s$^{-1}$)')
    ax.set_ylabel('stations')
    ax.legend(frameon=False)
    ax.set_title('(b) RMSE gain of CERRA (all 32 shown)', fontsize=8)
    save(fig, 'f2_errorstructure')


# ------------------------------------------------------------------ F3 H1 (log-ratio)
def fig3():
    m = st.set_index('station_id')
    ci = h1ci.set_index('station_id')
    sids = [s for s in m.index if s in ci.index]
    x = np.log((m.loc[sids, 'rmse_era5'] / m.loc[sids, 'rmse_cerra']).to_numpy())
    y = np.log((m.loc[sids, 'E_pers_era5'] / m.loc[sids, 'E_pers_cerra']).to_numpy())
    ylo = y - ci.loc[sids, 'ci_low'].to_numpy()
    yhi = ci.loc[sids, 'ci_high'].to_numpy() - y
    cls = [m.loc[s, 'class'] for s in sids]
    rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    fig, ax = plt.subplots(figsize=(4.3, 3.4), constrained_layout=True)
    for c in CLASSES:
        mm = np.array(cls) == c
        ax.errorbar(x[mm], y[mm], yerr=[ylo[mm], yhi[mm]], fmt='o', ms=4.5,
                    c=C_CLS[c], ecolor='#999', elinewidth=0.7, capsize=1.5,
                    markeredgecolor='k', markeredgewidth=0.4, zorder=3,
                    label=c.replace('_', '-'))
    ax.axhline(0, color='#888', lw=0.8)
    ax.axvline(0, color='#888', lw=0.8)
    ax.set_xlabel('relative RMSE gain\nln(RMSE$_{\\mathrm{ERA5}}$/RMSE$_{\\mathrm{CERRA}}$)')
    ax.set_ylabel('relative persistence gain\nln(E$_{\\mathrm{ERA5}}$/E$_{\\mathrm{CERRA}}$)')
    ax.set_title(f'Spearman $\\rho$ = {rho:.2f}, 95% CI [$-$0.10, 0.49]')
    ax.legend(loc='upper left', frameon=False, fontsize=6.5)
    ax.annotate('bars: 95% CI of the persistence gain\n(year-block bootstrap, 1000 draws)',
                xy=(0.98, 0.02), xycoords='axes fraction', ha='right',
                fontsize=6.2, color='#555')
    ax.annotate('preregistered difference-based $\\rho$ = 0.16, CI [$-$0.10, 0.62]',
                xy=(0.98, 0.95), xycoords='axes fraction', ha='right',
                fontsize=6.2, color='#555')
    save(fig, 'f3_h1_scatter')


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
        rup = h3.station_id.isin(rupt7)
        ax.scatter(h3.ratio_obs[rup], h3[f'ratio_{prod}'][rup], s=60, facecolor='none',
                   edgecolor='k', linewidth=1.0, label='H4 break station (7)')
        lims = [0.4, 1.05]
        ax.plot(lims, lims, color='#888', lw=0.8, ls='--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('P90/P50 ratio, HadISD')
        ax.set_title({'cerra': 'CERRA', 'era5': 'ERA5'}[prod], color=c)
        ax.legend(frameon=False, loc='lower right', fontsize=6.4)
    axes[0].set_ylabel('P90/P50 ratio, reanalysis')
    save(fig, 'f7_h3_ratios')


# ------------------------------------------------------------------ F9 O5
def fig9():
    sel = o5ps[(o5ps.alpha == 0.14) & (o5ps.q == 0.7)]
    piv = sel.pivot_table(index='station_id', columns='product', values='S_median').dropna()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1, 1.05]})
    ax = axes[0]
    data = [piv[p].to_numpy() for p in ('obs', 'cerra', 'era5')]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True,
                    medianprops=dict(color='k', lw=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.0))
    rng = np.random.default_rng(7)
    for i, (vals, c) in enumerate(zip(data, (C_OBS, C_CERRA, C_ERA5))):
        ax.scatter(rng.normal(i + 1, 0.055, len(vals)), vals, s=9, c=c,
                   edgecolor='k', linewidth=0.25, alpha=0.85, zorder=4)
    for b, c in zip(bp['boxes'], (C_OBS, C_CERRA, C_ERA5)):
        b.set_facecolor(c)
        b.set_alpha(0.35)
        b.set_edgecolor('#555')
    ax.axhline(piv.obs.median(), color='#444', lw=0.8, ls=':')
    ax.annotate(f'HadISD median {piv.obs.median():.1f} d', xy=(0.06, piv.obs.median() + 2),
                xycoords=('axes fraction', 'data'), fontsize=6.4, color='#444',
                va='bottom', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.85))
    ax.set_xticklabels(['HadISD', 'CERRA', 'ERA5'])
    ax.set_ylabel('storage depth (days of mean production)')
    ax.set_title(r'(a) per-station median annual depth, $\alpha$=0.14, $q$=0.7', fontsize=7.6)
    ax = axes[1]
    lnc = np.log(piv.cerra / piv.obs).to_numpy()
    lne = np.log(piv.era5 / piv.obs).to_numpy()
    bp = ax.boxplot([lnc, lne], widths=0.45, patch_artist=True,
                    medianprops=dict(color='k', lw=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.0))
    for i, (vals, c) in enumerate(zip((lnc, lne), (C_CERRA, C_ERA5))):
        ax.scatter(rng.normal(i + 1, 0.055, len(vals)), vals, s=9, c=c,
                   edgecolor='k', linewidth=0.25, alpha=0.85, zorder=4)
    for b, c in zip(bp['boxes'], (C_CERRA, C_ERA5)):
        b.set_facecolor(c)
        b.set_alpha(0.35)
        b.set_edgecolor('#555')
    cen = o5v3[(o5v3.alpha == 0.14) & (o5v3.q == 0.7)].iloc[0]
    for i, (m, ci) in enumerate(((cen.cerra_lnratio_median, json.loads(cen.cerra_lnratio_ci)),
                                 (cen.era5_lnratio_median, json.loads(cen.era5_lnratio_ci)))):
        ax.errorbar(i + 1.33, m,
                    yerr=[[m - ci[0]], [ci[1] - m]], fmt='_', ms=14,
                    ecolor='k', elinewidth=1.4, capsize=4, capthick=1.4, zorder=6)
    ax.axhline(0, color='#888', lw=0.8, ls='--')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['CERRA', 'ERA5'])
    ax.set_ylabel(r'ln($S_{\mathrm{prod}}$/$S_{\mathrm{HadISD}}$)')
    ax.set_title('(b) paired per-station log-ratio (bars: 95% CI of median)', fontsize=7.6)
    ax.annotate('grid medians of per-station ratios:\n'
                f'CERRA {o5v3.cerra_pers_ratio_median.min():.2f}\u2013{o5v3.cerra_pers_ratio_median.max():.2f}, '
                f'ERA5 {o5v3.era5_pers_ratio_median.min():.2f}\u2013{o5v3.era5_pers_ratio_median.max():.2f}',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=6.2, color='#555')
    save(fig, 'f9_storage')


# ------------------------------------------------------------------ F4 stats dump
def fig4_stats():
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
    out = {}
    for k in (3, 5, 10):
        out[f'k{k}'] = {p: {'events_per_year': float(np.median([r[0] for r in res[k][p]])),
                            'mean_dur': float(np.median([r[1] for r in res[k][p]])),
                            'max_dur': float(np.median([r[2] for r in res[k][p]])),
                            'days_pct': float(np.median([r[3] for r in res[k][p]]))}
                        for p in ('obs', 'cerra', 'era5')}
    json.dump(out, open(os.path.join(RES, 'f4_stats_v3.json'), 'w'), indent=1)
    print('f4 stats:', json.dumps({k: out[k] for k in ('k5', 'k10')}, indent=1))


if __name__ == '__main__':
    fig2()
    fig3()
    fig7()
    fig9()
    fig4_stats()
    print('V3 FIGURES DONE')
