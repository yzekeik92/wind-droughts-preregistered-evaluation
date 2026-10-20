#!/usr/bin/env python3
"""Reviewer-revision figures: F1 (all stations named) and F3 (regression + 95% CI ellipse).

F1 uses an iterative bbox-based label de-overlap solver (renderer-exact, like
adjustText): radial seeds, pairwise repulsion, dot avoidance and axes-bound push-in.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D

RES = '/home/z/my-project/download/M2_results'
FIG = os.path.join(RES, 'figures')
os.makedirs(FIG, exist_ok=True)

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
_dem = pd.read_csv('/tmp/my-project/verification/dem_covariates.csv',
                   dtype={'station_id': str})[['station_id', 'lat', 'lon']]
st = st.merge(_dem, on='station_id', how='left')

# Short display names (official HadISD primary names; airport qualifiers omitted)
SHORT = {
    '600010': 'Hierro', '600050': 'La Palma', '600150': 'Tenerife Norte',
    '600200': 'Sta Cruz Tenerife', '600250': 'Tenerife Sur',
    '600300': 'Gran Canaria', '600330': 'Hassan I', '600350': 'Fuerteventura',
    '600400': 'Lanzarote', '600600': 'Sidi Ifni', '601010': 'Tanger',
    '601050': 'Larache', '601150': 'Angads', '601270': 'Taza',
    '601350': 'Rabat', '601410': 'Saiss', '601500': 'Bassatine',
    '601560': 'Mohammed V', '601950': 'Midelt', '602100': 'Moulay Ali Cherif',
    '602300': 'Menara', '602520': 'Al Massira', '602650': 'Ouarzazate',
    '603180': 'Saniat Rmel', '604900': 'Es Senia', '605200': 'Sidi Bel Abbes',
    '605310': 'Zenata', '605710': 'Bechar', '606020': 'Beni Abbes',
    '606200': 'Touat', '606560': 'Tindouf', '614150': 'Nouadhibou',
}
assert set(SHORT) == set(st.station_id), 'short-name table mismatch'

npz = np.load(os.path.join(RES, 'fig_data.npz'), allow_pickle=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'))
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------------- F1 map
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
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.9), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.5, 1]})
    ax = axes[0]
    draw_countries(ax, gj, names)
    for _, r in st.iterrows():
        ax.scatter(r.lon, r.lat, s=24, c=C_CLS[r['class']], edgecolor='k',
                   linewidth=0.4, zorder=5)

    sids = st.station_id.tolist()
    lons = st.lon.to_numpy()
    lats = st.lat.to_numpy()
    labels = [SHORT[s] for s in sids]

    # ---- radial seeds: push away from nearby stations, then solver refines ----
    offs = []
    for i in range(len(sids)):
        push = np.zeros(2)
        for j in range(len(sids)):
            if i == j:
                continue
            dvec = np.array([(lons[i] - lons[j]) * 11.7,
                             (lats[i] - lats[j]) * 10.3])  # approx pt/deg
            d = np.linalg.norm(dvec)
            if d < 1e-6:
                dvec = np.array([1.0, 0.0])
                d = 1.0
            if d < 60:
                push += dvec / d * (60.0 - d)
        n = np.linalg.norm(push)
        offs.append(push / n * 14.0 if n > 1e-6 else np.array([12.0, 12.0]))
    # manual seeds for the Canary cluster (empty Atlantic margins)
    manual_idx = {sids.index(s): o for s, o in {
        '600010': (-4, -40), '600050': (0, -26), '600250': (-30, -12),
        '600150': (-30, 34), '600200': (-12, 20), '600300': (24, -22),
        '600350': (28, 16), '600400': (30, 28),
        '601500': (-44, -10), '602520': (16, -14),
    }.items()}
    for i, o in manual_idx.items():
        offs[i] = np.array(o, dtype=float)

    annots = []
    for i in range(len(sids)):
        an = ax.annotate(labels[i], (lons[i], lats[i]), textcoords='offset points',
                         xytext=tuple(offs[i]), fontsize=6.2, color='#333',
                         ha='center', va='center', zorder=6)
        annots.append(an)

    # dot positions in display coords at draw time
    fig.canvas.draw()

    def get_bboxes():
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        return [an.get_window_extent(renderer) for an in annots], renderer

    pt_per_px = 72.0 / fig.dpi
    for it in range(250):
        bboxes, renderer = get_bboxes()
        disp = ax.transData.transform(np.column_stack([lons, lats]))
        moved = 0.0
        # label-label repulsion
        for i in range(len(annots)):
            for j in range(i + 1, len(annots)):
                bi, bj = bboxes[i], bboxes[j]
                if bi.overlaps(bj):
                    ci = np.array([bi.x0 + bi.width / 2, bi.y0 + bi.height / 2])
                    cj = np.array([bj.x0 + bj.width / 2, bj.y0 + bj.height / 2])
                    dvec = ci - cj
                    n = np.linalg.norm(dvec)
                    dvec = dvec / n if n > 1e-6 else np.array([1.0, 0.0])
                    push = dvec * 3.5
                    offs[i] += push
                    offs[j] -= push
                    moved += 3.5
        # label vs foreign dot repulsion
        for i in range(len(annots)):
            bi = bboxes[i]
            ci = np.array([bi.x0 + bi.width / 2, bi.y0 + bi.height / 2])
            for k in range(len(sids)):
                if k == i:
                    continue
                dvec = ci - disp[k]
                n = np.linalg.norm(dvec)
                if n < (bi.width / 2 + 4):
                    offs[i] += (dvec / (n if n > 1e-6 else 1.0)) * 3.0
                    moved += 3.0
        # keep inside axes bounds
        axbb = ax.get_window_extent(renderer)
        for i in range(len(annots)):
            bi = bboxes[i]
            if bi.x0 < axbb.x0 + 1:
                offs[i][0] += 5.0
                moved += 5.0
            if bi.x1 > axbb.x1 - 1:
                offs[i][0] -= 5.0
                moved += 5.0
            if bi.y0 < axbb.y0 + 1:
                offs[i][1] += 5.0
                moved += 5.0
            if bi.y1 > axbb.y1 - 1:
                offs[i][1] -= 5.0
                moved += 5.0
        for i in range(len(annots)):
            annots[i].set_position(tuple(offs[i]))
        if moved < 0.5:
            print(f'label solver converged at iter {it}')
            break

    # recreate all labels once: leader arrow when displaced far, plain otherwise
    for i, an in enumerate(annots):
        an.remove()
        if np.hypot(*offs[i]) > 14:
            annots[i] = ax.annotate(labels[i], (lons[i], lats[i]),
                                    textcoords='offset points',
                                    xytext=tuple(offs[i]), fontsize=6.2,
                                    color='#333', ha='center', va='center',
                                    zorder=6,
                                    arrowprops=dict(arrowstyle='-', color='#999',
                                                    lw=0.45, shrinkA=1.0,
                                                    shrinkB=2.5))
        else:
            annots[i] = ax.annotate(labels[i], (lons[i], lats[i]),
                                    textcoords='offset points',
                                    xytext=tuple(offs[i]), fontsize=6.2,
                                    color='#333', ha='center', va='center',
                                    zorder=6)
    # measurement-only check with plain text (leaders excluded from bboxes)
    plain = [ax.annotate(labels[i], (lons[i], lats[i]),
                         textcoords='offset points', xytext=tuple(offs[i]),
                         fontsize=6.2, color='#333', ha='center', va='center',
                         zorder=6) for i in range(len(annots))]
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bbs = [a.get_window_extent(ren) for a in plain]
    ncol = sum(1 for i in range(len(plain)) for j in range(i + 1, len(plain))
               if bbs[i].overlaps(bbs[j]))
    print('remaining label overlaps:', ncol)
    for a in plain:
        a.remove()

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

    import netCDF4 as nc
    ax = axes[1]
    CERRA_FILE = '/tmp/my-project/CERRA_data/cerra_1995_morocco.nc'
    ERA5_FILE = '/tmp/my-project/ERA5_data/1995.nc'
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
    leg = ax.legend(loc='lower left', frameon=True, markerscale=3,
                    facecolor='white', framealpha=0.85, edgecolor='#bbb')
    leg.set_zorder(7)
    ax.set_aspect(1.0 / np.cos(np.radians(35)))
    save(fig, 'f1_map')


# ---------------------------------------------------------------------- F3 H1
def fig3():
    G_rmse, G_pers = npz['G_rmse'], npz['G_pers']
    rho, ci = float(npz['rho']), npz['ci']
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)

    # least-squares fit (faint) and bivariate 95% confidence ellipse (faint)
    b, a = np.polyfit(G_rmse, G_pers, 1)
    xr = np.linspace(G_rmse.min() * 1.08, G_rmse.max() * 1.08, 50)
    ax.plot(xr, a + b * xr, color='#777', lw=0.8, ls='--', zorder=2)
    mean = np.array([G_rmse.mean(), G_pers.mean()])
    cov = np.cov(np.vstack([G_rmse, G_pers]))
    evals, evecs = np.linalg.eigh(cov)
    chi2_95 = 5.991  # 2 dof, 95%
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    ang = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
    ell = Ellipse(xy=mean,
                  width=2 * np.sqrt(chi2_95 * evals[0]),
                  height=2 * np.sqrt(chi2_95 * evals[1]),
                  angle=ang, fill=False, color='#999', lw=0.8, ls=':', zorder=2)
    ax.add_patch(ell)

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
    ax.annotate('dashed: least-squares fit; dotted: 95% confidence ellipse',
                xy=(0.98, 0.055), xycoords='axes fraction',
                ha='right', fontsize=6.0, color='#555')
    ax.annotate('right: CERRA better RMSE', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', fontsize=6.5, color='#555')
    save(fig, 'f3_h1_scatter')


if __name__ == '__main__':
    fig1()
    fig3()
    print('REVISION FIGURES DONE')
