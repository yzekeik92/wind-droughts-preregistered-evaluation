#!/usr/bin/env python3
"""Generate paper/macros.tex: every recurring, error-prone number in the
manuscript is defined as a LaTeX macro derived directly from the frozen
result sheets. This removes manual transcription (the source of the
0.38-vs-0.41 and Delta-z sign errors).

Single sources of truth:
  M2_results/revision_v7.json          (ICC brackets, cluster-aware p, event
                                        anchors, POD/CSI gaps, storage slopes)
  M2_results/h1_equivalence_v6.json    (rho, TOST p, Bayes factors)
  M2_results/class_table_v3.csv        (DEM-based dz_C per class)
  M2_results/era5_dz_bilinear_v5.csv   (model-orography dz_E per class)
  M2_results/era5_orography_v4.csv     (nearest-corner dz, archipelago)
  M2_results/o5_logratio_v3.csv        (central-configuration storage ratios)

check_consistency.py re-derives every value and fails if macros.tex has
drifted from the sheets. Run order: gen_macros.py -> tectonic -> check.
"""
import json
import math
from pathlib import Path

import pandas as pd

RES = Path('/home/z/my-project/download/M2_results')
PAP = Path('/home/z/my-project/download/paper')

rv7 = json.loads((RES / 'revision_v7.json').read_text())
h1eq = json.loads((RES / 'h1_equivalence_v6.json').read_text())
cls = pd.read_csv(RES / 'class_table_v3.csv')
dzb = pd.read_csv(RES / 'era5_dz_bilinear_v5.csv')
o4 = pd.read_csv(RES / 'era5_orography_v4.csv')
o5 = pd.read_csv(RES / 'o5_logratio_v3.csv')

A = rv7['A_pooled_icc']
B = rv7['B_storage_regression']['regressions']
C = rv7['C_event_anchors']
BF = h1eq['bayes_factors']


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f0(x):
    return f"{x:.0f}"


def f1(x):
    return f"{x:.1f}"


# ---------------------------------------------------------------- macros
# name -> (value, source comment)
up = A['p_beta_binomial_upper_bracket']
lo = A['p_beta_binomial_lower_bracket']
bf_vals = [
    BF['primary_diff_spearman']['bf_neg_tr'],
    BF['logratio_spearman']['bf_neg_tr'],
    BF['primary_diff_spearman_neff23']['bf_neg_tr'],
]
bf_post = [
    BF['primary_diff_spearman']['post_p_neg'],
    BF['logratio_spearman']['post_p_neg'],
    BF['primary_diff_spearman_neff23']['post_p_neg'],
]
central = o5[(o5['alpha'] == 0.14) & (o5['q'] == 0.7)].iloc[0]
dz_e_arch = dzb[dzb['class'] == 'archipelago']['dz_era5_bilinear'].median()
dz_corner_arch = o4[o4['class'] == 'archipelago']['dz_model_near'].median()

MACROS = {
    # --- H1 equivalence (h1_equivalence_v6.json) ---
    'cBFrange': (f"{f0(min(bf_vals))}--{f0(max(bf_vals))}",
                 'one-sided BF range (diff 28.4, log 14.1, n_eff=23 12.7)'),
    'cBFfullagg': (f1(BF['block_means_diff_n8']['bf_neg_tr']),
                   'BF under full 8-block aggregation'),
    'cBFposterior': (f"{f2(min(bf_post))}--{f2(max(bf_post))}",
                     'posterior P(rho<0.5) range'),
    'cRhoDiff': (f2(h1eq['point']['rho_diff']), 'Spearman rho, difference-based'),
    'cRhoLog': (f2(h1eq['point']['rho_log']), 'Spearman rho, log-ratio'),
    'cTostPDiff': (f3(h1eq['tost_bootstrap']['diff']['p_pos_ge_margin']),
                   'one-sided TOST p, difference-based'),
    'cTostPLog': (f3(h1eq['tost_bootstrap']['log']['p_pos_ge_margin']),
                  'one-sided TOST p, log-ratio'),
    'cLogCiHi': (f3(h1eq['tost_bootstrap']['log']['ci95'][1]),
                 'log-ratio CI upper bound'),
    'cDiffCiHi': (f3(h1eq['tost_bootstrap']['diff']['ci95'][1]),
                  'difference-based CI upper bound'),
    # --- ICC brackets and cluster-aware p (revision_v7.json, section A) ---
    'cIccLower': (f3(A['pooled_icc_all5_lower_bracket']), 'pooled ICC, all 5 tests'),
    'cIccUpper': (f3(A['pooled_icc_identified3_upper_bracket']),
                  'pooled ICC, 3 identified tests (conservative)'),
    'cGridMaxP': (f2(A['max_p_H2H3_any_rho_0_to_02']),
                  'max p over ICC grid 0-0.20, H2+H3'),
    'cpHtwoC': (f3(up['H2_cerra']), 'beta-binomial H2 CERRA @ upper bracket'),
    'cpHtwoE': (f3(up['H2_era5']), 'beta-binomial H2 ERA5 @ upper bracket'),
    'cpHthreeC': (f3(up['H3_cerra']), 'beta-binomial H3 CERRA @ upper bracket'),
    'cpHthreeE': (f3(up['H3_era5']), 'beta-binomial H3 ERA5 @ upper bracket'),
    'cpHfour': (f3(up['H4_cerra']), 'beta-binomial H4 CERRA @ upper bracket'),
    'cpHfourPooled': (f3(lo['H4_cerra']), 'beta-binomial H4 CERRA @ lower bracket'),
    # --- event anchors and detection gaps (revision_v7.json, section C) ---
    'cEvObs': (f2(C['medians_T1_floored_k5']['obs']['events_per_year']),
               'observed events per station-year (T1 floored, k>=5)'),
    'cEvC': (f2(C['medians_T1_floored_k5']['cerra']['events_per_year']),
             'CERRA events per station-year'),
    'cEvE': (f2(C['medians_T1_floored_k5']['era5']['events_per_year']),
             'ERA5 events per station-year'),
    'cDurObs': (f2(C['medians_T1_floored_k5']['obs']['mean_dur_days']),
                'observed median spell duration (d)'),
    'cDurC': (f2(C['medians_T1_floored_k5']['cerra']['mean_dur_days']),
              'CERRA median spell duration (d)'),
    'cDurE': (f2(C['medians_T1_floored_k5']['era5']['mean_dur_days']),
              'ERA5 median spell duration (d)'),
    'cPodGap': (f"{f2(C['pod_gap_range'][0])}--{f2(C['pod_gap_range'][1])}",
                'POD gap range across pipeline variants'),
    'cCsiGap': (f"{f2(C['csi_gap_range'][0])}--{f2(C['csi_gap_range'][1])}",
                'CSI gap range across pipeline variants'),
    # --- storage ratios (o5_logratio_v3.csv, central alpha=0.14 q=0.7) ---
    'cStoreC': (f2(central['cerra_pers_ratio_median']),
                'CERRA median per-station storage ratio, central config'),
    'cStoreE': (f2(central['era5_pers_ratio_median']),
                'ERA5 median per-station storage ratio, central config'),
    # --- storage regression slopes (revision_v7.json, section B) ---
    'cSlopeMean': (f2(B['era5_on_mean_wind_blockboot']['slope_per_ms']),
                   'ERA5 ln-ratio on own mean wind (original spec)'),
    'cSlopeCross': (f2(B['era5_on_cerra_wind']['slope_per_ms']),
                    'ERA5 ln-ratio on CERRA wind (cross regressor)'),
    'cSlopeFE': (f2(B['era5_on_cerra_wind_classFE']['slope_per_ms']),
                 'cross regressor + class fixed effects'),
    'cSlopeNoArch': (f2(B['era5_on_cerra_wind_excl_archipelago']['slope_per_ms']),
                     'cross regressor, archipelago excluded'),
    # --- height references (era5_dz_bilinear_v5.csv / era5_orography_v4.csv) ---
    'cDzBilArchMed': (f0(abs(dz_e_arch)),
                      '|median dz_era5_bilinear| archipelago; dz median is '
                      f"{dz_e_arch:+.1f} m (station minus cell), i.e. the model "
                      'surface sits ABOVE the station'),
    'cDzCornerArchMed': (f0(abs(dz_corner_arch)),
                         '|median dz_model_near| archipelago; dz median is '
                         f"{dz_corner_arch:+.1f} m, i.e. the nearest-corner model "
                         'elevation exceeds the station'),
}

# per-class height references consumed by Appendix Table A3 (checked, not
# macro-substituted, in check_consistency.py); keys = table display names
DZ_TABLE = {
    'Archipelago': (round(cls.set_index('class').loc['archipelago', 'dz_C']),
                    round(dz_e_arch)),
    'Atlantic coast': (round(cls.set_index('class').loc['atlantic', 'dz_C']),
                       round(dzb[dzb['class'] == 'atlantic']['dz_era5_bilinear'].median())),
    'Strait/Mediterr.': (round(cls.set_index('class').loc['strait_med', 'dz_C']),
                         round(dzb[dzb['class'] == 'strait_med']['dz_era5_bilinear'].median())),
    'Interior': (round(cls.set_index('class').loc['interior', 'dz_C']),
                 round(dzb[dzb['class'] == 'interior']['dz_era5_bilinear'].median())),
}


def render(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def main():
    lines = [
        '% AUTO-GENERATED by scripts/gen_macros.py from the frozen result',
        '% sheets in M2_results/ -- DO NOT EDIT BY HAND.',
        '% Re-run scripts/gen_macros.py; scripts/check_consistency.py fails',
        '% if this file drifts from the sheets.',
        '',
    ]
    for name, (value, comment) in MACROS.items():
        lines.append(f"% {comment}")
        lines.append(render(name, value))
        lines.append('')
    (PAP / 'macros.tex').write_text('\n'.join(lines))
    manifest = {
        'macros': {k: v[0] for k, v in MACROS.items()},
        'sources': {k: v[1] for k, v in MACROS.items()},
        'dz_table_by_class': DZ_TABLE,
    }
    (PAP / 'macros_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"wrote {PAP/'macros.tex'} ({len(MACROS)} macros)")
    for name, (value, _) in MACROS.items():
        print(f"  \\{name} = {value}")


if __name__ == '__main__':
    main()
