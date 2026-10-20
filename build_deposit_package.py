#!/usr/bin/env python3
"""Build the submission-ready pre-registration deposit package (ZIP).

Contents: frozen prereg protocol versions, deviation/ampliation log,
reviewer-anticipation protocol, hash manifests, evidence tables (M1a/M1b/M2),
pipeline code, HadISD provenance (official tests_codes.txt + station list
extract), and a README with the post-hoc archival note and verification steps.
"""
import hashlib
import os
import shutil
import subprocess
import datetime

BASE = '/home/z/my-project/download'
PKG = os.path.join(BASE, 'prereg_deposit')
STAGE = '/tmp/my-project/deposit_stage'
ZIP = os.path.join(BASE, 'prereg_deposit_v1.2.zip')

if os.path.exists(STAGE):
    shutil.rmtree(STAGE)
if os.path.exists(PKG):
    shutil.rmtree(PKG)
os.makedirs(STAGE, exist_ok=True)
os.makedirs(PKG, exist_ok=True)

# ---- content ----
files = [
    (f'{BASE}/prereg_draft_v1.md',            'prereg/prereg_v1.0_draft.md'),
    (f'{BASE}/prereg_v1.1.md',                'prereg/prereg_v1.1.md'),
    (f'{BASE}/prereg_v1.2.md',                'prereg/prereg_v1.2_FROZEN.md'),
    (f'{BASE}/Ampliacion_prereg_M1b_M2.md',   'prereg/ampliation_confirmatory_M1b_M2.md'),
    (f'{BASE}/REVIEWER_PROTOCOL.md',          'prereg/reviewer_anticipation_protocol.md'),
    (f'{BASE}/SHA256SUMS.txt',                'hashes/SHA256SUMS_preregistration.txt'),
    (f'{BASE}/SHA256SUMS_confirmatory.txt',   'hashes/SHA256SUMS_confirmatory.txt'),
    (f'{BASE}/Deviation_Log.md',              'Deviation_Log.md'),
    (f'{BASE}/Deviation_Log.pdf',             'Deviation_Log.pdf'),
]
for src, dst in files:
    assert os.path.exists(src), src
    os.makedirs(os.path.join(STAGE, os.path.dirname(dst)), exist_ok=True)
    shutil.copy2(src, os.path.join(STAGE, dst))

trees = [
    (f'{BASE}/M1a_evidence',   'evidence_M1a'),
    (f'{BASE}/M1b_pilot',      'evidence_M1b_pilot'),
    (f'{BASE}/M2_results',     'results_M2'),
]
for src, dst in trees:
    shutil.copytree(src, os.path.join(STAGE, dst))

# pipeline code
code_dst = os.path.join(STAGE, 'code')
os.makedirs(code_dst, exist_ok=True)
for s in os.listdir('/home/z/my-project/scripts'):
    if s.endswith('.py'):
        shutil.copy2(f'/home/z/my-project/scripts/{s}', code_dst)

# paper source with generated number macros (auditable sheets -> macros -> text chain)
psrc = os.path.join(STAGE, 'paper_source')
os.makedirs(psrc, exist_ok=True)
PAP = os.path.join(BASE, 'paper')
for f in os.listdir(PAP):
    if f.endswith(('.tex', '.bib')) or f in ('macros_manifest.json',):
        shutil.copy2(os.path.join(PAP, f), psrc)
shutil.copytree(os.path.join(PAP, 'figures'), os.path.join(psrc, 'figures'))
shutil.copy2(os.path.join(PAP, 'Manuscript_DoReanalysesGetTheDroughtsRight_v12.pdf'), psrc)

# HadISD provenance (appendix A sources)
prov = os.path.join(STAGE, 'provenance_hadisd')
os.makedirs(prov, exist_ok=True)
shutil.copy2('/tmp/my-project/verification/hadisd_tests_codes.txt', prov)
shutil.copy2('/tmp/my-project/verification/hadisd_station_fullinfo.txt', prov)
with open(os.path.join(prov, 'QC_CODE_SOURCE.txt'), 'w') as f:
    f.write(
        'HadISD QC code cross-check source\n'
        'Repository: https://github.com/rjhd2/HadISD_v3\n'
        'Commit inspected: 420ebb3c965b61ad394f15b0f71a37b73985a97d (2021-06-17)\n'
        'Key files: internal_checks.py (test->column invocations), '
        'neighbour_checks.py (FLAG_COL_DICT, month-clearing columns 44-48),\n'
        'qc_tests/climatological.py and qc_tests/distributional_gap.py '
        '(tentative flag value 2).\n'
        'tests_codes.txt downloaded from the Met Office HadISD v343_2025f '
        'online material on 2026-10-17.\n')

README = """# Pre-specified protocol and results package: "Do reanalyses get the droughts right?"

**Archival status: the protocol is pre-specified (fixed before the
confirmatory run) and deposited after the analysis.** The frozen protocol
(prereg/prereg_v1.2_FROZEN.md) and its
companion artefacts were completed and hash-stamped before the confirmatory
reanalysis-versus-station statistics were computed; the public repository
deposit could only be made at manuscript submission. We therefore call the
protocol "pre-specified", not "pre-registered": filesystem timestamps and
SHA-256 hashes document the internal timeline, but they are not cryptographic
proof of precedence, and readers should weigh the pre-specification claims
accordingly. This decision is deviation #13; the protocol freeze also required
it to be declared before the confirmatory run (prereg v1.2, item 7 of the
changelog).

## Deviation log

**Deviation_Log.pdf / Deviation_Log.md** (package root) is the readable
deviation register: deviations #11 to #19 with what changed, why, and the
manuscript section that references each entry. Deviations #1 to #10 are in the
changelog history of the preregistration. The post-review additions are:
deviation #14 (reporting re-expressions, outputs with the `_v3` suffix in
`results_M2/`), #15 (cluster-aware checks, first-difference diagnostics,
independent storage partition, real model orography for ERA5, spell-convention
audit; `_v4` suffix or `revision_v4.json`), #16 (beta-binomial calibration,
per-variant event metrics of the sensitivity table, wind-regression storage
diagnostic, bilinear-orography dz, extended bibliographic check; `_v5` suffix
or `revision_v5.json`, `systematic_search_v5.json`), #17 (post-hoc H1
equivalence analysis: TOST and one-sided Bayes factor; `h1_equivalence_v6.json`),
#18 (common ICC with brackets and p(ICC) grid, storage-wind regression
re-examined with cross regressors, class fixed effects and archipelago
exclusion, event-metric unification, range-first quoting; `revision_v7.json`,
`o5_wind_regression_v7.csv`) and #19 (context bibliographic expansion from 23
to 56 DOI-verified references).

## Verification

```
sha256sum -c hashes/SHA256SUMS_preregistration.txt    # frozen protocol set
sha256sum -c hashes/SHA256SUMS_confirmatory.txt       # confirmatory artefacts
md5sum    -c hashes/MD5SUMS_package.txt               # every file in this package
```

(The two SHA-256 manifests reference the original relative paths; map them to
the directories of this package when re-checking. The MD5 manifest uses
package-relative paths, so `md5sum -c` works directly from the package root.)

## Contents

- `Deviation_Log.md` / `Deviation_Log.pdf`: readable deviation register
  (#11 to #19) with manuscript cross-references.
- `prereg/`: protocol v1.0 (draft), v1.1, **v1.2 (frozen)**, the ampliation
  record of the confirmatory phase (conventions D7-D10, deviations #13 to #19)
  and the reviewer-anticipation protocol that governed every design decision.
- `hashes/`: SHA-256 manifests (pre-registration set, confirmatory set) and
  the MD5 manifest of the complete package.
- `evidence_M1a/`: observations-only eligibility and audit tables (station
  selection never touched reanalysis values).
- `evidence_M1b_pilot/`: pilot-pairing records (5 seeded stations).
- `results_M2/`: confirmatory results sheets (H1-H4, event scores, storage
  depths, sensitivity variants), the revision recomputation sheets
  (`revision_v3.json` to `revision_v7.json`, `*_v3`/`_v4`/`_v5` suffixed
  sheets and `*_v7` outputs) and figure code outputs.
- `code/`: complete pipeline (audit, eligibility, pairing, confirmatory
  statistics, revision recomputations, invariant joins, DEM direct-download
  attempts, CERRA-orography CDS join script, bibliographic check, figures,
  appendix generation). Seed: 20261017.
- `paper_source/`: manuscript source with generated number macros, the frozen
  result sheets they derive from (macros_manifest.json) and the compiled
  manuscript PDF (v12).
- `provenance_hadisd/`: official HadISD v3.4.3 (2025f) `tests_codes.txt`,
  the station list with names, and the QC-code cross-check record
  (github.com/rjhd2/HadISD_v3, commit 420ebb3).

## Contact
Corresponding author: withheld for double-blind review; contact details with
the journal submission.
"""
with open(os.path.join(STAGE, 'README.md'), 'w') as f:
    f.write(README)

# move staged tree into download/prereg_deposit for user visibility
if os.path.exists(PKG):
    shutil.rmtree(PKG)
shutil.copytree(STAGE, PKG)

# manifest of the package itself
manifest = []
for root, _, fns in os.walk(STAGE):
    for fn in sorted(fns):
        p = os.path.join(root, fn)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        rel = os.path.relpath(p, STAGE)
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%dT%H:%M:%S')
        manifest.append(f'{h}  {mt}  {rel}')
with open(os.path.join(PKG, 'MANIFEST.sha256'), 'w') as f:
    f.write('# sha256 | mtime | path\n' + '\n'.join(manifest) + '\n')

# MD5 manifest of the complete package (package-relative paths; md5sum -c works
# from the package root). Covers every packaged file except this manifest.
md5_lines = []
for root, _, fns in os.walk(PKG):
    for fn in sorted(fns):
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, PKG)
        if rel == os.path.join('hashes', 'MD5SUMS_package.txt'):
            continue
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        md5_lines.append(f'{h}  {rel}')
md5_lines.sort(key=lambda s: s.split('  ', 1)[1])
with open(os.path.join(PKG, 'hashes', 'MD5SUMS_package.txt'), 'w') as f:
    f.write('# MD5 of every file in this package (paths relative to the package root).\n'
            '# Verify from the package root with: md5sum -c hashes/MD5SUMS_package.txt\n'
            + '\n'.join(md5_lines) + '\n')
# rebuild ZIP from PKG (so both manifests included)
if os.path.exists(ZIP):
    os.remove(ZIP)
subprocess.run(['zip', '-qr', ZIP, '.'], cwd=PKG, check=True)
zh = hashlib.sha256(open(ZIP, 'rb').read()).hexdigest()
print('package files:', len(manifest))
print('zip:', ZIP, f'{os.path.getsize(ZIP)/1e6:.1f} MB', 'sha256', zh[:16] + '...')
