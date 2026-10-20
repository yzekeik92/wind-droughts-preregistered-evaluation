# Pre-specified protocol and results package: "Do reanalyses get the droughts right?"

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
