#!/usr/bin/env python3
"""Pre-submission consistency sweep for the manuscript (v8).

Checks:
 1. Macro drift  -- every value in paper/macros.tex is re-derived from the
    frozen result sheets (gen_macros.MACROS); fails on any mismatch.
 2. Managed literals -- numbers that must appear ONLY via \\c... macros are
    scanned as raw literals in the tex sources; residual hits must be on an
    explicit whitelist (different quantities that coincide numerically).
    The historically wrong value 0.066 (v7 transcription of the H4
    beta-binomial p; the sheet says 0.0651) is absolutely forbidden.
 3. Sign guard -- dz = z_station - z_cell (positive: station above cell);
    the archipelago ERA5-bilinear median is negative (-77 m), so the model
    surface sits ABOVE the station. "below the station" must not appear;
    "above the station" must appear in sec_data (S2.3) and sec_results (S4.1).
 4. Forbidden vocabulary -- "overestimat*" presupposes the station is the
    truth (H3 wording); "this environment" is lab-notebook language.
 5. Appendix Table A3 (tab:dz) -- values re-derived from the CSV sheets.
 6. Abstract length -- <= 250 words, counted after rendering macros.

Exit code 0 = all clear; 1 = failures.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, '/home/z/my-project/scripts')
import gen_macros  # noqa: E402  (re-derives every value from the sheets)

PAP = Path('/home/z/my-project/download/paper')
TEX_FILES = ['main.tex', 'sec_intro.tex', 'sec_data.tex', 'sec_methods.tex',
             'sec_results.tex', 'sec_discussion.tex', 'sec_backmatter.tex',
             'sec_appendix.tex']

failures = []
notes = []


def check(cond, msg_ok, msg_fail):
    if cond:
        print(f"  PASS  {msg_ok}")
    else:
        print(f"  FAIL  {msg_fail}")
        failures.append(msg_fail)


# ------------------------------------------------------------- 1. drift
print("[1] Macro drift against frozen sheets")
tex_macros = {}
for line in (PAP / 'macros.tex').read_text().splitlines():
    m = re.match(r'\\newcommand\{\\(\w+)\}\{(.+)\}\s*$', line)
    if m:
        tex_macros[m.group(1)] = m.group(2)
check(tex_macros == {k: v[0] for k, v in gen_macros.MACROS.items()},
      f"all {len(tex_macros)} macros match the sheets",
      "macros.tex has drifted from the sheets: " +
      "; ".join(f"{k}: tex={tex_macros.get(k)} sheet={gen_macros.MACROS[k][0]}"
                for k in set(tex_macros) | set(gen_macros.MACROS)
                if tex_macros.get(k) != gen_macros.MACROS[k][0]))

# ------------------------------------------------- 2. managed literals
print("[2] Managed literals (values that must travel via macros)")
WHITELIST = {
    'sec_appendix.tex': [r'^\d{6} &'],                     # per-station table rows
    'sec_data.tex': [r'^Class &|^\w[\w/.\s]*&\s*\d',       # tab:classes rows
                    r'^Archipelago|Atlantic coast|Strait|Interior'],
    'sec_results.tex': [
        r'E_\{\\mathrm\{CERRA\}\}\) = 0\.025',            # ln E ratio, not TOST p
        r'ratios 0\.55--0\.60',                            # H3 SD ratios
        r'0\.49 at 601950 and 0\.55 at 602100',            # per-station CVs
        r'\[-0\.33,\\,-0\.13\]',                           # slope CI bound, not slope
        r'\[-0\.13,\\,0\.55',                              # ERA5-partition CI bound
        r'0\.17--0\.19',                                   # H1 sensitivity rho range
        r'all-8-slot means &.*0\.38',                      # CSI of the slot variant
        r'changes the ratios by at most \$\+0\.004\$',     # bias-adjustment delta
        r'1/256\\approx0\.004',                            # sign-flip floor
        r'1\.13 for ERA5 at the central configuration',    # POOLED ERA5 ratio
        r'they are 1\.13 \(CERRA',                         # >=4 m/s partition
        r'medians to 1\.13',                               # no-flag subset medians
        r'storage median ratio 1\.13 \(C\)',               # drop-both subset
    ],
    'sec_discussion.tex': [
        r'interior 0\.59 vs 0\.55',                        # POD medians
        r'RMSE 1\.06 vs 1\.09',                            # Strait RMSE
        r'\(1\.10 vs 1\.13\)',                             # pooled reversal
        r'unchanged to within 0\.004',                     # bias-adjustment delta
    ],
    'sec_methods.tex': [
        r'range from \$-0\.19\$ to',                       # raw ANOVA ICC range
        r'1/256\\approx0\.004',                            # sign-flip floor
    ],
}
WHITELIST['sec_results.tex'].append(r'range from \$-0\.19\$ to')  # L213 ICC range
ZERO_TOLERANCE = [r'0\.066']  # the v7 transcription error


def scan_literals():
    problems = []
    for fname in TEX_FILES:
        text = (PAP / fname).read_text()
        wl = WHITELIST.get(fname, [])
        for lineno, line in enumerate(text.splitlines(), 1):
            for pat in ZERO_TOLERANCE:
                if re.search(r'(?<![\d.])' + pat + r'(?!\d)', line):
                    problems.append(f"{fname}:{lineno} forbidden value {pat}: {line.strip()}")
            for pat, macro in SCAN_PATTERNS:
                if re.search(pat, line):
                    if any(re.search(w, line) for w in wl):
                        continue
                    problems.append(f"{fname}:{lineno} literal for \\{macro}: {line.strip()}")
    return problems


SCAN_PATTERNS = [
    (r'(?<![\d.])0\.42(?!\d)', 'cEvC'),
    (r'(?<![\d.])0\.38(?!\d)', 'cEvE'),
    (r'(?<![\d.])0\.55(?!\d)', 'cEvObs'),
    (r'5\.93(?!\d)', 'cDurObs'),
    (r'5\.86(?!\d)', 'cDurC'),
    (r'5\.73(?!\d)', 'cDurE'),
    (r'0\.12--0\.18', 'cPodGap'),
    (r'0\.05--0\.15', 'cCsiGap'),
    (r'13--28', 'cBFrange'),
    (r'0\.97--0\.99', 'cBFposterior'),
    (r'(?<![\d.])0\.050(?!\d)', 'cTostPDiff'),
    (r'(?<![\d.])0\.025(?!\d)', 'cTostPLog'),
    (r'0\.497(?!\d)', 'cLogCiHi'),
    (r'0\.622(?!\d)', 'cDiffCiHi'),
    (r'(?<![\d.])0\.126(?!\d)', 'cIccUpper'),
    (r'(?<![\d.])0\.065(?!\d)', 'cpHfour'),
    (r'(?<![\d.])0\.014(?!\d)', 'cpHtwoC'),
    (r'(?<![\d.])0\.007(?!\d)', 'cpHtwoE'),
    (r'(?<![\d.])0\.010(?!\d)', 'cpHthreeC'),
    (r'(?<![\d.])0\.004(?!\d)', 'cpHthreeE'),
    (r'(?<![\d.])-0\.27(?!\d)', 'cSlopeMean'),
    (r'(?<![\d.])-0\.19(?!\d)', 'cSlopeCross'),
    (r'(?<![\d.])-0\.02(?!\d)', 'cSlopeFE'),
    (r'(?<![\d.])-0\.13(?!\d)', 'cSlopeNoArch'),
    (r'(?<![\d.])1\.13(?!\d)', 'cStoreC'),
    (r'(?<![\d.])1\.09(?!\d)', 'cStoreE'),
]
lits = scan_literals()
check(not lits, "no un-macro-ized managed literals",
      "residual literals found:\n    " + "\n    ".join(lits))

# --------------------------------------------------------- 3. sign guard
print("[3] Delta-z sign guard (dz = station - cell; arch median is negative)")
alltex = {f: (PAP / f).read_text() for f in TEX_FILES}
below = [(f, i, l.strip()) for f, t in alltex.items()
         for i, l in enumerate(t.splitlines(), 1)
         if re.search(r'below the station(?![s])', l)]  # 'the stations' = historical quote
check(not below, "no 'below the station' anywhere",
      "'below the station' still present:\n    " +
      "\n    ".join(f"{f}:{i}" for f, i, _ in below))
for f in ('sec_data.tex', 'sec_results.tex'):
    check('above the station' in alltex[f],
          f"{f} states the model surface sits above the station",
          f"{f} lost the corrected 'above the station' wording")

# ------------------------------------------------------ 4. vocabulary
print("[4] Forbidden vocabulary")
for pat, why in [(r'\boverestimat', "'overestimate*' presupposes station truth"),
                 (r'this environment', "lab-notebook language 'this environment'"),
                 (r'-0\.102|0\.103', "defensive -0.102/-0.103 parenthetical")]:
    hits = [(f, i) for f, t in alltex.items()
            for i, l in enumerate(t.splitlines(), 1) if re.search(pat, l)]
    check(not hits, f"no {why}", f"{why} -- hits: {hits}")

# ------------------------------------------------- 5. appendix dz table
print("[5] Appendix Table A3 (tab:dz) against CSV medians")
app = alltex['sec_appendix.tex']
block = app.split('\\label{tab:dz}')[1].split('\\bottomrule')[0]
rows = {}
for m in re.finditer(
        r'^(Archipelago|Atlantic coast|Strait/Mediterr\.|Interior)\s*&\s*'
        r'(\$-\$)?(\d+)\s*&\s*(\$-\$)?(\d+)', block, re.M):
    rows[m.group(1)] = (int(m.group(3)) * (-1 if m.group(2) else 1),
                        int(m.group(5)) * (-1 if m.group(4) else 1))
check(rows == gen_macros.DZ_TABLE,
      f"tab:dz values match the sheets: {rows}",
      f"tab:dz mismatch: table={rows} sheets={gen_macros.DZ_TABLE}")

# ------------------------------------------------- 6. abstract length
print("[6] Abstract length (<= 250 words)")
main = alltex['main.tex']
abs_txt = main.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
manifest = {k: v[0] for k, v in gen_macros.MACROS.items()}
for name, val in manifest.items():
    abs_txt = abs_txt.replace('\\' + name + '{}', val).replace('\\' + name, val)
abs_txt = abs_txt.replace('\\ms', 'm/s').replace('\\,', ' ').replace('~', ' ')
abs_txt = abs_txt.replace('\\%', '%').replace('\\#', '#').replace('\\,', ' ')
abs_txt = re.sub(r'\\[a-zA-Z]+\*?', ' ', abs_txt)
abs_txt = re.sub(r'[${}]', ' ', abs_txt)
words = [w for w in abs_txt.split() if re.search(r'[A-Za-z0-9]', w)]
check(len(words) <= 250, f"abstract has {len(words)} words",
      f"abstract has {len(words)} words (> 250)")

# ------------------------------------------------- wiring
print("[7] macro wiring")
check('\\input{macros}' in main, "main.tex inputs macros.tex",
      "main.tex does not \\input{macros}")

# --------------------------------------------- 8. citation numbers
print("[8] Citation-number audit (hardcoded [n] vs unsrtnat order)")
sys.path.insert(0, '/home/z/my-project/scripts')
import check_citations  # noqa: E402
cit_problems = check_citations.run(quiet=True)
check(not cit_problems, "no hardcoded citation numbers; all keys wired",
      "citation audit problems:\n    " + "\n    ".join(cit_problems))

print()
if failures:
    print(f"CHECK FAILED: {len(failures)} problem(s)")
    sys.exit(1)
print("ALL CHECKS PASSED")
