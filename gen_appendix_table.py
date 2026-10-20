#!/usr/bin/env python3
"""Generate sec_appendix.tex (Appendix A: HadISD QC suite) from official sources.

Rows come verbatim from the Met Office tests_codes.txt (v3.4.3 2025f online
material, downloaded 2026-10-17). Variable mapping and column accounting come
from the released QC code (github.com/rjhd2/HadISD_v3, FLAG_COL_DICT +
internal_checks.py) cross-checked empirically in released station files.
"""
import re

SRC = '/tmp/my-project/verification/hadisd_tests_codes.txt'
OUT = '/home/z/my-project/download/paper/sec_appendix.tex'

WIND = {'WRC', 'DWS', 'HWS', 'WSS', 'OCW', 'WVR', 'WSL', 'WRS', 'WSP',
        'DRS', 'HRS', 'RSS', 'WDL', 'WMB', 'BBB'}

# group -> (order, applies-to string)
VAR = {}
for c in ['DUP']:      VAR[c] = 'all variables'
for c in ['DNL']:      VAR[c] = 'T, Td, SLP, Ws, Wd'
for c in ['TFV', 'TGP', 'TRC', 'DTS', 'HTS', 'TSS', 'TCM', 'TSP', 'OCT', 'TVR']:
    VAR[c] = 'temperature'
for c in ['DFV', 'DGP', 'DRC', 'DDS', 'HDS', 'DSS', 'DCM', 'DSP', 'OCD', 'DVR',
          'DPD', 'SSS', 'DCF']:
    VAR[c] = 'dewpoint'
for c in ['SFV', 'SGP', 'PRC', 'DPS', 'HPS', 'PSS', 'PCM', 'PSP', 'OCS', 'SVR']:
    VAR[c] = 'sea-level pressure'
for c in ['WRC', 'DWS', 'HWS', 'WSS', 'OCW', 'WVR', 'WSL', 'WRS', 'WSP']:
    VAR[c] = 'wind speed'
for c in ['DRS', 'HRS', 'RSS', 'WDL']:
    VAR[c] = 'wind direction'
for c in ['NGC', 'CUOT', 'CUOL', 'CUOM', 'CUOH', 'CST', 'FLW', 'FMC']:
    VAR[c] = 'cloud'
VAR['STNLP'] = 'station-level pressure'
VAR['PPTN'] = 'precipitation'
for c in ['TMB']:      VAR[c] = 'temperature'
for c in ['DMB']:      VAR[c] = 'dewpoint'
for c in ['SMB']:      VAR[c] = 'sea-level pressure'
for c in ['WMB']:      VAR[c] = 'wind speed'
for c in ['BBB']:      VAR[c] = 'wind direction'
for c in ['CMB', 'HMB', 'LMB', 'MMB', 'BMB']: VAR[c] = 'cloud'
for c in ['ALL_T']:    VAR[c] = 'temperature'
for c in ['ALL_Td']:   VAR[c] = 'dewpoint'
for c in ['ALL_SLP']:  VAR[c] = 'sea-level pressure'
for c in ['ALL_W']:    VAR[c] = 'wind speed'
for c in ['ALL_WD']:   VAR[c] = 'wind direction'
VAR['ACL'] = 'cloud'
for c, v in [('STR_T', 'temperature'), ('STR_D', 'dewpoint'),
             ('STR_S', 'sea-level pressure'), ('STR_W', 'wind speed'),
             ('STR_WD', 'wind direction')]:
    VAR[c] = v
# v2-era climatological-outlier codes
VAR['TOT'] = 'temperature'
VAR['DOT'] = 'dewpoint'
VAR['SOT'] = 'sea-level pressure'

LEGACY = {'TOT', 'DOT', 'SOT'}

codes = []
with open(SRC) as f:
    for line in f:
        line = line.strip()
        if not line or '=' not in line:
            continue
        k, v = line.split('=', 1)
        codes.append((k.strip(), re.sub(r'\s+', ' ', v.strip())))

missing = [c for c, _ in codes if c not in VAR]
assert not missing, f'unmapped codes: {missing}'
assert len(codes) == 82, f'expected 82 codes, got {len(codes)}'

codes.sort(key=lambda x: x[0])


def esc(s):
    return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_')


def row(code, desc):
    bold = (r'\textbf{%s}' if code in WIND else '%s') % esc(code)
    var = VAR[code]
    note = ''
    if code in LEGACY:
        note = ' (v2-era)'
    return '%s & %s & %s%s' % (bold, var, esc(desc), note)


h1, h2 = codes[:41], codes[41:]
lines = []
for i in range(41):
    c1, d1 = h1[i]
    c2, d2 = h2[i]
    lines.append('%s & %s \\\\' % (row(c1, d1), row(c2, d2)))

TABLE_ROWS = '\n'.join(lines)

TEX = r'''\appendix
\setcounter{table}{0}
\renewcommand{\thetable}{A\arabic{table}}

\section{HadISD quality-control suite}
\label{app:qc}

\subsection{Structure of the flag matrix}
Each HadISD report carries a 71-column quality-control status vector (the
\texttt{quality\_control\_flags} variable of the released station files), so
the ``71 tests'' cited in Section~\ref{sec:hadisd} are read directly from the
data. Each column corresponds to one named test or clearing status; the codes
are documented by the Met Office in the official \texttt{tests\_codes.txt} of
version 3.4.3 (2025f), and the code-to-column mapping of the production chain
was cross-checked against the released QC source code
(\url{https://github.com/rjhd2/HadISD_v3}) and verified
empirically in released station files. Flag values follow the convention of
the QC code: 0 = pass; 1 = fail, in which case the value is removed by the
HadISD masking step and replaced by the reserved fill in the released files;
2 = tentative (suspect), in which case the value is retained but flagged;
$-1$ = informational. The 71 columns decompose into 58 test columns (one per
named test, plus the two shared checks applied to all variables), 3
neighbour-outlier status columns (temperature, dewpoint and sea-level
pressure; second processing stage), 5 month-clearing statuses (temperature,
dewpoint, sea-level pressure, wind speed, wind direction) and 5 cloud
month-clearing status columns, which the version-3 production chain does not
populate and which are empty in the released files inspected for this study.
Table~\ref{tab:qccodes} lists all 82 documented codes, including the six
per-variable aggregates and the five string aggregates that exist only in the
summary outputs, not as matrix columns.

\begin{table}[tbp]
\centering
\footnotesize
\caption{HadISD quality-control codes (version 3.4.3, 2025f), transcribed
verbatim from the official \texttt{tests\_codes.txt} distributed with the
dataset. Variable mapping follows the released QC production chain.
Bold codes are the wind-speed or wind-direction checks. T, temperature;
Td, dewpoint; SLP, sea-level pressure; Ws, wind speed; Wd, wind direction.
The suffix (v2-era) marks codes documented for the version-2 suite that the
version-3 chain subsumes into the climatological-outlier family.}
\label{tab:qccodes}
\begin{tabular}{@{}l l l @{\hspace{1.2em}} l l l@{}}
\toprule
Code & Applies to & Check & Code & Applies to & Check \\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\end{table}

\subsection{Use in the confirmatory pipeline}
The default validity criterion of Section~\ref{sec:hadisd} (finite,
non-negative wind speed) automatically excludes the reports that HadISD
itself removed, because removed values are replaced by the reserved fill
(flag~1). Tentative flags (value~2) are set by the QC code only in the
distributional-gap and climatological-outlier tests, and in the released
files inspected they appear in exactly those families. Across the 32
confirmatory stations in the 1995--2024 window they affect 6{,}065 of
1{,}324{,}811 valid wind reports (0.46\%). The pre-registered strict variant
is not a subset of tests: a report is discarded if \emph{any} of the 71
columns carries a tentative flag, a deliberately conservative filter applied
uniformly to all stations and both models. Its effect is confined to
$\le 0.01$\ms{} in paired median RMSE (Table~\ref{tab:sens}), so no headline
result depends on the handling of tentative flags. The wind family of the
suite (bold in Table~\ref{tab:qccodes}) comprises 15 wind-specific
checks---records (\textbf{WRC}); day, hour and straight-string streaks
(\textbf{DWS}, \textbf{HWS}, \textbf{WSS} for speed and \textbf{DRS},
\textbf{HRS}, \textbf{RSS} for direction); odd clusters (\textbf{OCW});
variance (\textbf{WVR}); the logic checks (\textbf{WSL} for speed,
\textbf{WDL} for direction); the wind-rose cross-check (\textbf{WRS}); the
spike test (\textbf{WSP})---together with the wind month-clearing statuses
(\textbf{WMB}, \textbf{BBB}) and the two shared checks (duplicates, diurnal)
applied to every variable.
'''

TEX = TEX.replace('__ROWS__', TABLE_ROWS)
with open(OUT, 'w') as f:
    f.write(TEX)
print('wrote', OUT, '| codes:', len(codes), '| wind codes:', len(WIND))
