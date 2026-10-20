#!/usr/bin/env python3
"""Deviation Log body PDF (Report route, ReportLab).

Builds the body (no cover; cover is rendered separately via html2poster.js
and merged with pypdf). Output: /home/z/my-project/scripts/devlog_body.pdf
"""
import os
import sys

PDF_SKILL_DIR = '/home/z/my-project/skills/pdf'
sys.path.insert(0, os.path.join(PDF_SKILL_DIR, 'scripts'))

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, CondPageBreak, KeepTogether,
                                HRFlowable)

# ---------- fonts ----------
FONT_DIR = '/usr/share/fonts'
pdfmetrics.registerFont(TTFont('NotoSerifSC', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf'))
pdfmetrics.registerFont(TTFont('NotoSerifSC-Bold', f'{FONT_DIR}/truetype/noto-serif-sc/NotoSerifSC-Bold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif', f'{FONT_DIR}/truetype/freefont/FreeSerif.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Bold', f'{FONT_DIR}/truetype/freefont/FreeSerifBold.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-Italic', f'{FONT_DIR}/truetype/freefont/FreeSerifItalic.ttf'))
pdfmetrics.registerFont(TTFont('FreeSerif-BoldItalic', f'{FONT_DIR}/truetype/freefont/FreeSerifBoldItalic.ttf'))
registerFontFamily('NotoSerifSC', normal='NotoSerifSC', bold='NotoSerifSC-Bold')
registerFontFamily('FreeSerif', normal='FreeSerif', bold='FreeSerif-Bold',
                   italic='FreeSerif-Italic', boldItalic='FreeSerif-BoldItalic')

from pdf import install_font_fallback  # noqa: E402
install_font_fallback()

# ---------- palette (palette.cascade, minimal mode) ----------
PAGE_BG       = colors.HexColor('#f5f4f4')
SECTION_BG    = colors.HexColor('#e9e9e7')
CARD_BG       = colors.HexColor('#efeeeb')
TABLE_STRIPE  = colors.HexColor('#efeeec')
HEADER_FILL   = colors.HexColor('#685f44')
COVER_BLOCK   = colors.HexColor('#706853')
BORDER        = colors.HexColor('#c3bdab')
ICON          = colors.HexColor('#a38a3e')
ACCENT        = colors.HexColor('#8a7127')
ACCENT_2      = colors.HexColor('#44a9ca')
TEXT_PRIMARY  = colors.HexColor('#201f1d')
TEXT_MUTED    = colors.HexColor('#848179')

# ---------- page geometry ----------
MARGIN = 2.2 * cm
PAGE_W, PAGE_H = A4
AVAIL_W = PAGE_W - 2 * MARGIN
AVAIL_H = PAGE_H - 2 * MARGIN
OUT = '/home/z/my-project/scripts/devlog_body.pdf'

# ---------- styles ----------
body = ParagraphStyle('Body', fontName='FreeSerif', fontSize=10.5, leading=16,
                      alignment=TA_JUSTIFY, textColor=TEXT_PRIMARY,
                      spaceBefore=0, spaceAfter=8)
h1 = ParagraphStyle('H1', fontName='FreeSerif', fontSize=16, leading=20,
                    alignment=TA_LEFT, textColor=HEADER_FILL,
                    spaceBefore=18, spaceAfter=4)
dev_h = ParagraphStyle('DevH', fontName='FreeSerif', fontSize=12.5, leading=16,
                       alignment=TA_LEFT, textColor=HEADER_FILL,
                       spaceBefore=14, spaceAfter=2)
dev_meta = ParagraphStyle('DevMeta', fontName='FreeSerif-Italic', fontSize=9,
                          leading=12.5, alignment=TA_LEFT, textColor=TEXT_MUTED,
                          spaceBefore=0, spaceAfter=6)
kicker = ParagraphStyle('Kicker', fontName='FreeSerif', fontSize=10, leading=14,
                        alignment=TA_LEFT, textColor=TEXT_MUTED, spaceAfter=2)
title_style = ParagraphStyle('TitleStyle', fontName='FreeSerif', fontSize=20,
                             leading=25, alignment=TA_LEFT,
                             textColor=TEXT_PRIMARY, spaceAfter=2)
tbl_head = ParagraphStyle('TblHead', fontName='FreeSerif', fontSize=9,
                          leading=11.5, alignment=TA_CENTER, textColor=colors.white)
tbl_cell = ParagraphStyle('TblCell', fontName='FreeSerif', fontSize=8.5,
                          leading=11, alignment=TA_LEFT, textColor=TEXT_PRIMARY)
note = ParagraphStyle('Note', fontName='FreeSerif', fontSize=9, leading=13,
                      alignment=TA_LEFT, textColor=TEXT_MUTED, spaceAfter=6)


def heading_block(text, style, rule=True):
    parts = [Paragraph(f'<b>{text}</b>', style)]
    if rule:
        parts.append(HRFlowable(width='100%', thickness=1.1, color=BORDER,
                                spaceBefore=0, spaceAfter=10))
    return parts


def deviation_block(d):
    head = Paragraph(f'<b>Deviation #{d["num"]} ({d["date"]}): {d["title"]}</b>', dev_h)
    meta = Paragraph(f'Trigger: {d["trigger"]}  |  Nature: {d["nature"]}', dev_meta)
    what = Paragraph(f'<b>What changed.</b> {d["what"]}', body)
    why = Paragraph(f'<b>Why.</b> {d["why"]}', body)
    where = Paragraph(f'<b>Where in the manuscript.</b> {d["where"]}', body)
    return [CondPageBreak(AVAIL_H * 0.18),
            KeepTogether([head, meta, what]),
            why, where]


INTRO_1 = ('This document is the readable companion to the machine-readable '
           'deviation record (<font name="FreeSerif-Italic">prereg/'
           'ampliation_confirmatory_M1b_M2.md</font> and the changelogs of the '
           'frozen protocol). Deviations #1 to #10 are recorded in the '
           'changelog history of the preregistration itself (draft v1.0, v1.1, '
           'v1.2) and cover design decisions taken during development, before '
           'the confirmatory run. This log details #11 to #19: one '
           'pre-confirmatory code defect, one pre-declared mechanical rule, '
           'one timeline deviation, and six post-review additions of analysis '
           'or reporting.')
INTRO_2 = ('Three facts frame every entry below. First, the frozen protocol '
           'text (<font name="FreeSerif-Italic">prereg/prereg_v1.2_FROZEN.md'
           '</font>) was never edited after the freeze; every post-freeze '
           'change is a numbered deviation, not a silent edit. Second, all '
           'post-review additions (#14 to #19) are reporting or analysis '
           'additions applied identically to both products; no confirmatory '
           'statistic, threshold, station set or decision rule was re-tuned, '
           'and the pre-specified tests are always quoted first, with the new '
           'quantities accompanying them as checks or diagnostics. Third, the '
           'paper says "pre-specified" rather than "pre-registered" because of '
           'deviation #13: the public deposit postdates the analysis, while '
           'every internal artifact carries a filesystem timestamp and a '
           'SHA-256 hash that predate the first confirmatory statistic. The '
           'resampling seed for all confirmatory and revision computations is '
           '20261017.')

SUMMARY_ROWS = [
    ('#11', '2026-10-17', 'M1b pilot diagnostics', 'Code defect fixed before the confirmatory run', 'Sec. 3.1'),
    ('#12', '2026-10-17', 'Calm-degeneracy risk seen in the pilot', 'Pre-declared mechanical rule (D6)', 'Sec. 3.3, 4.8'),
    ('#13', '2026-10-17', 'External accounts unavailable at analysis time', 'Timeline deviation: deposit deferred', 'Sec. 3.1, 5.4'),
    ('#14', '2026-10-18', 'Internal pre-submission audit', 'Reporting re-expressions only', 'Sec. 3.1, 3.6, 4.2, 4.8'),
    ('#15', '2026-10-18', 'Internal review, round 2', 'Analysis and reporting additions', 'Sec. 4.4 to 4.7, App. A, App. B'),
    ('#16', '2026-10-18', 'Internal review, round 3', 'Calibration, diagnostics, bibliographic check', 'Sec. 3.6, 4.4, 4.7, sensitivity table, App. B'),
    ('#17', '2026-10-18', 'Internal review, round 4', 'Post-hoc equivalence analysis (quantifies, does not alter)', 'Sec. 3.4, 4.3, verdicts table'),
    ('#18', '2026-10-19', 'Internal review, round 5', 'Common ICC, robustness, range-first quoting', 'Sec. 3.6, 4.7, 4.8'),
    ('#19', '2026-10-20', 'Manuscript review round (v9)', 'Context bibliographic expansion (23 to 56 references)', 'Sec. 3.1, App. B'),
]

DEVIATIONS = [
    dict(num=11, date='2026-10-17', title='Daily-value indexing bug, fixed before the confirmatory run',
         trigger='M1b pilot diagnostics', nature='Code defect fixed pre-confirmatory',
         what=('The per-slot closest-observation selection indexed the global wind array with subset '
               'indices, so the daily observed values in the pairing table were drawn from the wrong '
               'reports. The bug was fixed, the M1a observation pipeline was re-run end to end, and the '
               'eligibility tables were verified byte-identical after the fix (eligibility depends only '
               'on day counts, not values), so the eligibility result (37 eligible stations) and the '
               'pilot draw stand; only the daily wind values changed. Pre-fix artifacts are retained '
               'under verification/backup_pre_bugfix_20261017/.'),
         why=('The M1b pilot layer did its job: its diagnostics exposed a physics contradiction, '
              'near-zero observation-model correlations while CERRA and ERA5 correlated mutually at '
              '0.74 to 0.94, confirmed by a raw per-day spot check. The fix predates every '
              'confirmatory statistic.'),
         where=('Section 3.1 (pipeline validation, "archived as deviation #11 with pre-fix backups"); '
                'frozen protocol prereg v1.2, item 6; reviewer-anticipation register entry R15.')),
    dict(num=12, date='2026-10-17', title='Calm-degeneracy guard for the spell threshold (D6)',
         trigger='Calm-degeneracy risk seen in the pilot', nature='Pre-declared mechanical rule',
         what=('A mechanical guard, declared before any confirmatory statistic: if any station-year P10 '
               'of observed daily means equals exactly 0.0 (pilot station 605180 shows 1,071 zero days '
               'out of 7,810), the floored threshold max(P10, 0.5 m/s), the anemometer start speed, is '
               'applied uniformly to all stations and both products, and the raw (unfloored) variant is '
               'reported alongside.'),
         why=('At calm-heavy stations the annual P10 can be exactly zero, which degenerates the '
              'definition of observed low-wind spells (spells would become runs of exact-zero reports). '
              'Declaring the rule a priori removes any freedom in choosing the threshold after seeing '
              'results, and the raw variant shows that no headline classification depends on the floor.'),
         where=('Section 3.3 (spell thresholds and the calm guard), Section 4.8 (raw-threshold variants '
                'leave all verdicts unchanged), the confirmatory verdicts table notes; frozen protocol '
                'prereg v1.2, item 7.')),
    dict(num=13, date='2026-10-17', title='External deposit deferred, confirmatory run authorized',
         trigger='External accounts unavailable at analysis time', nature='Timeline deviation',
         what=('The v1.2 freeze required a public OSF/Zenodo deposit with a DOI before the first '
               'confirmatory statistic. The confirmatory run proceeded on 2026-10-17 over the 32 '
               'confirmatory stations without the external deposit; the public deposit is made at '
               'manuscript submission.'),
         why=('No external repository account was available to the team at analysis time. The '
              'mitigation is full temporal traceability: every artifact keeps a filesystem timestamp '
              'and a SHA-256 content hash predating the confirmatory run, the hash manifest is frozen '
              'inside the package, and the protocol files and changelogs are immutable. The residual '
              'weakness is stated rather than hidden.'),
         where=('Section 3.1 (the "pre-specified" versus "pre-registered" paragraph), Section 5.4 '
                '(limitations), cover letter. This deviation is the reason the paper never claims '
                'pre-registration.')),
    dict(num=14, date='2026-10-18', title='Internal pre-submission audit, reporting re-expressions',
         trigger='Internal pre-submission audit', nature='Reporting re-expressions only',
         what=('Seven reporting re-expressions, applied identically to both products. (1) The H1 '
               'persistence gain is re-expressed as per-station log-ratios ln(E_ERA5/E_CERRA); E is '
               'dimensionless and the v2 text mislabeled it in m/s. The pre-specified difference-based '
               'statistic is retained as confirmatory (rho = 0.156, interval [-0.100, 0.611]; log-ratio '
               'rho = 0.209). (2) Bootstrap-based excess p-values are replaced by exact one-sided '
               'binomial tests with Holm correction, since the spatial bootstrap cannot resolve '
               'p-values below 1e-4 with eight blocks. (3) Row counts corrected (338,591 paired days, '
               '324,562 valid for the 32 stations; the earlier figures included the three sensitivity '
               'stations) and the selection funnel documented. (4) The storage statistic is redefined '
               'as the paired per-station log-ratio ln(S_prod/S_obs) with spatial-block intervals; the '
               'pooled median ratio is reported as secondary. (5) Exclusion sensitivities added for the '
               'seven H4-flagged and six calm-dominated stations. (6) The per-class table is '
               're-aggregated per product (the earlier pooled medians mixed the two products). '
               '(7) Terminology aligned ("pre-specified (deposited after the analysis)"; timestamps '
               'not presented as cryptographic proof).'),
         why=('Internal audit of the v2 manuscript before submission; re-interpretations and reporting '
              'changes only, with the frozen protocol text unchanged.'),
         where=('Section 3.1 (deviation #14 sentence), Section 3.6 (exact binomial tests), Section 4.2 '
                '(log-ratio H1), Section 4.8 (exclusion sensitivities), the per-class and skill tables. '
                'Outputs carry the _v3 suffix in results_M2/.')),
    dict(num=15, date='2026-10-18', title='Review round 2: cluster checks, first differences, independent storage partition, real model orography',
         trigger='Internal review, round 2', nature='Analysis and reporting additions',
         what=('Eight additions. (1) Sign structure of H1 verified: 14 of 32 per-station intervals '
               'exclude zero, 12 favouring CERRA (sign test p = 0.013); summary language changed from '
               '"scatter around zero" to a small consistently signed advantage. (2) Cluster-aware '
               'checks for H2, H3 and H4: ICC(1) over the eight spatial blocks, design effects, and '
               'sign-flip permutation of block contributions. (3) H3 first-difference diagnostics (SD '
               'ratio 0.58 CERRA / 0.54 ERA5; correlation 0.63 / 0.66; ERA5-CERRA mutual 0.87 on '
               'levels, 0.90 on differences; observed CV 0.086 versus 0.035 to 0.038 modelled). '
               '(4) The 3-hourly smoothing hypothesis for the POD gap tested and rejected (CERRA '
               'lag-1 autocorrelation is lower than ERA5 at 25 of 32 stations). (5) Event medians '
               'recomputed without calm sites and without the floor. (6) Storage: the observed >= 4 m/s '
               'partition gains intervals, and an independent partition by each station mean reanalysis '
               'wind is added. (7) ERA5 model orography and land-sea mask joined from the WeatherBench2 '
               'redistribution; the per-class table ERA5 height reference switches to model orography, '
               'and the earlier "cells hundreds of metres below the stations" narrative is identified '
               'as sign-inverted. (8) Appendix B gains the systematic search record; Appendix A gains '
               'the per-station detection table; the wind QC code count is corrected (13 codes plus '
               '2 month-clearing statuses).'),
         why=('Round 2 of the internal review; analysis and reporting additions that do not change any '
              'frozen rule.'),
         where=('Section 3.1 (deviation #15 sentence), Sections 4.4 to 4.7, the per-class table, '
                'Appendices A and B. Outputs carry the _v4 suffix or appear in '
                'results_M2/revision_v4.json.')),
    dict(num=16, date='2026-10-18', title='Review round 3: calibration, per-variant event metrics, direct storage diagnostic, extended bibliographic check',
         trigger='Internal review, round 3', nature='Calibration and diagnostics',
         what=('Eight additions. (1) The share tests gain a calibrated cluster-aware check: an exact '
               'beta-binomial with the intra-class correlation estimated from the same eight blocks; '
               'the sign-flip permutation is downgraded to a coarse resolution floor (1/256). H2 CERRA '
               'becomes "marginal but significant" (p = 0.015); H4 CERRA reads 0.066 in the calibrated '
               'check. (2) The sensitivity summary table is extended with event-level metrics (E, POD, '
               'CSI, event frequency, mean duration) for every pipeline variant. (3) The storage-wind '
               'regression is added (ERA5 slope -0.269 per m/s, interval [-0.390, -0.162]; CERRA flat), '
               'and the calm-half interval of round 2 is declared degenerate (three effective blocks), '
               'with "not significant" kept distinct from "sign change". (4) The '
               'identical-across-series claim is corrected to "same procedure", with variance ratios '
               'reported. (5) The ERA5 height reference switches to bilinear interpolation of model '
               'orography at the exact pairing footprint (class medians -77, -64, -54, -64 m); the '
               'ocean-corner attribution is presented as a hypothesis (regression p = 0.65). (6) '
               'Appendix B is extended with OpenAlex anchor metadata and a forward-and-backward '
               'citation chase from the two anchor works (70 citing works and 27 references; 14 citing '
               'works and 92 references). (7) A sixth limitation is added: no stations at Tarfaya, '
               'Dakhla or Essaouira; observed 10-m wind spans 1.54 to 7.08 m/s, so results are not '
               'reusable for the Atlantic hydrogen hubs. (8) Wording calibrations ("about six times"; '
               'ERA5 12 of 30 without the two no-spell stations; abstract compressed).'),
         why=('Round 3 review; post-hoc additions quoted alongside, never instead of, the pre-specified '
              'tests.'),
         where=('Section 3.1, Section 3.6, Section 4.4, Section 4.7, the sensitivity summary table, the '
                'per-class table, Appendix B, Section 5.4. Outputs carry the _v5 suffix or appear in '
                'revision_v5.json and systematic_search_v5.json.')),
    dict(num=17, date='2026-10-18', title='Review round 4: H1 equivalence analysis (post hoc)',
         trigger='Internal review, round 4', nature='Post-hoc analysis that quantifies without altering',
         what=('Two quantifications of the evidence toward the "transfer weaker than the margin" side '
               'of the frozen H1 decision rule, neither of which alters the rule. (1) A TOST '
               'equivalence test against the pre-specified margin rho = 0.5 built on the frozen '
               'spatial-block bootstrap: one-sided p = 0.050 (difference-based) and 0.025 (log-ratio), '
               'borderline at the 5% level on a bootstrap known to undercover. (2) A one-sided default '
               'Bayes factor (Ly, Verhagen and Wagenmakers 2016) favouring rho < 0.5: 28.4 '
               '(difference-based) and 14.1 (log-ratio), posterior probability 0.988 and 0.977; the '
               'factor collapses to 1.1 when the eight blocks are aggregated to block means, a variant '
               'that is underpowered and is declared uninformative rather than as evidence against. '
               'Sensitivities: prior width 24 to 39; effective n = 23 gives 12.7. The grid integration '
               'is validated three independent ways.'),
         why=('Round 4 review asked for the evidence against strong transfer to be quantified; the '
              'confirmatory verdict remains "inconclusive" and the new quantities accompany it.'),
         where=('Sections 3.4 and 3.6, Section 4.3 (explicitly labelled post hoc), the confirmatory '
                'verdicts table, abstract, Section 5.1, limitations. Artifacts: h1_equivalence_v6.json, '
                'script h1_equivalence_v6.py; two new DOI-verified references (ly2016, lakens2017).')),
    dict(num=18, date='2026-10-19', title='Review round 5: common ICC, storage regression revisited, metric unification, range-first quoting',
         trigger='Internal review, round 5', nature='Calibration, robustness, reporting policy',
         what=('Four groups. (1) The per-test intra-class correlations used by #16 are noisy (raw '
               'estimates from -0.188 to +0.129, two truncating at zero), which had made the H2 '
               'contrast look product-specific; they are replaced by a common ICC with brackets (0.013 '
               'lower, 0.126 upper, pooling ANOVA sums of squares) and a p(ICC) grid over 0 to 0.20: '
               'H2 and H3 remain significant for both products at every ICC in the grid (maximum '
               'p = 0.028). (2) The storage-wind gradient is re-examined with a cross regressor '
               '(interval includes zero), class fixed effects (slope -0.020) and archipelago exclusion '
               '(-0.133); the resource dependence is downgraded to a specification-dependent '
               'hypothesis, with the endogeneity channel documented. (3) Event metrics are unified to '
               'the per-station convention (observed 0.55 events per year and 5.93 days; CERRA 0.42 '
               'and 5.86; ERA5 0.38 and 5.73), and detection gaps are quoted as ranges (POD 0.12 to '
               '0.18; CSI 0.05 to 0.15; frequency -3% to -39%). (4) The "quote the range, not the '
               'extreme" policy is applied to the Bayes factors (13 to 28 across re-expressions) and '
               'to the H4 headline (calibrated value 0.066 quoted). First-party DEM and CDS download '
               'attempts were retried and documented; ready-to-run scripts remain in the deposit.'),
         why=('Round 5 review; honest calibration of the cluster-aware check and of post-hoc gradients. '
              'No confirmatory verdict changes.'),
         where=('Section 3.6, Section 4.7 (the three robustness checks), Section 4.8, abstract, the '
                'confirmatory verdicts table. Outputs: revision_v7.json, o5_wind_regression_v7.csv.')),
    dict(num=19, date='2026-10-20', title='Context bibliographic expansion (23 to 56 references)',
         trigger='Manuscript review round (v9)', nature='Bibliographic expansion, context search',
         what=('The reference list expands from 23 to 56 entries through a context search, kept '
               'separate from the novelty search of Appendix B: Crossref relevance queries in ten '
               'categories, a backward citation chase from the introductions of four anchor works, and '
               'a forward chase of the oldest anchor citing works. Every added entry was verified '
               'entry by entry against its Crossref record at DOI level (complete authors, journal, '
               'volume, issue, pages, year) before inclusion; nothing was fabricated. OpenAlex was '
               'attempted first and rate-limited; its queries are archived. Scopus and Web of Science '
               'were not queried (subscription access), and the appendix states this bound '
               'explicitly.'),
         why=('Each background statement of the introduction, data, methods and discussion now cites '
              'the specific work that supports it; no hypothesis, threshold, station set or statistic '
              'changes.'),
         where=('Section 3.1 (deviation #19 sentence), Appendix B (subsection "Context bibliographic '
                'search (current revision)"), the reference list. Scripts bibcheck_crossref*.py and '
                'archived query dumps accompany the package.')),
]

VERIFY = ('Every claim above is reproducible from the package. The frozen protocol set and the '
          'confirmatory artifacts carry SHA-256 manifests (hashes/), the complete package carries a '
          'SHA-256 manifest with timestamps (MANIFEST.sha256) and an MD5 manifest '
          '(hashes/MD5SUMS_package.txt) that can be checked from the package root with md5sum -c. '
          'Filesystem timestamps of the preregistration files predate the first confirmatory '
          'statistic (2026-10-17); the public deposit postdates the analysis, which is why the '
          'manuscript uses "pre-specified" and declares the timeline explicitly in Section 3.1. The '
          'seed for all resampling is 20261017, and the revision scripts (revise_v3_analysis.py '
          'through revise_v7_analysis.py, h1_equivalence_v6.py) reproduce every number quoted above '
          'from the deposited result sheets.')


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - MARGIN + 14, PAGE_W - MARGIN, PAGE_H - MARGIN + 14)
    canvas.setFont('FreeSerif', 7.5)
    canvas.setFillColor(TEXT_MUTED)
    canvas.drawString(MARGIN, PAGE_H - MARGIN + 18,
                      'Deviation Log | Supplementary material to "Do reanalyses get the droughts right?" (v12)')
    canvas.line(MARGIN, 0.62 * cm, PAGE_W - MARGIN, 0.62 * cm)
    canvas.drawString(MARGIN, 0.45 * cm, 'Pre-specified protocol package v1.2')
    canvas.drawRightString(PAGE_W - MARGIN, 0.45 * cm, str(doc.page))
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN + 16, bottomMargin=MARGIN,
                            title='Deviation Log (supplementary)',
                            author='Z.ai', creator='Z.ai',
                            subject='Deviations #11 to #19 of the pre-specified protocol')
    story = []
    story.append(Paragraph('<b>Deviation Log</b>', title_style))
    story.append(Paragraph('Pre-specified protocol deviations #11 to #19 and post-review additions, '
                           'with manuscript cross-references', kicker))
    story.append(Paragraph('Supplementary material to the manuscript "Do reanalyses get the droughts '
                           'right?" (v12) | Pre-specified protocol package v1.2 | 21 October 2026',
                           kicker))
    story.append(HRFlowable(width='100%', thickness=1.4, color=ACCENT,
                            spaceBefore=6, spaceAfter=14))

    story.extend(heading_block('1. Purpose and reading guide', h1))
    story.append(Paragraph(INTRO_1, body))
    story.append(Paragraph(INTRO_2, body))

    story.extend(heading_block('2. Summary table', h1))
    data = [[Paragraph('<b>#</b>', tbl_head), Paragraph('<b>Date</b>', tbl_head),
             Paragraph('<b>Trigger</b>', tbl_head), Paragraph('<b>Nature</b>', tbl_head),
             Paragraph('<b>Manuscript anchor</b>', tbl_head)]]
    for r in SUMMARY_ROWS:
        data.append([Paragraph(c, tbl_cell) for c in r])
    ratios = [0.06, 0.11, 0.24, 0.28, 0.31]
    col_w = [x * AVAIL_W for x in ratios]
    assert sum(col_w) <= AVAIL_W + 0.5
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign='CENTER')
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), HEADER_FILL),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.4, BORDER),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]
    for i in range(1, len(data)):
        style_cmds.append(('BACKGROUND', (0, i), (-1, i),
                           TABLE_STRIPE if i % 2 == 1 else colors.white))
    t.setStyle(TableStyle(style_cmds))
    story.append(t)
    story.append(Spacer(1, 6))
    story.append(Paragraph('Table 1. Deviations #11 to #19. Manuscript section numbers refer to the '
                           'submitted version (v12): Sections 3 (Methods), 4 (Results), 5 (Discussion), '
                           'Appendices A and B.', note))

    story.extend(heading_block('3. Deviations', h1))
    for d in DEVIATIONS:
        story.extend(deviation_block(d))

    story.extend(heading_block('4. Verification', h1))
    story.append(Paragraph(VERIFY, body))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print('body OK:', OUT)


if __name__ == '__main__':
    build()
