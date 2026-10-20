#!/usr/bin/env python3
"""Audit hardcoded citation numbers [n] in the manuscript sources.

The bibliography uses natbib/unsrtnat: reference numbers are assigned by
order of first citation, so any hand-written [n] in the text drifts
silently whenever entries are inserted or removed. This is exactly how
the Appendix-B anchor numbers ([5]/[6], written in an early revision)
ended up pointing at Otero/Leahy after the bibliography grew.

What this script checks
 1. Citation order: main.tex \\input order -> \\cite/\\citep/\\citet/
    \\citealp keys in document order -> the unsrtnat number of each key.
 2. Key integrity: every cited key exists in refs.bib, and every
    refs.bib entry is cited somewhere (uncited entries silently
    disappear from unsrtnat and renumber the whole list).
 3. Hardcoded-number scan: every bare [n] (or [n, m, ...]) token in the
    sources is resolved against the computed numbering and validated:
    the surrounding paragraph must mention the referenced entry's
    authors -- either literally (\\citet names) or via another \\cite
    key in the same paragraph. If the paragraph names a DIFFERENT
    entry's authors instead, that is a hard failure with a hint.

Exit 0 = all clear; 1 = failures.  Run with --selftest to exercise the
detector on the historical Appendix-B bug (expects exactly 2 failures).
"""
import re
import sys
import unicodedata
from pathlib import Path

PAP = Path('/home/z/my-project/download/paper')
BIB = PAP / 'refs.bib'
MAIN = PAP / 'main.tex'

HARDCODED = re.compile(r'\[\d+(?:\s*,\s*\d+)*\]')
CITE = re.compile(r'\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}')
INPUT = re.compile(r'\\input\{([^}]*)\}')


def _norm(s):
    s = unicodedata.normalize('NFD', s)
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()


def parse_bib():
    """key -> dict(surnames=[...], first=...)"""
    text = BIB.read_text(encoding='utf8')
    entries = {}
    for m in re.finditer(r'@(\w+)\s*\{\s*([\w:+.-]+)\s*,', text):
        kind, key = m.group(1).lower(), m.group(2)
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        block = text[m.end():i - 1]
        if kind in ('comment', 'string', 'preamble'):
            continue
        am = re.search(r'author\s*=\s*[{"]*(.*?)[}"]*\s*,?\n\s*(?:title|journal|booktitle|volume|year|pages|doi|note|publisher|howpublished|editor|address|url|urldate|organization)\s*=', block, re.S)
        am2 = re.search(r'author\s*=\s*\{(.*?)\}', block, re.S)
        raw = (am2.group(1) if am2 else (am.group(1) if am else ''))
        surnames = []
        for a in raw.split(' and '):
            a = a.strip().strip('{}')
            if not a:
                continue
            sur = a.split(' and ')[0]
            sur = sur.split(',')[0] if ',' in sur else sur.split()[-1]
            sur = sur.replace('{', '').replace('}', '').strip()
            if sur and sur.lower() not in ('others', 'and', 'etal', 'et al.'):
                surnames.append(sur)
        entries[key] = {'surnames': surnames,
                        'norm': {_norm(s) for s in surnames}}
    return entries


def citation_order(entries):
    """unsrtnat number per key, following main.tex \\input order."""
    order = []
    seen = set()

    def visit(path):
        text = path.read_text(encoding='utf8')
        body = text
        for m in re.finditer(CITE, body):
            for k in m.group(1).split(','):
                k = k.strip()
                if k and k not in seen:
                    seen.add(k)
                    order.append(k)
        for m in INPUT.finditer(text):
            inc = m.group(1).strip()
            p = PAP / (inc if inc.endswith('.tex') else inc + '.tex')
            if p.exists():
                visit(p)

    visit(MAIN)
    return {k: i + 1 for i, k in enumerate(order)}


def paragraphs(fname):
    t = (PAP / fname).read_text(encoding='utf8')
    out, cur = [], []
    for line in t.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            out.append('\n'.join(cur))
            cur = []
    if cur:
        out.append('\n'.join(cur))
    return out


def audit(entries, numbering, files):
    problems = []
    # 2. key integrity
    cited = set(numbering)
    for k in cited:
        if k not in entries:
            problems.append(f"\\cite{{{k}}} has no entry in refs.bib")
    for k in entries:
        if k not in cited:
            problems.append(
                f"refs.bib entry '{k}' is never cited (unsrtnat drops it and renumbers)")
    # 3. hardcoded [n]
    for fname in files:
        for para in paragraphs(fname):
            for m in HARDCODED.finditer(para):
                nums = [int(x) for x in re.findall(r'\d+', m.group(0))]
                # window surnames: literal \\citet names + surnames of
                # entries cited via \\cite* inside the same paragraph
                win = set()
                for cm in CITE.finditer(para):
                    for k in cm.group(1).split(','):
                        k = k.strip()
                        if k in entries:
                            win |= entries[k]['norm']
                win |= {_norm(w) for w in
                        re.findall(r'[A-ZÀ-Ž][\wÀ-ž\'\-]+', para)}
                for n in nums:
                    key = next((k for k, v in numbering.items() if v == n), None)
                    if key is None:
                        problems.append(
                            f"{fname}: hardcoded [{n}] exceeds the reference list")
                        continue
                    ens = entries.get(key, {'norm': set()})['norm']
                    if ens & win:
                        continue          # a named author matches entry n
                    # hint: which cited entry does the paragraph look like?
                    hit = [k for k in entries
                           if k in cited and k != key
                           and entries[k]['norm'] & win]
                    problems.append(
                        f"{fname}: hardcoded [{n}] resolves to '{key}' "
                        f"(#{n}) but the paragraph names none of its authors"
                        + (f" -- paragraph instead matches {[h for h in hit if h != key][:3]}"
                           if hit else ""))
    return problems


def run(quiet=False):
    entries = parse_bib()
    numbering = citation_order(entries)
    files = [m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex'
             for m in INPUT.finditer(MAIN.read_text(encoding='utf8'))]
    files = [f for f in files if (PAP / f).exists()]
    problems = audit(entries, numbering, files)
    if not quiet:
        print(f"  bib entries: {len(entries)} | cited (unsrtnat order): "
              f"{len(numbering)}")
        if problems:
            for p in problems:
                print("  FAIL", p)
        else:
            print("  PASS no hardcoded citation numbers; all keys wired")
    return problems


def selftest():
    """The historical v10 bug: [5]/[6] were written when potisomporn2023
    and gutierrez2024 were refs 5 and 6; after insertions [5]=otero2022
    and [6]=leahy2013, while the paragraph itself cites the anchor works
    potisomporn2023/gutierrez2024. The audit must flag both hardcoded
    numbers and hint at the two anchor keys."""
    global PAP, BIB, MAIN
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    bib = [
        ('a2020', 'Adler, Anne'), ('b2021', 'Brown, Bob'),
        ('c2022', 'Clark, Carol'), ('d2023', 'Dunn, Dave'),
        ('otero2022', 'Otero, Noelia and Martius, Olivia'),
        ('leahy2013', 'Leahy, Paul and McKeogh, Eamon'),
        ('patlakas2017', 'Patlakas, Platon and others'),
        ('potisomporn2023', 'Potisomporn, Pim and others'),
        ('gutierrez2024', 'Guti{\\\'e}rrez, Juan and others'),
    ]
    (tmp / 'refs.bib').write_text('\n'.join(
        f'@article{{{k}, author={{{a}}}, title={{T}}, journal={{J}}, year={y}}}'
        for (k, a), y in zip(bib, range(2020, 2029))))
    (tmp / 'main.tex').write_text('\\input{sec_intro}\\input{sec_appendix}\n')
    (tmp / 'sec_intro.tex').write_text(
        '\\citep{a2020} \\citep{b2021} \\citep{c2022} \\citep{d2023} '
        '\\citep{otero2022} \\citep{leahy2013,patlakas2017}\n')
    (tmp / 'sec_appendix.tex').write_text(
        'a citation chase from the two anchor works [5] and [6] '
        '\\citep{potisomporn2023,gutierrez2024}.\n\n'
        'The chase covered all works citing [5] (70) and all works citing [6] (14) '
        'together with [6]\'s 92 references and [5]\'s 27 references.\n')
    PAP, BIB, MAIN = tmp, tmp / 'refs.bib', tmp / 'main.tex'
    entries = parse_bib()
    numbering = citation_order(entries)
    assert numbering['otero2022'] == 5 and numbering['leahy2013'] == 6, \
        f'fixture numbering wrong: {numbering}'
    problems = audit(entries, numbering, ['sec_appendix.tex'])
    # para 1: [5] and [6] next to \citep{potisomporn,gutierrez} -> hinted;
    # para 2: four hand numbers ([5] [6] [6]'s [5]'s) with no \cite -> bare.
    ok = (len(problems) == 6 and
          sum("resolves to 'otero2022'" in p for p in problems) == 3 and
          sum("resolves to 'leahy2013'" in p for p in problems) == 3 and
          sum('paragraph instead matches' in p for p in problems) == 2 and
          any('potisomporn2023' in p for p in problems))
    print(f"  selftest: {len(problems)} problems detected "
          f"({'DETECTOR OK' if ok else 'DETECTOR BROKEN'})")
    for p in problems:
        print('   -', p)
    return 0 if ok else 1


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        sys.exit(selftest())
    sys.exit(1 if run() else 0)
