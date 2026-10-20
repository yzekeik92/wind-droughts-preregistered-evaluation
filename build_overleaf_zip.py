#!/usr/bin/env python3
"""Build a clean Overleaf-ready zip of the v12 manuscript.

Contents (and nothing else; Overleaf compiles from source):
  main.tex, macros.tex, sec_*.tex (7), refs.bib,
  figures/f1..f9 (.pdf only; PNGs are unused duplicates),
  README_Overleaf.txt (compiler/root instructions)

Excluded: main.pdf, main.log, main.blg, macros_manifest.json,
          old versioned PDFs v1..v9.
"""
import zipfile
from pathlib import Path

PAPER = Path("/home/z/my-project/download/paper")
OUT = Path("/home/z/my-project/download/Manuscript_v12_Overleaf.zip")
STAGE = Path("/home/z/my-project/scripts/overleaf_stage")

TEX_FILES = [
    "main.tex", "macros.tex",
    "sec_intro.tex", "sec_data.tex", "sec_methods.tex", "sec_results.tex",
    "sec_discussion.tex", "sec_backmatter.tex", "sec_appendix.tex",
    "refs.bib",
]
FIGS = [
    "f1_map.pdf", "f2_errorstructure.pdf", "f3_h1_scatter.pdf",
    "f4_spell_stats.pdf", "f5_t1t2.pdf", "f6_podfarcsi.pdf",
    "f7_h3_ratios.pdf", "f8_h4_breaks.pdf", "f9_storage.pdf",
]

README = """\
Overleaf upload package: "Do reanalyses get the droughts right?" (v12, 30 pp)

HOW TO USE
1) Overleaf: New Project -> Upload Project -> select this .zip.
2) Root document: main.tex (already detected automatically).
3) Compiler: pdfLaTeX (default). TeX Live 2022 or newer works
   (packages used: newtx, natbib, booktabs, tabularx, adjustbox,
   caption, enumitem, microtype, tikz, hyperref, lineno; all standard).
4) Bibliography: BibTeX via \\bibliography{refs} (refs.bib included;
   style unsrtnat). Overleaf runs it automatically.

NOTES
- macros.tex holds ALL numerical results, auto-generated from the frozen
  result sheets (scripts/gen_macros.py in the deposit). Do not hand-edit:
  run gen_macros.py + check_consistency.py if any number changes.
- Figures are the 9 PDFs under figures/. PNG copies in the repo are
  unused duplicates and were left out of this package.
- refs.bib has 56 entries (v9 context-bibliographic expansion, deviation
  #19). Every entry DOI-verified against Crossref; the kittel2026 entry
  carries the Author Correction (17:6035) in its note field. See
  Appendix B.
- Double-blind: author names/affiliations intentionally withheld.

FILES
main.tex            root document (abstract + highlights + \\input sections)
macros.tex          auto-generated numerical macros (34)
sec_intro/data/methods/results/discussion/backmatter/appendix.tex
refs.bib            56 entries (v9 expansion; corr. folded into kittel2026)
figures/            f1..f9 (PDF, vector)
"""

def main():
    import shutil
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)

    for f in TEX_FILES:
        src = PAPER / f
        assert src.exists(), f"missing {src}"
        shutil.copy2(src, STAGE / f)

    for f in FIGS:
        src = PAPER / "figures" / f
        assert src.exists(), f"missing {src}"
        shutil.copy2(src, STAGE / "figures" / f)

    (STAGE / "README_Overleaf.txt").write_text(README, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(STAGE).as_posix())

    names = zipfile.ZipFile(OUT).namelist()
    print(f"OK {OUT}  ({OUT.stat().st_size/1024:.0f} KiB, {len(names)} files)")
    for n in names:
        print("  ", n)

if __name__ == "__main__":
    main()
