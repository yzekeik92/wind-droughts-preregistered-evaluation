#!/usr/bin/env python3
"""Verify candidate references against Crossref API (real metadata only).

Polite pool (mailto), modest pacing, incremental JSON save after each query.
Output: scripts/bib_candidates_crossref.json + readable digest.
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
MAILTO = "validation-study@example.org"

TITLE_CANDIDATES = {
    "ramon2019":      "What global reanalysis best represents near-surface winds",
    "olauson2018":    "ERA5: The new champion of wind power modelling",
    "staffell2016":   "Using bias-corrected reanalysis to simulate current and future wind power output",
    "vautard2010":    "Northern Hemisphere atmospheric stilling partly attributed to an increase in aerosols",
    "pryor2010":      "Climate change impacts on wind energy: A review",
    "mcgraw1996":     "Forming inferences about some intraclass correlation coefficients",
    "schuirmann1987": "A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability",
    "dunn2012":       "HadISD: A quality-controlled global synoptic report database for selected variables at long-term stations",
    "bollmeyer2015":  "Towards a high-resolution regional reanalysis for the European CORDEX domain",
    "zeng2019":       "A reversal in global terrestrial stilling and its implications for wind energy production",
    "leahy2013":      "Persistence of low wind speed conditions and implications for wind power variability",
    "patlakas2017":   "Low wind speed events: persistence and frequency",
    "gualtieri2012":  "Methods to extrapolate wind resource to the turbine hub height: A review",
    "gualtieri2022":  "Analysing the uncertainties of reanalysis data used for wind energy applications: A review",
    "journee2024":    "On the suitability of global reanalysis datasets for wind energy yield estimation",
    "minola2016":     "Homogenization and trend analysis of wind speed records in Sweden",
    "wan2010":        "Homogenization and trend analysis of Canadian marine surface wind norms",
    "zerrahn2017":    "Long-run power storage requirements for high shares of renewables: results and sensitivities",
    "williams1975":   "The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity",
    "lahiri2003":     "Resampling methods for dependent data",
}

TOPIC_CANDIDATES = {
    "dunkelflaute":    "Dunkelflaute low wind power events Germany",
    "cerra_eval":      "CERRA regional reanalysis wind evaluation Europe",
    "era5_coast":      "ERA5 wind speed bias coastline stations",
    "canary_wind":     "reanalysis wind energy Canary Islands assessment",
    "h2_morocco":      "green hydrogen Morocco north Africa",
    "wind_morocco":    "wind resource assessment Morocco stations",
    "h2_storage":      "hydrogen storage sizing wind electrolyzer buffer",
    "p90_variability": "interannual variability wind resource assessment P90",
    "med_wind_trend":  "surface wind speed trends Mediterranean stations",
    "regional_vs_glo": "regional reanalysis versus global reanalysis wind energy comparison",
}

def get(url, tries=3):
    delay = 4
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"bibcheck-context/1.0 (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                return {"error": str(e)}
            time.sleep(delay)
            delay *= 2

def norm(item):
    auths = []
    for a in (item.get("author") or [])[:6]:
        nm = " ".join(x for x in [a.get("given",""), a.get("family","")] if x)
        if nm: auths.append(nm)
    date = (item.get("issued") or {}).get("date-parts") or [[None]]
    return {
        "title":   (item.get("title") or [""])[0],
        "doi":     item.get("DOI"),
        "year":    date[0][0],
        "journal": (item.get("container-title") or [""])[0] or None,
        "vol":     item.get("volume"),
        "issue":   item.get("issue"),
        "pages":   item.get("page"),
        "authors": auths,
        "type":    item.get("type"),
        "cited_by": item.get("is-referenced-by-count"),
        "publisher": item.get("publisher"),
    }

def main():
    import os
    out_path = "/home/z/my-project/scripts/bib_candidates_crossref.json"
    out = {"title": {}, "topic": {}}
    if os.path.exists(out_path):
        out = json.load(open(out_path))

    def do(key, query, rows, bucket):
        u = f"{BASE}?mailto={MAILTO}&query.bibliographic={urllib.parse.quote(query)}&rows={rows}&select=DOI,title,container-title,volume,issue,page,author,issued,type,is-referenced-by-count,publisher"
        res = get(u)
        if "error" in res:
            out[bucket][key] = [{"error": res["error"]}]
            print(f"{bucket} {key}: ERROR {res['error']}")
        else:
            items = res.get("message", {}).get("items", [])
            out[bucket][key] = [norm(it) for it in items]
            print(f"{bucket} {key}: OK ({len(items)} hits)")
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
        time.sleep(1.0)

    for key, q in TITLE_CANDIDATES.items():
        if key not in out["title"]:
            do(key, q, 3, "title")
    for key, q in TOPIC_CANDIDATES.items():
        if key not in out["topic"]:
            do(key, q, 10, "topic")
    print("done")

if __name__ == "__main__":
    main()
