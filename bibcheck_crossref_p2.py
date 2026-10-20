#!/usr/bin/env python3
"""Pass 2: verify the candidates that failed match or were newly discovered."""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
MAILTO = "validation-study@example.org"

QUERIES = {
    # re-queries for the four that missed
    "pryor2010":      "Pryor Barthelmie climate change impacts on wind energy a review Renewable and Sustainable Energy Reviews",
    "zeng2019":       "Zeng reversal in global terrestrial stilling implications for wind energy production Nature Geoscience",
    "journee2024":    "Journee suitability of global reanalysis datasets for wind energy yield estimation Scientific Reports",
    "minola2016":     "Minola homogenization trend analysis wind speed records Sweden",
    # newly discovered / to pin down
    "dunn2012erl":    "HadISD quality-controlled global synoptic report database selected variables long-term stations 1973-2011",
    "kittel2025":     "Long-duration electricity storage needs for coping with Dunkelflaute events in Europe",
    "patra2025":      "Evaluation of ERA5 COSMO-REA6 and CERRA in simulating wind speed along the French coasts",
    "jourdier2023":   "Evaluation of CERRA for wind energy applications Jourdier",
    "schallenberg2018":"Spatial planning to estimate the offshore wind energy potential in coastal regions and islands Canary",
    "pryor2018":      "Interannual variability of wind climates and wind turbine annual energy production",
    "davidson2022":   "Limitations of reanalysis data for wind power applications",
    "ourya2023":      "Assessment of green hydrogen production in Morocco",
    "taroual2024":    "Marine renewable driven green hydrogen production case Morocco wind wave",
    "soukissian2024": "evaluation of offshore wind climate in Greece using the CERRA reanalysis dataset",
    "pirazzoli2003":  "Recent near-surface wind changes in the central Mediterranean and Alpine areas",
    "gandoin2024":    "Underestimation of strong wind speeds offshore in ERA5",
    "collocating":    "Collocating wind data case study verification of the CERRA dataset",
    "elkhchine2019":  "wind energy potential assessment Morocco El Khchine Sriti",
    "kousksou2015":   "Morocco energy strategy renewable energy wind Kousksou",
}

SELECT = "DOI,title,container-title,volume,issue,page,author,issued,type,is-referenced-by-count,publisher,created,issued,deposited"

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
            time.sleep(delay); delay *= 2

def norm(item):
    auths = []
    for a in (item.get("author") or [])[:8]:
        nm = " ".join(x for x in [a.get("given",""), a.get("family","")] if x)
        if nm: auths.append(nm)
    issued = (item.get("issued") or {}).get("date-parts") or [[None]]
    return {
        "title": (item.get("title") or [""])[0],
        "doi": item.get("DOI"),
        "year_issued": issued[0][0],
        "journal": (item.get("container-title") or [""])[0] or None,
        "vol": item.get("volume"),
        "issue": item.get("issue"),
        "pages": item.get("page"),
        "authors": auths,
        "type": item.get("type"),
        "cited_by": item.get("is-referenced-by-count"),
        "publisher": item.get("publisher"),
    }

def main():
    out_path = "/home/z/my-project/scripts/bib_candidates_crossref_p2.json"
    out = {}
    for key, q in QUERIES.items():
        u = f"{BASE}?mailto={MAILTO}&query.bibliographic={urllib.parse.quote(q)}&rows=3&select={SELECT}"
        res = get(u)
        if "error" in res:
            out[key] = [{"error": res["error"]}]
            print(f"{key}: ERROR {res['error']}")
        else:
            items = res.get("message", {}).get("items", [])
            out[key] = [norm(it) for it in items]
            print(f"{key}: OK")
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
        time.sleep(1.0)
    print("done")

if __name__ == "__main__":
    main()
