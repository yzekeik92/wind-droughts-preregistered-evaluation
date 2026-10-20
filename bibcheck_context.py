#!/usr/bin/env python3
"""Verify candidate references against OpenAlex (real metadata only).

v2: polite pool (mailto in URL), exponential backoff on HTTP 429,
1 s between calls, title-match scoring. No reference enters refs.bib
unless it comes back from this API with matching metadata.
"""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.openalex.org/works"
MAILTO = "validation-study@example.org"

# title searches: key -> expected title fragment
TITLE_CANDIDATES = {
    "ramon2019":      "What global reanalysis best represents near-surface winds",
    "olauson2018":    "ERA5: The new champion of wind power modelling",
    "staffell2016":   "Using bias-corrected reanalysis to simulate current and future wind power output",
    "vautard2010":    "Northern Hemisphere atmospheric stilling partly attributed to",
    "pryor2010":      "Climate change impacts on wind energy a review",
    "mcgraw1996":     "Forming inferences about some intraclass correlation coefficients",
    "schuirmann1987": "A comparison of the two one-sided tests procedure and the power approach",
    "dunn2012":       "HadISD: A quality-controlled global synoptic report database for selected variables",
    "bollmeyer2015":  "Towards a high-resolution regional reanalysis for the European CORDEX domain",
    "zeng2019":       "A reversal in global terrestrial stilling and its implications for wind energy production",
    "leahy2013":      "Persistence of low wind speed conditions and implications for wind power variability",
    "patlakas2017":   "Low wind speed events: persistence and frequency",
    "gualtieri2012":  "Methods to extrapolate wind resource to the turbine hub height",
    "gualtieri2022":  "Analysing the uncertainties of reanalysis data used for wind energy applications",
    "journee2024":    "On the suitability of global reanalysis datasets for wind energy",
    "minola2016":     "Homogenization and trend analysis of wind speed records in Sweden",
    "wan2010":        "Homogenization and trend analysis of Canadian marine surface wind norms",
    "zerrahn2017":    "Long-run power storage requirements for high shares of renewables",
    "williams1975":   "The analysis of binary responses from toxicological experiments involving reproduction and teratogenicity",
    "lahiri2003":     "Resampling methods for dependent data",
}

# topical searches for the categories I could not pin to a known paper
TOPIC_CANDIDATES = {
    "dunkelflaute":    "Dunkelflaute",
    "cerra_eval":      "CERRA reanalysis wind evaluation",
    "era5_coast":      "ERA5 wind speed bias coastline",
    "canary_wind":     "reanalysis wind energy Canary Islands",
    "h2_morocco":      "green hydrogen Morocco",
    "wind_morocco":    "wind resource assessment Morocco",
    "h2_storage":      "hydrogen storage sizing wind electrolyzer",
    "p90_variability": "interannual variability wind resource P90 bankability",
    "med_wind_trend":  "surface wind speed trends Mediterranean",
    "regional_vs_glo": "regional reanalysis global reanalysis comparison wind energy",
}

SELECT = "display_name,doi,publication_year,biblio,primary_location,authorships,type,cited_by_count"

def get(url, tries=6):
    delay = 2
    last_err = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": f"bibcheck-context/1.0 (mailto:{MAILTO})"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 429:
                time.sleep(delay)
                delay = min(delay * 2, 45)
                continue
            return {"error": last_err}
        except Exception as e:
            last_err = str(e)
            time.sleep(delay)
            delay = min(delay * 2, 45)
    return {"error": last_err}

def norm(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    auths = [a["author"]["display_name"] for a in (work.get("authorships") or [])[:6]]
    b = work.get("biblio") or {}
    pages = b.get("first_page") or ""
    if b.get("last_page"):
        pages = f"{pages}--{b.get('last_page')}"
    return {
        "title":   work.get("display_name"),
        "doi":     (work.get("doi") or "").replace("https://doi.org/", ""),
        "year":    work.get("publication_year"),
        "journal": src.get("display_name"),
        "vol":     b.get("volume"),
        "issue":   b.get("issue"),
        "pages":   pages,
        "authors": auths,
        "type":    work.get("type"),
        "cited_by": work.get("cited_by_count"),
    }

def main():
    out = {"title": {}, "topic": {}}

    for key, frag in TITLE_CANDIDATES.items():
        u = f"{BASE}?mailto={MAILTO}&search={urllib.parse.quote(frag)}&select={SELECT}&per-page=3"
        res = get(u)
        if "error" in res:
            out["title"][key] = [{"error": res["error"]}]
            print(f"{key}: ERROR {res['error']}")
        else:
            out["title"][key] = [norm(w) for w in res.get("results", [])]
            top = out["title"][key][0] if out["title"][key] else None
            print(f"{key}: count={res.get('meta',{}).get('count')} top={(top or {}).get('title','')[:80]}")
        time.sleep(1.2)

    for key, topic in TOPIC_CANDIDATES.items():
        u = f"{BASE}?mailto={MAILTO}&search={urllib.parse.quote(topic)}&select={SELECT}&per-page=10"
        res = get(u)
        if "error" in res:
            out["topic"][key] = [{"error": res["error"]}]
            print(f"topic {key}: ERROR {res['error']}")
        else:
            out["topic"][key] = [norm(w) for w in res.get("results", [])]
            print(f"topic {key}: count={res.get('meta',{}).get('count')}")
        time.sleep(1.2)

    with open("/home/z/my-project/scripts/bib_candidates_openalex.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nJSON dump -> /home/z/my-project/scripts/bib_candidates_openalex.json")

if __name__ == "__main__":
    main()
