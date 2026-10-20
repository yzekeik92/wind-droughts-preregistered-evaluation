#!/usr/bin/env python3
"""Pass 3: full metadata (all authors, article numbers, pages) by direct DOI
lookup for the selected candidates + one title query (Otero 2022)."""
import json
import time
import urllib.parse
import urllib.request

BASE = "https://api.crossref.org/works"
MAILTO = "validation-study@example.org"

DOI_FULL = [
    "10.1002/qj.3616",            # ramon2019
    "10.1016/j.renene.2018.03.056", # olauson2018
    "10.1016/j.energy.2016.08.068", # staffell2016
    "10.1038/ngeo979",            # vautard2010
    "10.1016/j.rser.2009.07.028", # pryor2010
    "10.1037/1082-989X.1.1.30",   # mcgraw1996
    "10.1007/BF01068419",         # schuirmann1987
    "10.5194/cp-8-1649-2012",     # dunn2012
    "10.1002/qj.2486",            # bollmeyer2015
    "10.1038/s41558-019-0622-6",  # zeng2019
    "10.1002/we.1509",            # leahy2013
    "10.1002/we.2078",            # patlakas2017
    "10.1016/j.renene.2011.12.022", # gualtieri2012
    "10.1016/j.rser.2022.112741", # gualtieri2022
    "10.1175/JCLI-D-15-0636.1",   # minola2016
    "10.1175/2009JCLI3200.1",     # wan2010
    "10.1016/j.rser.2017.05.205", # schill2018 (storage)
    "10.2307/2529820",            # williams1975
    "10.1038/s41467-026-72681-5", # kittel2026
    "10.5194/asr-22-69-2025",     # patra2025
    "10.5194/asr-17-63-2020",     # jourdier2020
    "10.1016/j.energy.2017.10.084", # schallenberg2018
    "10.5194/wes-3-651-2018",     # pryor2018
    "10.1002/we.2759",            # davidson2022
    "10.1016/j.ijhydene.2022.12.362", # ourya2023
    "10.3390/jmse12050774",       # taroual2024
    "10.1002/joc.925",            # pirazzoli2003
    "10.5194/wes-9-1727-2024",    # gandoin2024
    "10.1016/j.heliyon.2019.e01830", # elkhchine2019
    "10.1016/j.rser.2015.02.056", # kousksou2015
    "10.1016/0960-1481(91)90095-7", # nfaoui1991
    "10.1088/1742-6596/2875/1/012016", # rouholahnejad2024
]

TITLE_EXTRA = {
    "otero2022": "copula-based assessment of renewable energy droughts across Europe",
}

SELECT = "DOI,title,container-title,volume,issue,page,author,issued,type,is-referenced-by-count,article-number"

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
    for a in (item.get("author") or []):
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
        "artno": item.get("article-number"),
        "authors": auths,
        "type": item.get("type"),
        "cited_by": item.get("is-referenced-by-count"),
    }

def main():
    out_path = "/home/z/my-project/scripts/bib_selected_crossref.json"
    out = {}
    for doi in DOI_FULL:
        u = f"{BASE}/{urllib.parse.quote(doi)}?mailto={MAILTO}"
        res = get(u)
        if "error" in res:
            out[doi] = {"error": res["error"]}
            print(f"{doi}: ERROR {res['error']}")
        else:
            out[doi] = norm(res.get("message", {}))
            print(f"{doi}: OK  {out[doi]['year_issued']} | {out[doi]['journal']}")
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
        time.sleep(1.0)
    for key, q in TITLE_EXTRA.items():
        u = f"{BASE}?mailto={MAILTO}&query.bibliographic={urllib.parse.quote(q)}&rows=2&select={SELECT}"
        res = get(u)
        items = res.get("message", {}).get("items", []) if "error" not in res else []
        out[key] = [norm(it) for it in items] or [{"error": res.get("error", "none")}]
        print(f"{key}: {len(items)} hits")
        json.dump(out, open(out_path, "w"), indent=1, ensure_ascii=False)
        time.sleep(1.0)
    print("done")

if __name__ == "__main__":
    main()
