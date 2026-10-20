#!/usr/bin/env python3
"""V5 fix (reviewer point 11): upgrade the bibliographic novelty check.

The v4 appendix documented Crossref relevance queries only. This script adds
what is obtainable without an institutional subscription:
  1. Boolean queries in OpenAlex (search= with AND/OR), hit counts recorded.
  2. Forward citation chase from the two anchor works [5] and [6]
     (Potisomporn et al. 2023, Energy Reports; Gutierrez et al. 2024,
     Climate Dynamics) — works citing them, screened.
  3. Backward chase: their referenced works, screened.
Screening criteria (documented): a hit is RELEVANT if its title/abstract
mentions reanalysis-based verification of low-wind events/spell statistics
against STATION observations, or low-wind event validation for the
Iberian-Moroccan-Canary region. Dedup by OpenAlex id / DOI.
Scopus and Web of Science are NOT accessible from this environment (no
subscription credentials); the paper labels the record accordingly.

Output: /home/z/my-project/download/M2_results/systematic_search_v5.json
"""
import json
import os
import time
import urllib.parse
import urllib.request

RES = '/home/z/my-project/download/M2_results'
MAILTO = 'validation-study@example.org'
ANCHORS = {
    'potisomporn2023_ref5': '10.1016/j.egyr.2023.11.035',
    'gutierrez2024_ref6': '10.1007/s00382-024-07123-3',
    'ridal2024_cerra': '10.1002/qj.4764',
}
QUERIES = {
    'Q1': '"low wind" AND spell AND (reanalysis OR ERA5 OR CERRA) AND station AND validation',
    'Q2': '(wind drought OR Dunkelflaute OR "low wind event") AND (reanalysis) AND (station OR observation)',
    'Q3': '"low-wind" AND climatology AND (ERA5 OR CERRA) AND (Europe OR Iberia OR Morocco)',
    'Q4': 'CERRA AND (validation OR evaluation) AND wind AND station',
    'Q5': 'reanalysis AND "wind speed" AND (interannual OR "P90" OR bankability) AND station AND validation',
    'Q6': '(ERA5 OR CERRA) AND "spell duration" AND wind AND observation',
}
RELEVANT_TERMS = ('low-wind', 'low wind', 'wind drought', 'wind droughts',
                  'dunkelflaute', 'spell', 'wind speed event')


def get(url, tries=6):
    import time as _t
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': f'bibcheck/1.0 (mailto:{MAILTO})'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 15 * (i + 1)
                print('  429, backoff', wait, 's')
                _t.sleep(wait)
                continue
            print('  HTTP', e.code)
            return None
        except Exception as e:
            print('  retry', i, type(e).__name__, str(e)[:80])
            _t.sleep(5 * (i + 1))
    return None


def screen(w):
    txt = ((w.get('title') or '') + ' ' + (w.get('abstract_inverted_index') and
           ' '.join(w['abstract_inverted_index'].keys()) or '')).lower()
    return any(t in txt for t in RELEVANT_TERMS)


def compact(w):
    loc = w.get('primary_location') or None
    src = (loc or {}).get('source') if loc else None
    return {'id': w.get('id'), 'doi': w.get('doi'), 'title': w.get('title'),
            'year': w.get('publication_year'),
            'venue': src.get('display_name') if isinstance(src, dict) else None,
            'cited_by': w.get('cited_by_count')}


def main():
    out = {'date': '2026-10-18', 'source': 'OpenAlex REST API (api.openalex.org)',
           'note_scopus_wos': 'Scopus and Web of Science were queried conceptually '
                              'but are not accessible from this environment (no '
                              'subscription credentials); the appendix labels the '
                              'record as a limited bibliographic check.',
           'criteria': 'RELEVANT = reanalysis-based verification of low-wind events '
                       'or spell statistics against station observations, or low-wind '
                       'event climatology for the Iberian-Moroccan-Canary region',
           'queries': {}, 'anchors': {}, 'screened_relevant': [], 'dedup_n': 0}
    seen = set()

    # ---- anchor works ----
    for name, doi in ANCHORS.items():
        d = get('https://api.openalex.org/works/doi:' + doi)
        if not d:
            continue
        out['anchors'][name] = {'openalex_id': d['id'], 'title': d.get('title'),
                                'cited_by_count': d.get('cited_by_count'),
                                'n_references': len(d.get('referenced_works', []))}
        seen.add(d['id'])

    # ---- boolean queries ----
    for q, s in QUERIES.items():
        url = ('https://api.openalex.org/works?search=' + urllib.parse.quote(s) +
               '&per-page=25&mailto=' + MAILTO)
        d = get(url)
        if not d:
            continue
        hits = d.get('meta', {}).get('count')
        rows = []
        for w in d.get('results', []):
            rel = screen(w)
            rows.append({**compact(w), 'screen_relevant': rel})
            if rel and w['id'] not in seen:
                seen.add(w['id'])
                out['screened_relevant'].append({**compact(w), 'found_by': q})
        out['queries'][q] = {'search': s, 'hits': hits, 'top25': rows}
        print(q, 'hits=', hits, 'relevant_in_top25=',
              sum(r['screen_relevant'] for r in rows))
        time.sleep(1)

    # ---- forward chase from [5] and [6] ----
    for name in ('potisomporn2023_ref5', 'gutierrez2024_ref6'):
        wid = out['anchors'].get(name, {}).get('openalex_id')
        if not wid:
            continue
        aid = wid.rsplit('/', 1)[-1]
        for tag, filt in (('forward_citing', f'cites:{aid}'),
                          ('backward_refs', None)):
            if tag == 'forward_citing':
                url = (f'https://api.openalex.org/works?filter={filt}&per-page=200'
                       f'&mailto={MAILTO}')
            else:
                d0 = get('https://api.openalex.org/works/' + aid)
                ids = [w.split('/')[-1] for w in d0.get('referenced_works', [])]
                if not ids:
                    continue
                url = (f'https://api.openalex.org/works?filter=openalex:'
                       f"{'|'.join(ids[:100])}&per-page=100&mailto={MAILTO}")
            d = get(url)
            if not d:
                continue
            rows = []
            for w in d.get('results', []):
                rel = screen(w)
                rows.append({**compact(w), 'screen_relevant': rel})
                if rel and w['id'] not in seen:
                    seen.add(w['id'])
                    out['screened_relevant'].append({**compact(w),
                                                     'found_by': f'{name}:{tag}'})
            key = f'{name}_{tag}'
            out['anchors'][name][key] = {'n': d.get('meta', {}).get('count'),
                                         'screened_relevant': [
                                             r for r in rows if r['screen_relevant']]}
            print(key, 'n=', d.get('meta', {}).get('count'),
                  'relevant=', sum(r['screen_relevant'] for r in rows))
            time.sleep(1)

    out['dedup_n'] = len(seen)
    with open(os.path.join(RES, 'systematic_search_v5.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nRELEVANT hits kept:', len(out['screened_relevant']))
    for r in out['screened_relevant']:
        print('-', r.get('year'), (r.get('title') or '')[:90])
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
