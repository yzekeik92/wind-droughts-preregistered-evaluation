#!/usr/bin/env python3
"""V5 fix (reviewer point 11): bibliographic check for the novelty claim.

Upgrade over the v4 Crossref-only record:
  - Anchor works [5] (Potisomporn et al. 2023, Energy Reports,
    DOI 10.1016/j.egyr.2023.11.035) and [6] (Gutierrez et al. 2024, Climate
    Dynamics, DOI 10.1007/s00382-024-07123-3), plus the CERRA system paper
    (Ridal et al. 2024, QJRMS).
  - Forward citation chase (works citing [5] and [6]) and backward chase
    (their references) via the Semantic Scholar Graph API, screened by
    documented criteria, deduplicated.
  - Boolean/keyword queries in Semantic Scholar + the v4 Crossref queries.
  - OpenAlex anchor metadata (ids/citation counts) recorded via WebFetch.
Scopus and Web of Science are NOT accessible from this environment (no
subscription credentials); the appendix labels this a limited bibliographic
check accordingly.

Output: /home/z/my-project/download/M2_results/systematic_search_v5.json
"""
import json
import os
import time
import urllib.parse
import urllib.request

RES = '/home/z/my-project/download/M2_results'
BASE = 'https://api.semanticscholar.org/graph/v1'
RELEVANT_TERMS = ('low wind', 'low-wind', 'wind drought', 'dunkelflaute',
                  'wind speed event', 'spell', 'energy drought', 'wind lull')
REGION_TERMS = ('iberia', 'iberian', 'morocco', 'moroccan', 'canary',
                'canaries', 'north-west africa', 'northwest africa', 'maghreb')


def get(url, tries=5):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'bibcheck-v5/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 10 * (i + 1)
                print(f'  429 backoff {wait}s')
                time.sleep(wait)
                continue
            print('  HTTP', e.code, url[:90])
            return None
        except Exception as e:
            print('  err', type(e).__name__, str(e)[:70])
            time.sleep(3)
    return None


def screen(title):
    t = (title or '').lower()
    if not t:
        return False
    windish = ('wind' in t) or ('reanalys' in t) or ('reanalyz' in t) or ('era5' in t)
    rel = windish and any(k in t for k in RELEVANT_TERMS)
    reg = any(k in t for k in REGION_TERMS)
    return rel or (windish and reg)


def compact(p, extra=None):
    d = {'title': p.get('title'), 'year': p.get('year'), 'venue': p.get('venue'),
         'paperId': p.get('paperId'), 'doi': (p.get('externalIds') or {}).get('DOI'),
         'screen_relevant': screen(p.get('title'))}
    if extra:
        d.update(extra)
    return d


def main():
    out = {
        'date': '2026-10-18',
        'sources': 'Crossref REST API (v4 record, retained), Semantic Scholar '
                   'Graph API (citations/references chase + keyword queries), '
                   'OpenAlex anchor metadata',
        'note_scopus_wos': 'Scopus and Web of Science require subscription '
                           'credentials not available in this environment; the '
                           'manuscript therefore labels this a limited '
                           'bibliographic check, not a systematic review.',
        'criteria': 'RELEVANT if title links reanalysis/wind data with low-wind '
                    'events, wind droughts, Dunkelflaute, spells or energy '
                    'droughts, OR concerns wind validation/climatology for the '
                    'Iberian-Moroccan-Canary region; titles screened, '
                    'deduplicated by paperId/DOI.',
        'anchors': {}, 'queries': {}, 'chases': {}, 'relevant_hits': [],
    }
    seen = set()

    anchors = {
        'potisomporn2023_ref5': {
            'doi': '10.1016/j.egyr.2023.11.035',
            'openalex': 'W4388839998 (70 citing works, 27 references, via '
                        'api.openalex.org/works/doi:..., 2026-10-18)',
            's2_lookup': 'title search',
        },
        'gutierrez2024_ref6': {
            'doi': '10.1007/s00382-024-07123-3',
            'openalex': 'W4392096999 (14 citing works, 92 references, via '
                        'api.openalex.org/works/doi:..., 2026-10-18)',
            's2_lookup': 'DOI',
        },
    }
    # resolve [5] by title ([5]'s DOI is not indexed by S2)
    d = get(BASE + '/paper/search?query=' + urllib.parse.quote(
        'Evaluating ERA5 reanalysis predictions of low wind speed events around the UK')
        + '&fields=title,year,externalIds&limit=5')
    pid5 = None
    if d and d.get('data'):
        for p in d['data']:
            if p.get('title', '').lower().startswith('evaluating era5 reanalysis'):
                pid5 = p['paperId']
                break
    d6 = get(BASE + '/paper/DOI:10.1007/s00382-024-07123-3?fields=title,year,citationCount,referenceCount')
    pid6 = d6.get('paperId') if d6 else None
    out['anchors'] = {
        'potisomporn2023_ref5': {**anchors['potisomporn2023_ref5'],
                                 's2_paperId': pid5},
        'gutierrez2024_ref6': {**anchors['gutierrez2024_ref6'],
                               's2_paperId': pid6,
                               'title': d6.get('title') if d6 else None,
                               'citationCount': d6.get('citationCount') if d6 else None,
                               'referenceCount': d6.get('referenceCount') if d6 else None},
    }
    time.sleep(2)

    def chase(pid, name, tag, n_expected):
        if not pid:
            return
        if tag == 'forward':
            url = (BASE + f'/paper/{pid}/citations?fields=title,year,venue,externalIds'
                   f'&limit=100')
        else:
            url = (BASE + f'/paper/{pid}/references?fields=title,year,venue,externalIds'
                   f'&limit=100')
        d = get(url)
        if not d or 'data' not in d:
            return
        rows = []
        for it in d['data']:
            p = it.get('citingPaper') or it.get('citedPaper') or {}
            c = compact(p, {'found_by': f'{name}:{tag}'})
            rows.append(c)
            if c['screen_relevant'] and (c['paperId'] not in seen):
                seen.add(c['paperId'])
                out['relevant_hits'].append(c)
        out['chases'][f'{name}_{tag}'] = {
            'n_returned': len(rows), 'n_expected': n_expected,
            'relevant': [r for r in rows if r['screen_relevant']]}
        print(name, tag, 'returned', len(rows), 'relevant',
              sum(r['screen_relevant'] for r in rows))
        time.sleep(2)

    chase(pid5, 'ref5_potisomporn2023', 'forward', 70)
    chase(pid5, 'ref5_potisomporn2023', 'backward', 27)
    chase(pid6, 'ref6_gutierrez2024', 'forward', 14)
    chase(pid6, 'ref6_gutierrez2024', 'backward', 92)

    queries = {
        'S1': 'low wind spell reanalysis validation station',
        'S2': 'wind drought reanalysis station observations persistence',
        'S3': 'CERRA reanalysis wind validation',
        'S4': 'Dunkelflaute reanalysis events',
        'S5': 'reanalysis interannual variability wind P90 bankability station',
        'S6': 'low wind climatology ERA5 Europe Iberia Morocco',
    }
    for q, s in queries.items():
        d = get(BASE + '/paper/search?query=' + urllib.parse.quote(s) +
                '&fields=title,year,venue,externalIds&limit=20')
        if not d:
            continue
        rows = [compact(p, {'found_by': f'query:{q}'}) for p in d.get('data', [])]
        out['queries'][q] = {'search': s, 'total': d.get('total'),
                             'top20': rows}
        for r in rows:
            if r['screen_relevant'] and r['paperId'] not in seen:
                seen.add(r['paperId'])
                out['relevant_hits'].append(r)
        print(q, 'total', d.get('total'), 'relevant_top20',
              sum(r['screen_relevant'] for r in rows))
        time.sleep(2)

    with open(os.path.join(RES, 'systematic_search_v5.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nRELEVANT hits kept:', len(out['relevant_hits']))
    for r in out['relevant_hits']:
        print('-', r.get('year'), (r.get('title') or '')[:95])
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
