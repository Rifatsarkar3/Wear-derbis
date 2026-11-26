"""F8 SCOUTING (DIAG): fixed queries against the public DataCite, Zenodo, figshare and OpenAlex APIs for (a) open
labelled wear-particle classification image sets and (b) open oil-analysis / debris time series with within-unit
failure or onset events. Raw hits (title, DOI, repository, year, licence, type, URL) go to
outputs/diagnostics/f8_search_hits.csv; a hit counts as VERIFIED only after its record / landing page was checked by
hand (docs/diagnostics.md). Anonymous public API calls only; no login, no bot-protection bypass.
Run: .venv/Scripts/python.exe -I scripts/diag_scout.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script dir

import pandas as pd

from diag_common import OUT

UA = {"User-Agent": "research-data-scout/1.0", "Accept": "application/json"}
QUERIES = {
    "labelled_particles": ["wear particle classification", "wear debris image", "ferrography", "ferrograph image",
                           "wear particle recognition", "wear particle images dataset", "wear debris classification",
                           "ferrogram"],
    "oil_time_series": ["oil debris run-to-failure", "oil debris monitoring gearbox", "wear debris sensor bearing",
                        "lubricating oil analysis failure", "oil analysis engine failure", "inductive debris sensor",
                        "oil particle counter gearbox", "oil condition monitoring", "spectrometric oil analysis",
                        "gearbox run-to-failure oil"],
}


def get(url: str, data: dict | None = None) -> dict | list:
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={**UA, **({"Content-Type": "application/json"} if data else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def datacite(q: str) -> list[dict]:
    d = get("https://api.datacite.org/dois?" + urllib.parse.urlencode(
        {"query": q, "resource-type-id": "dataset", "page[size]": 50}))
    return [{"title": (a["titles"] or [{}])[0].get("title", ""), "doi": a["doi"], "repository": a.get("publisher"),
             "year": a.get("publicationYear"), "licence": ";".join(r.get("rightsIdentifier") or r.get("rights", "")
                                                                  for r in a.get("rightsList") or []),
             "type": (a.get("types") or {}).get("resourceTypeGeneral"), "url": a.get("url")}
            for a in (x["attributes"] for x in d["data"])]


def zenodo(q: str) -> list[dict]:
    d = get("https://zenodo.org/api/records?" + urllib.parse.urlencode({"q": q, "size": 25, "type": "dataset"}))
    return [{"title": h["metadata"].get("title", ""), "doi": h.get("doi", ""), "repository": "Zenodo",
             "year": h["metadata"].get("publication_date", "")[:4],
             "licence": (h["metadata"].get("license") or {}).get("id", ""), "type": "Dataset",
             "url": h["links"].get("self_html", "")} for h in d["hits"]["hits"]]


def figshare(q: str) -> list[dict]:
    d = get("https://api.figshare.com/v2/articles/search", {"search_for": q, "item_type": 3, "page_size": 50})
    return [{"title": a.get("title", ""), "doi": a.get("doi", ""), "repository": "figshare",
             "year": (a.get("published_date") or "")[:4], "licence": "", "type": a.get("defined_type_name"),
             "url": a.get("url_public_html", "")} for a in d]


def openalex(q: str) -> list[dict]:
    d = get("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": q, "filter": "type:dataset", "per-page": 50}))
    return [{"title": w.get("display_name") or "", "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
             "repository": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
             "year": w.get("publication_year"), "licence": (w.get("primary_location") or {}).get("license") or "",
             "type": w.get("type"), "url": (w.get("primary_location") or {}).get("landing_page_url") or ""}
            for w in d["results"]]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="F8 repository scouting")
    ap.add_argument("--sources", nargs="*", default=["datacite", "zenodo", "figshare", "openalex"],
                    help="re-query only these sources; their old rows are replaced, other rows kept")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls (OpenAlex rate-limits)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"datacite": datacite, "zenodo": zenodo, "figshare": figshare, "openalex": openalex}
    rows, errors = [], []
    for topic, qs in QUERIES.items():
        for q in qs:
            for name in a.sources:
                try:
                    rows += [{"topic": topic, "query": q, "source": name, **h} for h in fns[name](q)]
                except Exception as e:  # network / API errors are reported, never retried aggressively
                    errors.append({"topic": topic, "query": q, "source": name, "error": repr(e)[:200]})
                time.sleep(a.sleep)  # polite rate
    df, err = pd.DataFrame(rows), pd.DataFrame(errors, columns=["topic", "query", "source", "error"])
    if (OUT / "f8_search_hits.csv").exists() and set(a.sources) != set(fns):
        old, olde = pd.read_csv(OUT / "f8_search_hits.csv"), pd.read_csv(OUT / "f8_search_errors.csv")
        df = pd.concat([old[~old.source.isin(a.sources)], df], ignore_index=True)
        err = pd.concat([olde[~olde.source.isin(a.sources)], err], ignore_index=True)
    df.to_csv(OUT / "f8_search_hits.csv", index=False)
    err.to_csv(OUT / "f8_search_errors.csv", index=False)
    print(f"{len(df)} hits, {df.doi.nunique()} unique DOIs, {len(errors)} errors")


if __name__ == "__main__":
    main()
