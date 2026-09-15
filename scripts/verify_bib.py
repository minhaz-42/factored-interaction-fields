"""Audit paper/refs.bib: for every entry with an arXiv id (eprint field or url),
fetch the arXiv metadata and compare title and first author; flag mismatches and
entries without any verifiable identifier. Uses the arXiv Atom API.
"""
from __future__ import annotations
import re, sys, time, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

NS = {"a": "http://www.w3.org/2005/Atom"}


def parse_bib(text):
    entries = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, flags=re.S):
        fields = dict((k.strip().lower(), v.strip().strip("{}").strip('"')) for k, v in re.findall(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\")", m.group(3)))
        entries.append((m.group(1), m.group(2).strip(), fields))
    return entries


def arxiv_meta(aid):
    url = f"https://export.arxiv.org/api/query?id_list={aid}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                root = ET.fromstring(r.read())
            e = root.find("a:entry", NS)
            if e is None or e.find("a:title", NS) is None: return None
            return {"title": re.sub(r"\s+", " ", e.find("a:title", NS).text).strip(), "authors": [x.find("a:name", NS).text for x in e.findall("a:author", NS)]}
        except Exception as ex:
            time.sleep(3 * (attempt + 1))
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    bib = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/refs.bib").read_text()
    bad = 0
    for kind, key, f in parse_bib(bib):
        aid = f.get("eprint") or (re.search(r"arxiv\.org/abs/([\d.]+)", f.get("url", "")) or [None, None])[1]
        if not aid:
            print(f"[no-arxiv] {key}: {f.get('title','?')[:70]} | doi={f.get('doi','-')} url={f.get('url','-')}"); continue
        meta = arxiv_meta(aid); time.sleep(1.0)
        if meta is None:
            print(f"[unreachable] {key} ({aid})"); continue
        t_ok = norm(meta["title"]) == norm(f.get("title", ""))
        a_ok = norm(f.get("author", "").split(" and ")[0].split(",")[-1]) in norm(meta["authors"][0]) or norm(meta["authors"][0].split()[-1]) in norm(f.get("author", "").split(" and ")[0])
        flag = "OK" if (t_ok and a_ok) else "MISMATCH"; bad += flag != "OK"
        print(f"[{flag}] {key} ({aid}) title_ok={t_ok} author_ok={a_ok}" + ("" if t_ok else f"\n    bib:   {f.get('title','')}\n    arxiv: {meta['title']}"))
    print("mismatches:", bad)


if __name__ == "__main__":
    main()
