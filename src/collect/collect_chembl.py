import csv
import os
import time

import requests

BASE = "https://www.ebi.ac.uk/chembl/api/data"
RAW = "chembl_vacv_raw.csv"
DOCS = "chembl_docs.csv"

FIELDS = [
    "activity_id",
    "molecule_chembl_id",
    "canonical_smiles",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "target_chembl_id",
    "target_organism",
    "target_pref_name",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "document_chembl_id",
    "activity_comment",
]

DOC_FIELDS = ["document_chembl_id", "doi", "pubmed_id", "year", "title"]
WORDS = ["vaccinia", "vacv", "orthopox"]


class Sink:
    def __init__(self, path, cols):
        self.cols = cols
        self.seen = set()
        new = not os.path.exists(path)
        if not new:
            with open(path, encoding = "utf-8") as f:
                for r in csv.DictReader(f):
                    self.seen.add(str(r[cols[0]]))
        self.f = open(path, "a", newline = "", encoding = "utf-8")
        self.w = csv.DictWriter(self.f, fieldnames = cols)
        if new:
            self.w.writeheader()

    def add(self, row):
        k = str(row[self.cols[0]])
        if k in self.seen:
            return 0
        self.seen.add(k)
        self.w.writerow(row)
        return 1

    def flush(self):
        self.f.flush()


def done_load(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding = "utf-8") as f:
        return {l.strip() for l in f if l.strip()}


def done_mark(path, x):
    with open(path, "a", encoding = "utf-8") as f:
        f.write(x + "\n")


def api(res, key, params, limit = 200):
    rows = []
    offset = 0
    while True:
        p = dict(params)
        p["format"] = "json"
        p["limit"] = limit
        p["offset"] = offset
        d = None
        for att in range(4):
            try:
                r = requests.get(BASE + "/" + res, params = p, timeout = 90)
            except Exception as e:
                print("   сеть:", e)
                time.sleep(3 * (att + 1))
                continue
            if r.status_code == 200:
                d = r.json()
                break
            if r.status_code >= 500 or r.status_code == 429:
                time.sleep(3 * (att + 1))
                continue
            print("   HTTP", r.status_code, res, params)
            return rows, False
        if d is None:
            print("   не отвечает после 4 попыток:", res, params)
            return rows, False
        chunk = d.get(key, [])
        rows += chunk
        total = d["page_meta"]["total_count"]
        offset += len(chunk)
        if not chunk or offset >= total:
            break
        time.sleep(0.2)
    return rows, True


def hit(*parts):
    text = " ".join(str(p or "") for p in parts).lower()
    return any(w in text for w in WORDS)


def pick(a):
    return {f: a.get(f) for f in FIELDS}


def ids_cached(path, builder):
    if os.path.exists(path):
        got = sorted(done_load(path))
        print(" из кэша:", path, len(got))
        return got
    got = builder()
    with open(path, "w", encoding = "utf-8") as f:
        for x in sorted(got):
            f.write(x + "\n")
    return sorted(got)


def find_targets():
    found = {}
    for res, params in [("target/search", {"q": "vaccinia"}),
                        ("target/search", {"q": "orthopoxvirus"}),
                        ("target", {"organism__icontains": "vaccinia"})]:
        got, ok = api(res, "targets", params)
        print(" ", res, params, "->", len(got), "ok" if ok else "FAIL")
        for t in got:
            if hit(t.get("organism"), t.get("pref_name")):
                found[t["target_chembl_id"]] = t
    for k, v in list(found.items())[:10]:
        print("   ", k, "|", v.get("organism"), "|", v.get("pref_name"))
    return set(found)


def find_assays():
    found = {}
    for res, params in [("assay", {"assay_organism__icontains": "vaccinia"}),
                        ("assay/search", {"q": "vaccinia"}),
                        ("assay/search", {"q": "VACV"})]:
        got, ok = api(res, "assays", params)
        print(" ", res, params, "->", len(got), "ok" if ok else "FAIL")
        for a in got:
            if hit(a.get("assay_organism"), a.get("description")):
                found[a["assay_chembl_id"]] = a
    return set(found)


def crawl(sink, ids, field, done_path, label, extra = None):
    done = done_load(done_path)
    todo = [x for x in ids if x not in done]
    print(label, ":", len(todo), "осталось из", len(ids))
    bad = 0
    for i, x in enumerate(todo, 1):
        params = {field: x}
        if extra:
            params.update(extra)
        got, ok = api("activity", "activities", params)
        for a in got:
            sink.add(pick(a))
        if ok:
            done_mark(done_path, x)
        else:
            bad += 1
        if i % 25 == 0 or i == len(todo):
            sink.flush()
            print(" ", i, "/", len(todo), "| всего строк", len(sink.seen), "| пропущено", bad)
    sink.flush()


def docs_from_raw():
    out = set()
    if not os.path.exists(RAW):
        return []
    with open(RAW, encoding = "utf-8") as f:
        for r in csv.DictReader(f):
            d = r.get("document_chembl_id")
            if d:
                out.add(d)
    return sorted(out)


def main():
    sink = Sink(RAW, FIELDS)
    print("уже в файле:", len(sink.seen), "активностей\n")

    print("=== мишени ===")
    targets = ids_cached("_ids_target.txt", find_targets)
    print("мишеней:", len(targets))

    print("\n=== ассеи ===")
    assays = ids_cached("_ids_assay.txt", find_assays)
    print("ассеев:", len(assays))

    print("\n=== активности по ассеям ===")
    crawl(sink, assays, "assay_chembl_id", "_done_assay.txt", "ассеи")

    print("\n=== активности по мишеням ===")
    crawl(sink, targets, "target_chembl_id", "_done_target.txt", "мишени")

    docs = docs_from_raw()
    print("\nстатей:", len(docs))

    print("\n=== добираем цитотоксичность из тех же статей ===")
    wanted = ["CC50", "TC50", "TD50", "CD50", "Cytotoxicity",
              "Ratio CC50/IC50", "Ratio CC50/EC50", "Selectivity Index"]
    crawl(sink, docs, "document_chembl_id", "_done_doc.txt", "статьи",
          {"standard_type__in": ",".join(wanted)})

    print("\n=== DOI ===")
    meta = Sink(DOCS, DOC_FIELDS)
    done = done_load("_done_meta.txt")
    todo = [d for d in docs_from_raw() if d not in done]
    for i, d in enumerate(todo, 1):
        got, ok = api("document", "documents", {"document_chembl_id": d})
        for x in got:
            meta.add({k: x.get(k) for k in DOC_FIELDS})
        if ok:
            done_mark("_done_meta.txt", d)
        if i % 50 == 0 or i == len(todo):
            meta.flush()
            print(" ", i, "/", len(todo))
    meta.flush()

    print("\nготово:", len(sink.seen), "активностей,", len(meta.seen), "статей с метаданными")


if __name__ == "__main__":
    main()
