import csv
import os
import time

import requests

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

RAW = "pubchem_vacv_raw.csv"
DOCS = "pubchem_docs.csv"

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

TERMS = ["vaccinia", "vaccinia cytotoxicity", "orthopoxvirus"]
WANTED = {"IC50", "EC50", "CC50", "TC50", "TD50", "CD50", "ID50", "ED50"}


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


def get(url, params = None):
    for att in range(4):
        try:
            r = requests.get(url, params = params, timeout = 90)
        except Exception as e:
            print("   сеть:", e)
            time.sleep(3 * (att + 1))
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            return None
        if r.status_code >= 500 or r.status_code == 429:
            time.sleep(3 * (att + 1))
            continue
        print("   HTTP", r.status_code, url)
        return None
    return None


def find_aids():
    aids = set()
    for term in TERMS:
        r = get(EUTILS + "/esearch.fcgi", {
            "db": "pcassay", "term": term, "retmax": 2000, "retmode": "json"})
        if r is None:
            print(" ", term, "-> ошибка")
            continue
        got = r.json()["esearchresult"].get("idlist", [])
        print(" ", term, "->", len(got), "ассеев")
        aids |= set(got)
        time.sleep(0.4)
    return sorted(aids, key = int)


def concise(aid):
    r = get(PUG + "/assay/aid/" + aid + "/concise/JSON")
    if r is None:
        return []
    try:
        t = r.json()["Table"]
    except Exception:
        return []
    cols = t["Columns"]["Column"]
    out = []
    for row in t.get("Row", []):
        out.append(dict(zip(cols, row["Cell"])))
    return out


def smiles_for(cids):
    out = {}
    step = 100
    for i in range(0, len(cids), step):
        part = cids[i:i + step]
        r = get(PUG + "/compound/cid/" + ",".join(part) + "/property/CanonicalSMILES/JSON")
        if r is None:
            r = get(PUG + "/compound/cid/" + ",".join(part) + "/property/SMILES/JSON")
        if r is None:
            continue
        try:
            props = r.json()["PropertyTable"]["Properties"]
        except Exception:
            continue
        for p in props:
            s = p.get("CanonicalSMILES") or p.get("SMILES") or p.get("ConnectivitySMILES")
            if s:
                out[str(p["CID"])] = s
        print("   smiles", min(i + step, len(cids)), "/", len(cids))
        time.sleep(0.25)
    return out


def dois_for(pmids):
    out = []
    step = 150
    for i in range(0, len(pmids), step):
        part = pmids[i:i + step]
        r = get(EUTILS + "/esummary.fcgi", {
            "db": "pubmed", "id": ",".join(part), "retmode": "json"})
        if r is None:
            continue
        res = r.json().get("result", {})
        for p in part:
            rec = res.get(p)
            if not rec:
                continue
            doi = None
            for a in rec.get("articleids", []):
                if a.get("idtype") == "doi":
                    doi = a.get("value")
            out.append({
                "document_chembl_id": "PMID" + p,
                "doi": doi,
                "pubmed_id": p,
                "year": (rec.get("pubdate") or "")[:4],
                "title": rec.get("title"),
            })
        print("   doi", min(i + step, len(pmids)), "/", len(pmids))
        time.sleep(0.4)
    return out


def pick(r, aid):
    cid = str(r.get("CID") or "")
    name = (r.get("Activity Name") or "").strip()
    val = r.get("Activity Value [uM]")
    pmid = r.get("PubMed ID")
    return {
        "activity_id": "PC-" + aid + "-" + cid + "-" + name,
        "molecule_chembl_id": "CID" + cid,
        "canonical_smiles": None,
        "assay_chembl_id": "AID" + aid,
        "assay_description": r.get("Assay Name"),
        "assay_type": r.get("Bioassay Type"),
        "target_chembl_id": None,
        "target_organism": None,
        "target_pref_name": None,
        "standard_type": name,
        "standard_relation": "=",
        "standard_value": val,
        "standard_units": "uM",
        "document_chembl_id": ("PMID" + str(pmid)) if pmid else None,
        "activity_comment": r.get("Activity Outcome"),
    }


def main():
    sink = Sink(RAW, FIELDS)
    print("уже в файле:", len(sink.seen), "строк\n")

    print("=== поиск ассеев ===")
    if os.path.exists("_ids_aid.txt"):
        aids = sorted(done_load("_ids_aid.txt"), key = int)
        print(" из кэша:", len(aids))
    else:
        aids = find_aids()
        with open("_ids_aid.txt", "w", encoding = "utf-8") as f:
            for a in aids:
                f.write(a + "\n")
    print("всего ассеев:", len(aids))

    print("\n=== выгрузка измерений ===")
    done = done_load("_done_aid.txt")
    todo = [a for a in aids if a not in done]
    print("осталось:", len(todo))
    rows = []
    for i, aid in enumerate(todo, 1):
        for r in concise(aid):
            name = (r.get("Activity Name") or "").strip()
            if name not in WANTED:
                continue
            if r.get("Activity Value [uM]") in (None, ""):
                continue
            if not r.get("CID"):
                continue
            rows.append(pick(r, aid))
        done_mark("_done_aid.txt", aid)
        if i % 20 == 0 or i == len(todo):
            print(" ", i, "/", len(todo), "| пригодных строк", len(rows))
        time.sleep(0.25)

    print("\n=== SMILES по CID ===")
    cids = sorted({r["molecule_chembl_id"][3:] for r in rows})
    print("молекул:", len(cids))
    smi = smiles_for(cids)
    kept = 0
    for r in rows:
        s = smi.get(r["molecule_chembl_id"][3:])
        if not s:
            continue
        r["canonical_smiles"] = s
        kept += sink.add(r)
    sink.flush()
    print("записано строк:", kept, "| всего в файле:", len(sink.seen))

    print("\n=== DOI ===")
    pmids = sorted({r["document_chembl_id"][4:] for r in rows if r["document_chembl_id"]})
    print("статей:", len(pmids))
    meta = Sink(DOCS, DOC_FIELDS)
    for m in dois_for(pmids):
        meta.add(m)
    meta.flush()
    print("готово:", len(sink.seen), "строк,", len(meta.seen), "статей")


if __name__ == "__main__":
    main()
