"""Общее для коллекторов: HTTP с повторами, кэш ответов, перевод единиц.

Все сетевые ответы кладутся в data/external/cache. Повторный запуск берёт
их оттуда, поэтому сборка воспроизводима и не зависит от того, изменится ли
содержимое базы завтра.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import ROOT

CACHE = ROOT / "data" / "external" / "cache"
USER_AGENT = "vacv-si-hackathon/1.0 (research use)"

# Колонки промежуточного длинного формата: одна строка на одно измерение.
LONG_COLUMNS = [
    "source", "record_id", "molecule_id", "smiles",
    "standard_type", "relation", "value", "units", "value_uM",
    "endpoint", "organism", "assay_id", "assay_description",
    "document_id", "doi", "pubmed_id",
]


def _cache_path(url: str) -> Path:
    return CACHE / (hashlib.sha256(url.encode()).hexdigest()[:32] + ".json")


def fetch_json(url: str, *, use_cache: bool = True, retries: int = 4,
               pause: float = 0.25) -> dict | None:
    """GET с кэшем и повторами. Возвращает None, если ресурс недоступен."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url)
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.load(r)
            path.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:  # сеть, таймаут, битый JSON
            last = e
        time.sleep(1.5 * (attempt + 1))
    print(f"    не удалось получить {url[:90]}: {last}")
    return None


def build_url(base: str, path: str, params: dict) -> str:
    return f"{base}/{path}?" + urllib.parse.urlencode(params)


def to_micromolar(value, units, mol_weight: float | None = None) -> float | None:
    """Перевод концентрации в мкМ.

    Массовые единицы (мкг/мл) требуют молекулярной массы. Если её нет,
    запись остаётся без значения: лучше потерять строку, чем внести
    ошибку в три порядка.
    """
    if value is None or units is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None

    u = str(units).strip().lower().replace(" ", "")
    molar = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "μm": 1.0,
             "nm": 1e-3, "pm": 1e-6, "10'-6m": 1.0, "10'-9m": 1e-3}
    if u in molar:
        return v * molar[u]

    mass = {"g.l-1": 1e3, "mg.ml-1": 1e3, "ug.ml-1": 1.0, "µg.ml-1": 1.0,
            "ug/ml": 1.0, "mcg.ml-1": 1.0, "ng.ml-1": 1e-3, "mg.l-1": 1.0}
    if u in mass and mol_weight:
        # мкг/мл -> мкМ: (мкг/мл) / (г/моль) * 1000
        return v * mass[u] / mol_weight * 1000.0
    return None


def mol_weight(smiles: str) -> float | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return float(Descriptors.MolWt(m)) if m is not None else None


# Слова, по которым ассей опознаётся как измерение цитотоксичности,
# а не противовирусной активности.
CYTOTOX_WORDS = (
    "cytotox", "cytopath", "toxicity", "toxic concentration", "cell viability",
    "viability of", "50% cytotoxic", "cc50", "tc50", "cd50",
    "growth inhibition of", "antiproliferative",
)

POTENCY_TYPES = {"IC50", "EC50", "ID50"}
CYTOTOX_TYPES = {"CC50", "TC50", "CD50", "GI50"}


def classify_endpoint(standard_type: str | None, description: str | None,
                      is_virus_assay: bool) -> str | None:
    """Чем является измерение: потенцией против вируса или токсичностью."""
    t = (standard_type or "").upper()
    d = (description or "").lower()
    if t in CYTOTOX_TYPES:
        return "cytotoxicity"
    if any(w in d for w in CYTOTOX_WORDS):
        return "cytotoxicity"
    if t in POTENCY_TYPES and is_virus_assay:
        return "potency"
    return None
