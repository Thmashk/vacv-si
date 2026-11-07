"""Единая точка конфигурации. Все пути и константы берутся отсюда."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS = ROOT / "submissions"
REPORTS = ROOT / "reports"

TRAIN_CSV = RAW / "train_v2.csv"
TEST_CSV = RAW / "test.csv"
SAMPLE_CSV = RAW / "sample_submit.csv"

# Колонки, которые правилами запрещено использовать: из них SI считается делением.
FORBIDDEN = frozenset({"CC50_uM", "IC50_uM_VACV"})

TARGET = "SI VACV"
SMILES_COL = "SMILES"
ID_COL = "index"

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

# Границы бинов целевой переменной для двухчастной модели.
# Нулевой бин выделен отдельно: SI = 0 стоит там, где IC50 не измерен (26% трейна).
BIN_EDGES = [0.0, 1e-9, 5.0, 50.0, 500.0, float("inf")]
BIN_NAMES = ["ноль", "0–5", "5–50", "50–500", ">500"]

MORGAN_BITS = 1024
MORGAN_RADIUS = 2

# Множитель для среднего в верхнем бине. 1.0 — честная условная средняя.
# >1 делает прогноз агрессивнее на хвосте, <1 консервативнее.
TAIL_SCALE = 1.0
