"""Модели.

Метрика соревнования — MSE на исходном SI. Оптимальный прогноз при MSE —
условное математическое ожидание E[SI | молекула]. У SI крайне тяжёлый хвост:
одна молекула в трейне даёт 94% всей суммы квадратов. Поэтому модель, хорошо
предсказывающая «типичную» молекулу, почти не выигрывает у константы, а
модель, узнающая редкие высокоселективные вещества, выигрывает много.

Отсюда основная конструкция — ConditionalMeanModel: классификатор по бинам
целевой переменной, из которого напрямую собирается условное среднее.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from .config import SEED, TAIL_SCALE
from .cv import target_bins


class MeanBaseline:
    """Константа, равная среднему по трейну. Оптимальная константа при MSE."""

    name = "константа-среднее"

    def fit(self, X, y):
        self.value_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


class LogSmearingModel:
    """Регрессия на log10(1 + SI) с поправкой Дуана при обратном переходе.

    Среднее логарифма — это не логарифм среднего. Без поправки обратное
    преобразование систематически занижает прогноз, а под MSE занижение
    на хвосте стоит очень дорого.
    """

    name = "лог-регрессия со smearing"

    def __init__(self, seed: int = SEED, smearing: bool = True):
        self.seed = seed
        self.smearing = smearing

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        t = np.log10(y + 1.0)
        self.model_ = HistGradientBoostingRegressor(
            max_depth=4, max_iter=400, learning_rate=0.06,
            min_samples_leaf=5, l2_regularization=1.0, random_state=self.seed,
        )
        self.model_.fit(X, t)
        resid = t - self.model_.predict(X)
        self.smear_ = float(np.mean(10.0 ** resid)) if self.smearing else 1.0
        return self

    def predict(self, X):
        mu = self.model_.predict(X)
        return np.clip(10.0 ** mu * self.smear_ - 1.0, 0.0, None)


class ConditionalMeanModel:
    """Двухчастная модель: вероятность бина × среднее внутри бина.

    prediction = Σ_k P(бин = k | молекула) · E[SI | бин = k]

    Это прямая оценка условного среднего, то есть именно то, что
    минимизирует MSE. Хвост попадает в прогноз через вероятность верхнего
    бина: даже 10% уверенности в том, что молекула высокоселективна,
    поднимают прогноз на порядки, и под MSE это выгодно.
    """

    name = "условное среднее по бинам"

    def __init__(self, seed: int = SEED, tail_scale: float = TAIL_SCALE,
                 n_estimators: int = 700):
        self.seed = seed
        self.tail_scale = tail_scale
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        bins = target_bins(y)
        self.clf_ = RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=1,
            max_features="sqrt", class_weight=None,
            random_state=self.seed, n_jobs=-1,
        )
        self.clf_.fit(X, bins)
        top = int(bins.max()) if len(bins) else 0
        means = {}
        for k in np.unique(bins):
            m = float(np.mean(y[bins == k]))
            if k == top:
                m *= self.tail_scale
            means[int(k)] = m
        self.bin_means_ = np.array([means[int(c)] for c in self.clf_.classes_])
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        proba = self.clf_.predict_proba(X)
        return np.clip(proba @ self.bin_means_, 0.0, None)


class IsotonicCalibratedModel:
    """Лучший по ранжированию прогноз, переведённый в условное среднее.

    Логика: лог-модель хорошо упорядочивает молекулы (Spearman около 0.5),
    но её абсолютные значения занижены на хвосте. Изотоническая регрессия,
    обученная на честных OOF-предсказаниях, строит монотонное отображение
    «выход модели → среднее настоящего SI». Это ровно та функция, которая
    минимизирует MSE среди монотонных преобразований ранжировщика.
    """

    name = "лог + изотоническая калибровка"

    def __init__(self, seed: int = SEED, inner_splits: int = 5):
        self.seed = seed
        self.inner_splits = inner_splits

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import KFold

        y = np.asarray(y, dtype=float)
        inner = KFold(self.inner_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(y))
        for tr, va in inner.split(X):
            m = LogSmearingModel(seed=self.seed).fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])

        self.iso_ = IsotonicRegression(y_min=0.0, increasing=True,
                                       out_of_bounds="clip")
        self.iso_.fit(oof, y)
        self.base_ = LogSmearingModel(seed=self.seed).fit(X, y)
        return self

    def predict(self, X):
        return np.clip(self.iso_.predict(self.base_.predict(X)), 0.0, None)


class TailAwareForest:
    """Случайный лес прямо на исходном SI.

    Нужен как независимый взгляд: листья усредняют сырые значения, поэтому
    модель сама по себе даёт условное среднее, но сильно сглаживает хвост.
    """

    name = "лес на исходном SI"

    def __init__(self, seed: int = SEED, n_estimators: int = 700):
        self.seed = seed
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=1,
            max_features="sqrt", random_state=self.seed, n_jobs=-1,
        )
        self.model_.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return np.clip(self.model_.predict(X), 0.0, None)


class KernelLogModel:
    """Ядерная гребневая регрессия на log10(1+SI) с MinMax-ядром (Tanimoto по счётчикам).

    Ядро = K_minmax(Morgan) + w * K_линейное(культура/штамм). Сумма ядер остаётся ядром, так что
    культура сдвигает прогноз, а похожесть молекул задаёт форму. Поправка Дуана при обратном переходе.
    """

    name = "ядро MinMax + культура"

    def __init__(self, fp_idx, ctx_idx, alpha: float = 0.5, w_ctx: float = 0.5, seed: int = SEED):
        self.fp_idx, self.ctx_idx = np.asarray(fp_idx), np.asarray(ctx_idx)
        self.alpha, self.w_ctx, self.seed = alpha, w_ctx, seed

    def _k(self, A, B):
        from scipy.spatial.distance import cdist

        fa, fb = A[:, self.fp_idx].astype(np.float64), B[:, self.fp_idx].astype(np.float64)
        l1 = cdist(fa, fb, "cityblock")
        s = fa.sum(1)[:, None] + fb.sum(1)[None, :]
        k = (s - l1) / np.maximum(s + l1, 1e-9)
        ca, cb = A[:, self.ctx_idx].astype(np.float64), B[:, self.ctx_idx].astype(np.float64)
        return k + self.w_ctx * (ca @ cb.T) / max(ca.shape[1], 1)

    def fit(self, X, y):
        t = np.log10(np.asarray(y, dtype=float) + 1.0)
        self.X_, self.mu_ = X, float(t.mean())
        K = self._k(X, X)
        self.a_ = np.linalg.solve(K + self.alpha * np.eye(len(K)), t - self.mu_)
        resid = t - (self.mu_ + K @ self.a_)
        self.smear_ = float(np.mean(10.0 ** resid))
        return self

    def predict(self, X):
        mu = self.mu_ + self._k(X, self.X_) @ self.a_
        return np.clip(10.0 ** mu * self.smear_ - 1.0, 0.0, None)


class Blend:
    """Взвешенная смесь моделей. Веса подбираются по OOF-предсказаниям."""

    def __init__(self, models, weights=None, name: str = "смесь"):
        self.models = list(models)
        self.weights = weights
        self.name = name

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        if self.weights is None:
            self.weights = np.full(len(self.models), 1.0 / len(self.models))
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.clip(preds @ np.asarray(self.weights, dtype=float), 0.0, None)


def fit_blend_weights(oof: dict[str, np.ndarray], y: np.ndarray,
                      step: float = 0.05) -> dict[str, float]:
    """Перебор весов на сетке с шагом step, минимизируется MSE на OOF.

    Полный перебор надёжнее градиентных методов, когда моделей мало,
    а метрика определяется несколькими точками.
    """
    names = list(oof)
    mats = np.column_stack([oof[n] for n in names])
    y = np.asarray(y, dtype=float)
    k = len(names)
    best, best_w = np.inf, np.full(k, 1.0 / k)

    grid = np.arange(0.0, 1.0 + 1e-9, step)

    def rec(i, left, cur):
        nonlocal best, best_w
        if i == k - 1:
            w = np.array(cur + [left])
            err = float(np.mean((y - mats @ w) ** 2))
            if err < best:
                best, best_w = err, w
            return
        for v in grid:
            if v <= left + 1e-9:
                rec(i + 1, left - v, cur + [v])

    rec(0, 1.0, [])
    return dict(zip(names, best_w.tolist()))
