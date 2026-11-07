# Смесь «модель + соседи» (R4)

Протокол A — внешние как в тесте, B — без молекул валидации. Решать по RMSE log и Spearman.

| протокол | параметры | RMSE log | Spearman | MSE | MSE без гиганта |
|---|---|---|---|---|---|
| A | {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.667 | 0.608 | 4,113,458 | 66,776 |
| A | {'p': 20, 'lam': 0.3, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.668 | 0.606 | 4,113,574 | 66,892 |
| A | {'p': 20, 'lam': 1.0, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.670 | 0.603 | 4,116,752 | 68,476 |
| A | {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': None} | 0.672 | 0.613 | 4,113,349 | 66,666 |
| A | {'p': 20, 'lam': 1.0, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.672 | 0.602 | 4,116,794 | 68,518 |
| A | {'p': 20, 'lam': 0.3, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': None} | 0.679 | 0.612 | 4,123,803 | 77,153 |
| A | {'p': 20, 'lam': 1.0, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': None} | 0.686 | 0.606 | 4,142,485 | 94,290 |
| A | {'p': 20, 'lam': 1.0, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': None} | 0.696 | 0.597 | 4,172,934 | 124,834 |
| B | {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': None} | 0.748 | 0.554 | 4,254,236 | 207,995 |
| B | {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.748 | 0.554 | 4,254,236 | 207,995 |
| B | {'p': 20, 'lam': 0.3, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': None} | 0.751 | 0.552 | 4,259,193 | 212,968 |
| B | {'p': 20, 'lam': 0.3, 'w_ext': 0.5, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.751 | 0.552 | 4,259,193 | 212,968 |
| B | {'p': 20, 'lam': 1.0, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': None} | 0.754 | 0.548 | 4,264,467 | 216,654 |
| B | {'p': 20, 'lam': 1.0, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': 0.99} | 0.754 | 0.548 | 4,264,467 | 216,654 |

База: RMSE log 0.777, Spearman 0.515, MSE 4,269,358, без гиганта 220,865
Лучший A: {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': 0.99} -> RMSE log 0.667, Spearman 0.608
Лучший B: {'p': 20, 'lam': 0.3, 'w_ext': 1.0, 'culture_factor': 0.3, 'exact_thr': None} -> RMSE log 0.748, Spearman 0.554