# 🏠 Module 2 – House Prices: предобработка и регрессия

**Автор:** Иванов Семён Иванович, ПКТб-23-1  
**Дата:** 13-10-2026

## 📊 Результаты

| Модель | RMSE (log) | MAE (log) | R² | RMSE ($) | MAE ($) |
|---|---|---|---|---|---|
| Linear Regression | 0.1256 | 0.0897 | 0.8936 | $22 787 | $15 533 |
| Ridge (α=1.0) | 0.1251 | 0.0894 | 0.8945 | $22 717 | $15 488 |
| Lasso (α=0.0005) | 0.1222 | 0.0852 | 0.8993 | $22 008 | $14 727 |
| ElasticNet (α=0.001) | 0.1223 | 0.0852 | 0.8991 | $21 982 | $14 718 |
| **ElasticNet (tuned)** | **0.1186** | **0.0787** | **0.9051** | **$20 252** | **$13 421** |

**Лучшая модель:** ElasticNet (alpha=0.005, l1_ratio=0.7)  
**Время обучения:** ~0.06 сек

## 🚀 Быстрый старт

```python
import requests, joblib, json
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/IvanovSimoon/ml-course-ivanov/refs/heads/main/module-2-house-prices"

model    = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
scaler   = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
features = json.loads(requests.get(f"{BASE_URL}/models/feature_cols.json").content)