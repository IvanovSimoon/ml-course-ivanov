# 🛒 Module 4 — Online Shoppers: Инженерия признаков

**Автор:** Иванов Семён Иванович, ПКТб-23-1  
**Дата:** 2026-11-02

## 📊 Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8763 | 0.5768 | 0.7565 | 0.6546 | 0.9247 |
| XGBoost | 0.8848 | 0.6065 | 0.7304 | 0.6627 | 0.9247 |
| LightGBM | 0.8751 | 0.5712 | 0.7775 | 0.6585 | 0.9244 |
| CatBoost | 0.8686 | 0.5525 | 0.7984 | 0.6531 | 0.9258 |
| **LightGBM (Optuna)** | **0.8646** | **0.5407** | **0.8351** | **0.6564** | **0.9304** |

**Лучшая модель:** LightGBM (Optuna, 30 trials).  
**ROC-AUC = 0.9304**, Recall = 83.5%.

## 🚀 Быстрый старт

```python
import requests, joblib, json
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/IvanovSimoon/ml-course-ivanov/refs/heads/main/module-4-features"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
feature_cols = json.loads(requests.get(f"{BASE_URL}/models/feature_cols.json").content)