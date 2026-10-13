# 🏠 Module 2 – House Prices: предобработка и регрессия

**Автор:** Иванов Семён Иванович, ПКТ6-23-1  
**Дата:** 13.10.2026

## 📄 Описание

Полный пайплайн регрессии на датасете House Prices: EDA, обработка 19 столбцов с пропусками, обработка выбросов, логарифмирование целевой, Feature Engineering, обучение 4 моделей (Linear, Ridge, Lasso, ElasticNet), кросс-валидация, GridSearchCV, интерпретация и функция предсказания цены нового дома.

## 📊 Результаты

| Модель | RMSE (log) | MAE (log) | R² | RMSE ($) |
|---|---|---|---|---|
| Linear Regression | — | — | — | — |
| Ridge | — | — | — | — |
| Lasso | — | — | — | — |
| ElasticNet | — | — | — | — |

*(Заполним после обучения)*

## 🚀 Быстрый старт

```python
import requests, joblib, json
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/IvanovSimoon/ml-course-ivanov/main/module-2-house-prices"

model    = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/best_model.pkl").content))
scaler   = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
features = json.loads(requests.get(f"{BASE_URL}/models/feature_cols.json").content)