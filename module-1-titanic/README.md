# Module 1 – Titanic: EDA + бинарная классификация

**Автор:** Иванов С. [Отчество], ПКТ6-23-1  
**Дата:** 2026-XX-XX

## Описание

Полный цикл ML-проекта на датасете «Титаник»: EDA, обработка пропусков, feature engineering, обучение двух моделей (Logistic Regression, Decision Tree), оценка качества, интерпретация и функция предсказания для нового пассажира.

## Результаты

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|----------|-----------|--------|----|---------|
| Logistic Regression | 0.82 | 0.80 | 0.75 | 0.77 | 0.86 |
| Decision Tree | 0.79 | 0.74 | 0.71 | 0.72 | 0.81 |

**Время обучения:** ~0.03 сек (LR), ~0.01 сек (DT)

## Быстрый старт

```python
import joblib, requests, json
from io import BytesIO

BASE_URL = "https://raw.githubusercontent.com/IvanovSimoon/ml-course-ivanov/main/module-1-titanic"
model = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/lr_model.pkl").content))
scaler = joblib.load(BytesIO(requests.get(f"{BASE_URL}/models/scaler.pkl").content))
feature_cols = json.loads(requests.get(f"{BASE_URL}/models/feature_cols.json").content)