# Clasificación de cáncer de mama (Wisconsin Diagnostic)

![Resumen del proyecto: recall maligno 97,6 %, ROC-AUC 0,998, matriz de confusión y curva ROC en test](reports/figures/infografia.png)

Modelo de clasificación binaria para distinguir tumores **malignos** de **benignos** a partir de características nucleares extraídas de una aspiración con aguja fina (FNA).

Un falso negativo (maligno etiquetado como benigno) es más costoso que un falso positivo. La métrica principal es el **recall de la clase maligna**.

Dataset: [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) — 569 muestras, 357 benignas y 212 malignas, 30 features numéricas.

## Resultado

Test estratificado (114 casos). El escalado y los hiperparámetros se ajustan **solo en train**.

| Modelo | Accuracy | Recall (M) | F1 (M) | ROC-AUC |
|---|---|---|---|---|
| Dummy (mayoritaria) | 0.632 | 0.000 | 0.000 | 0.500 |
| **Regresión logística** | **0.991** | **0.976** | **0.988** | **0.998** |
| Random forest | 0.965 | 0.929 | 0.951 | 0.996 |

La regresión logística regularizada es el modelo elegido: más simple y con mejor recall (1 falso negativo, 0 falsos positivos).

<p align="center">
  <img src="reports/figures/matriz_confusion_lr.png" width="45%" alt="Matriz de confusión de la regresión logística en test: 72 benignos y 41 malignos correctos, 1 falso negativo, 0 falsos positivos">
</p>

![Curva ROC en test de regresión logística, random forest y dummy, con zoom en la esquina superior izquierda](reports/figures/roc_comparada.png)

## Enfoque

1. Limpieza: se elimina `id` y la columna vacía `Unnamed: 32`. `M → 1`, `B → 0`.
2. Split 80/20 **estratificado** *antes* de cualquier transformación.
3. `Pipeline` de scikit-learn: `StandardScaler` + modelo (el forest no escala).
4. `GridSearchCV` + `StratifiedKFold(5)` sobre train, scoring = recall de maligno. `refit=True` reentrena el mejor modelo en todo el train.
5. El test se evalúa una sola vez.

Las variables de tamaño (`radius`, `perimeter`, `area`) están casi duplicadas. Sirven para predecir; los coeficientes de la logística no se leen como efectos independientes. La señal se concentra en `concave points`, `perimeter` y `radius` (sobre todo *worst*): núcleos más grandes e irregulares se asocian a malignidad.

![Las 10 variables con mayor coeficiente en la regresión logística; todas empujan hacia maligno](reports/figures/top_features_lr.png)

## Cómo ejecutarlo

Requiere [uv](https://docs.astral.sh/uv/) y Python ≥ 3.12.

```bash
uv sync
uv run jupyter notebook notebooks/clasificacion_cancer_mama.ipynb
```

## Estructura

```
data/data.csv
notebooks/clasificacion_cancer_mama.ipynb   # análisis + sección 10 que genera las figuras
reports/figures/                            # PNG del README e infografía para LinkedIn
pyproject.toml
```

## Limitaciones

569 muestras de un solo centro, con features ya diseñadas (no hay imagen cruda). No sustituye un diagnóstico clínico. Un siguiente paso razonable es calibrar el umbral según el coste de un falso negativo.
