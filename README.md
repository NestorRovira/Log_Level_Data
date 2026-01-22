# DeepLV – Experimentación en Ingeniería de Software con Deep Learning

Este repositorio contiene la implementación experimental de un enfoque basado en Deep Learning para la **predicción automática del nivel de log** (`TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`) en proyectos software. Está inspirado en el paper **DeepLV (ICSE 2021)** y adaptado como parte de la práctica de la asignatura _Experimentación en Ingeniería de Software_.

El objetivo principal es **comparar dos formas de codificar la salida** de una red neuronal (One-Hot vs Ordinal) y evaluar su impacto en la calidad de la predicción, manteniendo constante la arquitectura y el entorno experimental.

---

##  Estructura del Repositorio

```
.
├── block_processing/
│   └── blocks/
│       ├── logged_syn_cassandra.csv
│       ├── logged_syn_karaf.csv
│       └── logged_syn_wicket.csv
│
├── src/
│   ├── block_processing.py
│   ├── block_level_LSTM.py
│   ├── run_experiment.py
│   ├── analyze_results.py
│   ├── Metrics.py
│   └── Helper.py
│
├── results/
│   └── experiment_cassandra/
│       ├── execution_plan.json
│       ├── results.csv
│       ├── results.jsonl
│       ├── models/
│       └── analysis/
│
└── README.md
```

---

## Componentes Principales

### `block_processing.py`
Encargado del preprocesamiento de los logs, incluyendo:
- Normalización del texto
- Agrupación por bloques
- Preparación de los CSV de entrada para el modelo

### `block_level_LSTM.py`
Implementa el pipeline completo de entrenamiento y evaluación:
- Tokenización y Word2Vec (Skip-gram)
- Construcción de la arquitectura Bi-LSTM
- Soporte para dos modelos:
  - One-Hot (baseline)
  - Ordinal (propuesto)
- Cálculo de métricas: **Accuracy**, **AUC**, **AOD**

### `run_experiment.py`
Script de orquestación experimental:
- Ejecuta 30 repeticiones por tratamiento
- Genera semillas aleatorias reales por par de ejecuciones
- Entrena con **Cassandra**
- Evalúa con **Karaf** y **Wicket**
- Guarda todos los resultados de forma estructurada y reproducible

### `analyze_results.py`
Script de análisis final automático:
- Estadísticos descriptivos
- Tests estadísticos (`t-test`, no paramétricos si aplica)
- Cálculo de tamaño del efecto
- Generación de tablas y gráficas listas para la memoria

### `Metrics.py`
Implementación de métricas específicas del experimento:
- **Accuracy**
- **AUC multiclase**
- **AOD (Average Ordinal Distance)**

---

##  Ejecución del Experimento (end-to-end)

### Lanzar el experimento completo:
```
python -m src.run_experiment --train cassandra --tests karaf,wicket --runs 30
```

### Ejecutar el análisis final:
```
python -m src.analyze_results --results_csv results/experiment_cassandra/results.csv \
                              --out_dir results/experiment_cassandra/analysis
```

---

##  Resultados Generados

| Archivo | Descripción                               |
|--------|-------------------------------------------|
| `results.csv` | Datos listos para análisis estadístico    |
| `results.jsonl` | Trazabilidad completa por ejecución       |
| `analysis/` | Tablas y figuras resultantes del analisis |
| `execution_plan.json` | Plan experimental reproducible            |

