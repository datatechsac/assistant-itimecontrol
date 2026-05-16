# 🧪 Branch: semana5 — Experimentos A/B y Validación

> **Tesis:** Asistente conversacional inteligente basado en fine-tuning para iTimeControl  
> **Semana:** 5 | **Fecha:** Mayo 2025

---

## ✅ Entregables de esta semana

| Entregable | Estado | Archivo |
|---|---|---|
| Experimentos A/B ejecutados | ✅ | `notebooks/05_experimentos_ab.ipynb` |
| Tabla estándar Baseline/Var1/Var2 | ✅ | `logs/tabla_resultados_semana5.csv` |
| Gráfico Recall@k vs k | ✅ | `logs/recall_at_k_semana5.png` |
| Feature set y pipeline documentado | ✅ | Sección 3 del notebook |
| Confirmación cero leakage | ✅ | Sección 2 del notebook |
| Cross Validation + Holdout | ✅ | Sección 7 del notebook |
| Log de experimentos | ✅ | `logs/experimentos_semana5.json` |

---

## 🔬 Diseño de Experimentos A/B

Se comparan **3 variantes**, cambiando **un factor por vez**:

### Baseline — TF-IDF Unigrams
- `ngram_range=(1,1)`
- `max_features=5000`
- Sin normalización de acentos
- Sin ajuste de TF

### Variante 1 — TF-IDF Bigrams + Normalización
**Features añadidas respecto al Baseline:**
- `ngram_range=(1,2)` → captura frases de dos palabras ("horas extras", "marcación manual")
- `strip_accents='unicode'` → normaliza "asistencia" = "asistencia"
- `sublinear_tf=True` → aplica log(TF), reduce dominancia de palabras muy frecuentes
- `max_features=8000` → mayor vocabulario

**Features removidas:** ninguna

### Variante 2 — TF-IDF Bigrams + Reranking
**Features añadidas respecto a Var1:**
- Reranking por longitud de documento (penaliza respuestas muy cortas)
- Filtro de similitud mínima ≥ 0.05 (elimina recuperaciones irrelevantes)
- `max_features=10000` → vocabulario aún mayor

**Features removidas:** ninguna

---

## 📊 Resultados

### Tabla estándar — Baseline / Var1 / Var2

| Métrica | Baseline | Var1 | Var2 | Mejor |
|---|---|---|---|---|
| ROUGE-1 | ver CSV | ver CSV | ver CSV | — |
| ROUGE-2 | ver CSV | ver CSV | ver CSV | — |
| ROUGE-L | ver CSV | ver CSV | ver CSV | — |
| BLEU | ver CSV | ver CSV | ver CSV | — |
| Recall@1 | ver CSV | ver CSV | ver CSV | — |
| Recall@3 | ver CSV | ver CSV | ver CSV | — |
| Recall@5 | ver CSV | ver CSV | ver CSV | — |
| Latencia (ms) | ver CSV | ver CSV | ver CSV | — |

> Valores exactos en `logs/tabla_resultados_semana5.csv`

### Gráfico clave — Recall@k vs k

![Recall@k](logs/recall_at_k_semana5.png)

---

## 🔒 Confirmación de Cero Leakage

```
✅ TF-IDF fit SOLO sobre train  
✅ Evaluación sobre test (nunca visto por el modelo)  
✅ Split: holdout aleatorio 80/10/10  
✅ Verificación automática de overlap train-test: 0 coincidencias  
```

El pipeline garantiza que **ninguna transformación ni parámetro aprendido del modelo** usa información del conjunto de test. El `TfidfVectorizer.fit()` se llama exclusivamente sobre `corpus_train`.

---

## ✔️ Validación

### Holdout
- Train: 80% | Val: 10% | Test: 10%
- Evaluación final reportada sobre test (datos no vistos)

### Cross Validation (5-fold estratificado)
- Modelo: Naive Bayes para clasificación de intención
- Estratificado por categoría de intención (`registro_asistencia`, `reportes`, `horarios`, etc.)
- Métricas: Accuracy, Precision, Recall, F1 (weighted)

```
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 📁 Estructura de archivos de esta semana

```
notebooks/
└── 05_experimentos_ab.ipynb    ← notebook principal de la semana

logs/
├── experimentos_semana5.json   ← log completo con parámetros y métricas
├── tabla_resultados_semana5.csv
├── recall_at_k_semana5.png     ← gráfico clave
└── validacion_cv_holdout_semana5.png
```

---

## 🚀 Cómo reproducir

```bash
# 1. Cambiar a este branch
git checkout semana5

# 2. Instalar dependencias (si no están)
pip install scikit-learn rouge-score matplotlib seaborn nltk

# 3. Ejecutar el notebook
jupyter notebook notebooks/05_experimentos_ab.ipynb
```

---

## 📌 Próximos pasos (Semana 6)

- Integrar el modelo fine-tuned (Mistral-7B + LoRA) como Variante 3
- Comparar RAG puro vs Fine-Tuning puro vs RAG + Fine-Tuning
- Evaluación con preguntas operativas reales de usuarios de iTimeControl
