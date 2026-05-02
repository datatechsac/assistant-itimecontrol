# 🤖 Asistente Conversacional Inteligente para iTimeControl

> Tesis: *"Asistente conversacional inteligente basado en fine-tuning para mejorar consultas operativas en el Sistema iTimeControl"*

## 📋 Autor

Jose Luis Espinoza Garcia

---

## 📊 Estado del Proyecto — Entregables

| Entregable | Estado | Archivo |
|---|---|---|
| README estructurado | ✅ Completo | `README.md` |
| Notebook EDA | ✅ Completo | `notebooks/02_EDA.ipynb` |
| Baseline mínimo ejecutado | ✅ Completo | `notebooks/03_baseline.ipynb` |
| Resultados y gráfica central | ✅ Completo | `notebooks/03_baseline.ipynb` |
| Pipeline corriendo | ✅ Completo | `run_pipeline.py` |

---

## 📋 Descripción

Este proyecto implementa un asistente conversacional inteligente que combina **Fine-Tuning** y **RAG (Retrieval-Augmented Generation)** para responder consultas operativas sobre el sistema iTimeControl. Se entrenó un modelo de lenguaje sobre documentación específica del sistema para mejorar la precisión y relevancia de las respuestas.

---

## 🏗️ Arquitectura del Proyecto

```
itimecontrol-assistant/
│
├── data/
│   ├── raw/              # PDFs originales del sistema iTimeControl
│   ├── processed/        # Texto extraído y normalizado
│   ├── chunks/           # Fragmentos JSON para RAG
│   ├── embeddings/       # Vectores generados
│   ├── datasets/         # Datasets JSONL para fine-tuning
│   └── vector_store/     # Base de datos vectorial (FAISS/ChromaDB)
│
├── src/
│   ├── preprocessing/    # Extracción y preparación de datos
│   ├── fine_tuning/      # Entrenamiento del modelo (LoRA/QLoRA)
│   ├── rag/              # Pipeline de RAG
│   ├── evaluation/       # Métricas y benchmarks
│   ├── api/              # FastAPI + interfaz Gradio
│   └── utils/            # Herramientas comunes
│
├── notebooks/            # Experimentos en Jupyter
├── models/               # Checkpoints guardados
├── logs/                 # Logs de entrenamiento
├── config.yaml           # Configuración global
├── requirements.txt      # Dependencias
└── .env.example          # Variables de entorno de ejemplo
```

---

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/itimecontrol-assistant.git
cd itimecontrol-assistant
```

### 2. Crear entorno virtual

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tus credenciales y rutas
```

---

## 📄 Flujo de Trabajo

### Paso 1 — Colocar los PDFs

Copia los manuales y documentación de iTimeControl en:
```
data/raw/
```

### Paso 2 — Preprocesamiento

```bash
# Extraer texto de los PDFs
python src/preprocessing/pdf_extractor.py

# Limpiar y normalizar el texto
python src/preprocessing/text_cleaner.py

# Dividir en chunks para RAG
python src/preprocessing/chunker.py

# Generar pares Pregunta-Respuesta para fine-tuning
python src/preprocessing/dataset_builder.py
```

### Paso 3 — Fine-Tuning

```bash
# Formatear el dataset al formato instrucción
python src/fine_tuning/data_formatter.py

# Iniciar entrenamiento con LoRA
python src/fine_tuning/trainer.py
```

### Paso 4 — Construir la base vectorial (RAG)

```bash
# Generar embeddings e indexar en FAISS
python src/rag/embedder.py
```

### Paso 5 — Evaluar el modelo

```bash
python src/evaluation/benchmark.py
```

### Paso 6 — Iniciar la API / Interfaz

```bash
# FastAPI
uvicorn src.api.chat_endpoint:app --reload

# Interfaz Gradio
python src/api/app.py
```

---

## ⚙️ Configuración

Edita `config.yaml` para ajustar los parámetros del modelo, chunking, y RAG:

```yaml
model:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  output_dir: "models/checkpoints"

training:
  num_epochs: 3
  learning_rate: 2e-4
  batch_size: 4

chunking:
  chunk_size: 512
  chunk_overlap: 64

rag:
  top_k: 5
  embedding_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

---

## 📊 Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Extracción PDF | pdfplumber, PyMuPDF |
| Fine-Tuning | HuggingFace Transformers, PEFT (LoRA), TRL |
| Cuantización | bitsandbytes (4-bit QLoRA) |
| Embeddings | sentence-transformers |
| Vector Store | FAISS / ChromaDB |
| API | FastAPI |
| Interfaz | Gradio |
| Evaluación | ROUGE, BLEU, NLTK |

---

## 📓 Notebooks

| Notebook | Descripción |
|---|---|
| `01_explore_data.ipynb` | Exploración de los PDFs y texto extraído |
| `02_finetune_experiment.ipynb` | Experimentos de fine-tuning paso a paso |
| `03_rag_experiment.ipynb` | Construcción y prueba del pipeline RAG |

---

## 🔬 Evaluación

El modelo es evaluado con las métricas:
- **ROUGE-1, ROUGE-2, ROUGE-L** — calidad de la generación de texto
- **BLEU** — precisión de los n-gramas generados
- **Exact Match** — respuestas exactas en consultas factuales

---

## 📁 Dataset

El dataset de entrenamiento se construye automáticamente desde los PDFs de iTimeControl usando `dataset_builder.py`. El formato es:

```json
{
  "instruction": "¿Cómo registro la asistencia de un empleado en iTimeControl?",
  "input": "",
  "output": "Para registrar la asistencia en iTimeControl, debe ingresar al módulo de..."
}
```

---

## 👤 Autor

**[Tu Nombre]**
- Universidad: [Nombre de tu Universidad]
- Carrera: [Ingeniería de Sistemas / Informática]
- Año: 2024-2025

---

## 📄 Licencia

Este proyecto es de uso académico. Ver `LICENSE` para más detalles.

---

## 🎓 Entregables Académicos

### 📓 Notebooks disponibles

| Notebook | Propósito |
|---|---|
| `notebooks/02_EDA.ipynb` | EDA completo: estadísticas, distribuciones, riesgos |
| `notebooks/03_baseline.ipynb` | Baseline: TF-IDF, Naive Bayes, KNN + métricas |
| `notebooks/01_explore_data.ipynb` | Exploración rápida del corpus |
| `notebooks/02_finetune_experiment.ipynb` | Experimento de entrenamiento |
| `notebooks/03_rag_experiment.ipynb` | Experimento RAG |

### ⚠️ Riesgos identificados en el EDA

| Riesgo | Descripción | Mitigación |
|---|---|---|
| **Data Leakage** | Las preguntas QA se generan del mismo texto que se indexa | Split estricto train/val/test; benchmark con preguntas externas |
| **Desbalance** | Un PDF puede dominar el corpus si es mucho más extenso | Sub-muestreo o ponderación de chunks por fuente |
| **Concept Drift** | El software iTimeControl puede actualizarse | Versionar documentos junto con el modelo; re-indexar en cada versión |

### 🏁 Baseline establecido

Modelos de referencia evaluados con ROUGE, BLEU y Precision@K:
- **TF-IDF + Coseno** — mejor baseline clásico
- **Naive Bayes** — clasificación de intención
- **KNN** — similitud por vecinos más cercanos

El sistema propuesto (RAG + Fine-Tuning) debe superar estos resultados.

### 🖥️ Guía para Demo Interna (5–10 min)

```bash
# 1. Mostrar estructura del proyecto (1 min)
ls src/

# 2. Ejecutar preprocesamiento en vivo (2 min)
python src/preprocessing/pdf_extractor.py
python src/preprocessing/chunker.py

# 3. Mostrar logs generados (1 min)
cat logs/app.log

# 4. Abrir notebooks en Jupyter (3 min)
jupyter notebook notebooks/02_EDA.ipynb        # mostrar gráficas
jupyter notebook notebooks/03_baseline.ipynb   # mostrar resultados baseline

# 5. Demo del pipeline completo (2 min)
python run_pipeline.py --stage preprocess
```
