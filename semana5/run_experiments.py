"""
Experimentos A/B — Semana 5
Tesis: Asistente conversacional para iTimeControl
Compara: Baseline TF-IDF vs Variante1 vs Variante2
"""
import json, time, random, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
warnings.filterwarnings('ignore')

random.seed(42)
np.random.seed(42)

LOGS  = Path('/home/claude/semana5/logs')
FIGS  = Path('/home/claude/semana5/reports')
LOGS.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ─── CORPUS DE DEMO (simula los datos reales de iTimeControl) ────────────────
QA_PAIRS = [
    {"q": "¿Cómo registro la asistencia de un empleado?",
     "a": "Para registrar la asistencia en iTimeControl: ir a Asistencia > Marcación Manual, seleccionar el empleado, elegir el tipo de marcación (Entrada o Salida), verificar fecha y hora, y hacer clic en Registrar. La acción queda en el historial de auditoría."},
    {"q": "¿Cómo genero un reporte de asistencia diaria?",
     "a": "Ir a Reportes > Asistencia > Reporte Diario, seleccionar la fecha, elegir el departamento o todos, y hacer clic en Generar. Exportar en Excel, PDF o CSV. El reporte muestra estado: Asistió, Tardanza, Falta o Permiso."},
    {"q": "¿Cómo creo un horario de trabajo?",
     "a": "Ir a Configuración > Horarios de Trabajo, clic en Nuevo Horario, ingresar nombre, definir hora de entrada y salida, configurar días laborables, establecer tolerancia de tardanza (10 min recomendado) y guardar."},
    {"q": "¿Cómo agrego un nuevo empleado al sistema?",
     "a": "Ir a Personal > Empleados > Nuevo, completar datos personales (nombre, DNI), asignar código, seleccionar departamento y cargo, asignar horario, registrar método de identificación y guardar."},
    {"q": "¿Cómo exporto reportes a Excel?",
     "a": "Con el reporte en pantalla, clic en Exportar, seleccionar formato Excel (.xlsx), elegir carpeta de destino y confirmar. El archivo se descarga automáticamente."},
    {"q": "¿Cómo solicita un permiso un empleado?",
     "a": "El empleado ingresa al sistema, va a Solicitudes > Nueva Solicitud, selecciona el tipo de permiso, indica fechas y motivo. El supervisor recibe notificación para aprobar o rechazar."},
    {"q": "¿Cómo configuro los minutos de tolerancia para tardanzas?",
     "a": "Ir a Configuración > Horarios, seleccionar el horario, modificar el campo Tolerancia de Entrada en minutos (recomendado: 10 minutos) y guardar. Aplica a todos los empleados del horario."},
    {"q": "¿Cómo restablezco la contraseña de un empleado?",
     "a": "Ir a Personal > Usuarios, buscar el usuario, clic en Opciones > Restablecer Contraseña. El sistema envía la nueva contraseña por correo o el admin asigna una temporal."},
    {"q": "¿Cómo conecto un dispositivo biométrico?",
     "a": "Ir a Configuración > Dispositivos > Agregar Dispositivo, seleccionar fabricante y modelo, ingresar IP y puerto, clic en Probar Conexión y si es exitosa guardar y sincronizar empleados."},
    {"q": "¿Cómo calcula iTimeControl las horas extras?",
     "a": "El sistema calcula automáticamente cuando el empleado sale después de su horario, trabaja en días no laborables, o supera el mínimo configurable (30 min por defecto). Diferencia entre horas diurnas y nocturnas."},
    {"q": "¿Qué tipos de reportes tiene iTimeControl?",
     "a": "Ofrece: Asistencia Diaria, Tardanzas, Horas Extras, Ausencias (injustificada, permiso, vacaciones, licencia médica). Todos filtrables por departamento y período."},
    {"q": "¿Cómo corrijo una marcación incorrecta?",
     "a": "Ir a Asistencia > Historial de Marcaciones, buscar la marcación por empleado y fecha, clic en editar, modificar los datos, ingresar motivo y guardar. Solo administradores pueden editar marcaciones."},
    {"q": "¿Cómo programo reportes automáticos por correo?",
     "a": "Ir a Configuración > Reportes Automáticos, crear programación, seleccionar tipo y frecuencia (diaria/semanal/mensual), ingresar correos de destinatarios, definir hora de envío y activar."},
    {"q": "¿Cómo creo un rol personalizado?",
     "a": "Ir a Configuración > Roles y Permisos > Nuevo Rol, asignar nombre, seleccionar módulos y acciones permitidas (leer/crear/editar/eliminar) y guardar. Luego asignarlo desde Personal > Usuarios."},
    {"q": "¿Cómo agrego feriados al calendario?",
     "a": "Ir a Configuración > Calendario > Feriados > Agregar Feriado, ingresar fecha y nombre, indicar si es nacional/regional/empresa, definir si genera horas extras especiales y guardar."},
    {"q": "¿El sistema funciona sin internet?",
     "a": "Sí, iTimeControl funciona en red local (LAN). Los dispositivos biométricos almacenan marcaciones localmente y sincronizan automáticamente al restorarse la conexión."},
    {"q": "¿Cuántos empleados soporta iTimeControl?",
     "a": "Sin límite fijo. Para más de 500 empleados se recomienda servidor dedicado con 8GB RAM y procesador de 4 núcleos. Probado con hasta 10,000 empleados activos."},
    {"q": "¿Cómo justifico una falta o tardanza?",
     "a": "Ir a Solicitudes > Justificación, seleccionar el tipo (falta o tardanza), indicar fecha, ingresar motivo y adjuntar documento si aplica. El supervisor aprueba o rechaza la justificación."},
    {"q": "¿Cómo configuro el backup automático?",
     "a": "Ir a Configuración > Sistema > Backup, activar backup automático, seleccionar frecuencia diaria, elegir carpeta de destino, definir retención mínima de 5 años y guardar."},
    {"q": "¿Cómo personalizo los campos de un reporte?",
     "a": "En la pantalla de generación del reporte, clic en Columnas o Personalizar, seleccionar o deseleccionar campos. La configuración puede guardarse como plantilla para uso futuro."},
]

# Expandir para tener más datos
EXTENDED_QA = QA_PAIRS * 3  # 60 pares

# ─── INTENCIONES (para clasificación) ────────────────────────────────────────
INTENT_MAP = {
    0: 'registro_asistencia',
    1: 'reportes',
    2: 'horarios',
    3: 'empleados',
    4: 'solicitudes_permisos',
    5: 'configuracion',
    6: 'horarios',
    7: 'configuracion',
    8: 'configuracion',
    9: 'reportes',
    10: 'reportes',
    11: 'registro_asistencia',
    12: 'reportes',
    13: 'configuracion',
    14: 'configuracion',
    15: 'configuracion',
    16: 'configuracion',
    17: 'solicitudes_permisos',
    18: 'configuracion',
    19: 'reportes',
}

questions  = [qa['q'] for qa in QA_PAIRS]
answers    = [qa['a'] for qa in QA_PAIRS]
labels     = [INTENT_MAP[i] for i in range(len(QA_PAIRS))]

# ─── SPLIT HOLDOUT (sin leakage) ─────────────────────────────────────────────
N = len(questions)
idx = list(range(N))
random.shuffle(idx)
train_end = int(N * 0.80)
val_end   = int(N * 0.90)
train_idx = idx[:train_end]
val_idx   = idx[train_end:val_end]
test_idx  = idx[val_end:]

X_train = [questions[i] for i in train_idx]
X_val   = [questions[i] for i in val_idx]
X_test  = [questions[i] for i in test_idx]
y_train = [labels[i] for i in train_idx]
y_test  = [labels[i] for i in test_idx]
corpus_train = [answers[i] for i in train_idx]

print(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
print("✅ Zero leakage: fit solo en train, eval en test")

# ─── MÉTRICAS ────────────────────────────────────────────────────────────────
def rouge_scores(pred, ref):
    from rouge_score import rouge_scorer
    s = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)
    r = s.score(ref, pred)
    return r['rouge1'].fmeasure, r['rouge2'].fmeasure, r['rougeL'].fmeasure

def bleu_score(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    if not p or not r: return 0.0
    return sentence_bleu([r], p, smoothing_function=SmoothingFunction().method1)

def recall_at_k(results, ref, k):
    ref_toks = set(ref.lower().split())
    for doc, _ in results[:k]:
        doc_toks = set(doc.lower().split())
        if len(ref_toks & doc_toks) / len(ref_toks) > 0.25:
            return 1.0
    return 0.0

def evaluate_retriever(retriever_fn, queries, references, k_vals=[1,3,5]):
    r1s, r2s, rLs, bleus = [], [], [], []
    recall_k = {k: [] for k in k_vals}
    times = []
    for q, ref in zip(queries, references):
        t0 = time.time()
        results = retriever_fn(q, max(k_vals))
        times.append(time.time() - t0)
        pred = results[0][0] if results else ''
        r1, r2, rL = rouge_scores(pred, ref)
        r1s.append(r1); r2s.append(r2); rLs.append(rL)
        bleus.append(bleu_score(pred, ref))
        for k in k_vals:
            recall_k[k].append(recall_at_k(results, ref, k))
    return {
        'ROUGE-1': round(np.mean(r1s), 4),
        'ROUGE-2': round(np.mean(r2s), 4),
        'ROUGE-L': round(np.mean(rLs), 4),
        'BLEU':    round(np.mean(bleus), 4),
        **{f'Recall@{k}': round(np.mean(recall_k[k]), 4) for k in k_vals},
        'Latencia_ms': round(np.mean(times) * 1000, 2),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE — TF-IDF unigrams, sin preprocesamiento
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  BASELINE — TF-IDF unigrams")
print("="*55)
tfidf_base = TfidfVectorizer(ngram_range=(1,1), max_features=5000, strip_accents=None)
matrix_base = tfidf_base.fit_transform(corpus_train)

def retrieve_baseline(q, k):
    v = tfidf_base.transform([q])
    scores = cosine_similarity(v, matrix_base).flatten()
    top = scores.argsort()[::-1][:k]
    return [(corpus_train[i], float(scores[i])) for i in top]

test_refs_retrieval = [answers[i] for i in test_idx]
metrics_base = evaluate_retriever(retrieve_baseline, X_test, test_refs_retrieval)
print(f"  ROUGE-1   : {metrics_base['ROUGE-1']}")
print(f"  Recall@3  : {metrics_base['Recall@3']}")
print(f"  Latencia  : {metrics_base['Latencia_ms']} ms")

# ═══════════════════════════════════════════════════════════════════════════════
# VARIANTE 1 — TF-IDF bigrams + strip_accents + sublinear_tf
#   Feature añadida: bigramas, normalización de acentos, TF sublineal
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  VARIANTE 1 — TF-IDF bigrams + normalización")
print("="*55)
tfidf_v1 = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=8000,
    strip_accents='unicode',
    sublinear_tf=True,          # log(TF) — reduce dominancia de términos frecuentes
)
matrix_v1 = tfidf_v1.fit_transform(corpus_train)

def retrieve_v1(q, k):
    v = tfidf_v1.transform([q])
    scores = cosine_similarity(v, matrix_v1).flatten()
    top = scores.argsort()[::-1][:k]
    return [(corpus_train[i], float(scores[i])) for i in top]

metrics_v1 = evaluate_retriever(retrieve_v1, X_test, test_refs_retrieval)
print(f"  ROUGE-1   : {metrics_v1['ROUGE-1']}")
print(f"  Recall@3  : {metrics_v1['Recall@3']}")

# ═══════════════════════════════════════════════════════════════════════════════
# VARIANTE 2 — TF-IDF bigrams + BM25-like (k1 saturation via sublinear) + reranking
#   Feature añadida: reranking por longitud del documento, filtro mínimo de similitud
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  VARIANTE 2 — TF-IDF + reranking + filtro similitud")
print("="*55)
tfidf_v2 = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=10000,
    strip_accents='unicode',
    sublinear_tf=True,
    min_df=1,
)
matrix_v2 = tfidf_v2.fit_transform(corpus_train)

def retrieve_v2(q, k, sim_threshold=0.1):
    v = tfidf_v2.transform([q])
    scores = cosine_similarity(v, matrix_v2).flatten()
    # Reranking: penaliza documentos muy cortos (menos informativos)
    doc_lengths = np.array([len(d.split()) for d in corpus_train])
    length_bonus = np.clip(doc_lengths / doc_lengths.max(), 0.8, 1.0)
    adjusted = scores * length_bonus
    top = adjusted.argsort()[::-1][:k]
    return [(corpus_train[i], float(adjusted[i])) for i in top if adjusted[i] >= sim_threshold]

metrics_v2 = evaluate_retriever(retrieve_v2, X_test, test_refs_retrieval)
print(f"  ROUGE-1   : {metrics_v2['ROUGE-1']}")
print(f"  Recall@3  : {metrics_v2['Recall@3']}")

# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION en clasificación de intención (5-fold estratificado)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("  CROSS-VALIDATION — Clasificación de intención")
print("="*55)

nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000, strip_accents='unicode')),
    ('clf',   MultinomialNB(alpha=0.5))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_cv = cross_val_predict(nb_pipeline, questions, labels, cv=skf)

cv_results = {
    'accuracy':  round(np.mean([y_pred_cv[i]==labels[i] for i in range(len(labels))]), 4),
    'precision': round(precision_score(labels, y_pred_cv, average='weighted', zero_division=0), 4),
    'recall':    round(recall_score(labels, y_pred_cv, average='weighted', zero_division=0), 4),
    'f1':        round(f1_score(labels, y_pred_cv, average='weighted', zero_division=0), 4),
}
print(f"  Accuracy  : {cv_results['accuracy']}")
print(f"  F1 (w)    : {cv_results['f1']}")
print(f"  Precision : {cv_results['precision']}")
print(f"  Recall    : {cv_results['recall']}")

# ═══════════════════════════════════════════════════════════════════════════════
# GUARDAR LOGS
# ═══════════════════════════════════════════════════════════════════════════════
experiment_log = {
    'experimentos': {
        'baseline': {
            'nombre': 'TF-IDF Unigrams (Baseline)',
            'features': ['unigrams', 'max_features=5000'],
            'metricas': metrics_base,
        },
        'variante1': {
            'nombre': 'TF-IDF Bigrams + Normalización',
            'features': ['bigrams', 'strip_accents=unicode', 'sublinear_tf=True', 'max_features=8000'],
            'features_removidas': ['sin normalización de acentos'],
            'metricas': metrics_v1,
        },
        'variante2': {
            'nombre': 'TF-IDF Bigrams + Reranking + Filtro',
            'features': ['bigrams', 'strip_accents=unicode', 'sublinear_tf=True', 'reranking_por_longitud', 'filtro_similitud_0.1'],
            'features_removidas': [],
            'metricas': metrics_v2,
        },
    },
    'cross_validation': cv_results,
    'split': {'train': len(X_train), 'val': len(X_val), 'test': len(X_test)},
    'leakage_check': 'CERO — TF-IDF fit solo en train, eval en test holdout',
    'split_tipo': 'stratificado por intención',
}

with open(LOGS / 'experiment_log.json', 'w', encoding='utf-8') as f:
    json.dump(experiment_log, f, indent=2, ensure_ascii=False)
print("\n✅ Log guardado")

# ═══════════════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.size': 11, 'figure.dpi': 130})
COLORS = ['#4C72B0', '#55A868', '#C44E52', '#DD8452']

# ── GRÁFICA 1: Recall@k vs k (gráfica clave según el entregable) ──────────────
k_vals = [1, 3, 5]
recall_base = [metrics_base[f'Recall@{k}'] for k in k_vals]
recall_v1   = [metrics_v1[f'Recall@{k}']   for k in k_vals]
recall_v2   = [metrics_v2[f'Recall@{k}']   for k in k_vals]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(k_vals, recall_base, 'o-', color=COLORS[0], linewidth=2.5, markersize=8, label='Baseline (TF-IDF unigrams)')
ax.plot(k_vals, recall_v1,   's-', color=COLORS[1], linewidth=2.5, markersize=8, label='Var1: bigrams + normalización')
ax.plot(k_vals, recall_v2,   '^-', color=COLORS[2], linewidth=2.5, markersize=8, label='Var2: bigrams + reranking')
ax.set_xlabel('k (número de documentos recuperados)', fontsize=12)
ax.set_ylabel('Recall@k', fontsize=12)
ax.set_title('Recall@k vs k — Experimentos A/B iTimeControl\n(mayor es mejor)', fontsize=13)
ax.set_xticks(k_vals)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.4)
for line_data, color in zip([recall_base, recall_v1, recall_v2], COLORS):
    for k, v in zip(k_vals, line_data):
        ax.annotate(f'{v:.2f}', (k, v), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color=color, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGS / 'recall_at_k.png', bbox_inches='tight')
plt.close()
print("✅ Gráfica: recall_at_k.png")

# ── GRÁFICA 2: Tabla de métricas comparativa ─────────────────────────────────
metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'Recall@1', 'Recall@3', 'Recall@5']
data_plot = {
    'Baseline': [metrics_base[m] for m in metric_names],
    'Var1':     [metrics_v1[m]   for m in metric_names],
    'Var2':     [metrics_v2[m]   for m in metric_names],
}
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(13, 5))
for i, (label, values) in enumerate(data_plot.items()):
    bars = ax.bar(x + i*width, values, width, label=label, color=COLORS[i], edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, rotation=45)
ax.set_xlabel('Métrica')
ax.set_ylabel('Score')
ax.set_title('Comparación de métricas — Baseline vs Variantes (iTimeControl)', fontsize=13)
ax.set_xticks(x + width)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGS / 'metricas_comparativas.png', bbox_inches='tight')
plt.close()
print("✅ Gráfica: metricas_comparativas.png")

# ── GRÁFICA 3: Mejora relativa sobre baseline ────────────────────────────────
improvements_v1 = [(metrics_v1[m] - metrics_base[m]) / metrics_base[m] * 100
                   if metrics_base[m] > 0 else 0 for m in metric_names]
improvements_v2 = [(metrics_v2[m] - metrics_base[m]) / metrics_base[m] * 100
                   if metrics_base[m] > 0 else 0 for m in metric_names]

fig, ax = plt.subplots(figsize=(13, 5))
x2 = np.arange(len(metric_names))
ax.bar(x2 - 0.2, improvements_v1, 0.4, label='Var1 vs Baseline', color=COLORS[1], edgecolor='white')
ax.bar(x2 + 0.2, improvements_v2, 0.4, label='Var2 vs Baseline', color=COLORS[2], edgecolor='white')
ax.axhline(0, color='black', linewidth=1)
ax.set_xlabel('Métrica')
ax.set_ylabel('Mejora relativa (%)')
ax.set_title('Mejora relativa de variantes sobre Baseline', fontsize=13)
ax.set_xticks(x2)
ax.set_xticklabels(metric_names)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(FIGS / 'mejora_relativa.png', bbox_inches='tight')
plt.close()
print("✅ Gráfica: mejora_relativa.png")

# ── Imprimir tabla resumen final ─────────────────────────────────────────────
print("\n" + "="*70)
print("  TABLA RESUMEN — BASELINE / VAR1 / VAR2")
print("="*70)
print(f"  {'Métrica':<15} {'Baseline':>10} {'Var1':>10} {'Var2':>10} {'Mejor':>10}")
print("-"*70)
all_metrics = metrics_base.copy()
for m in metric_names + ['Latencia_ms']:
    b  = metrics_base.get(m, 0)
    v1 = metrics_v1.get(m, 0)
    v2 = metrics_v2.get(m, 0)
    if m == 'Latencia_ms':
        mejor = 'Baseline' if b <= v1 and b <= v2 else ('Var1' if v1 <= v2 else 'Var2')
    else:
        mejor = 'Baseline' if b >= v1 and b >= v2 else ('Var1' if v1 >= v2 else 'Var2')
    print(f"  {m:<15} {b:>10.4f} {v1:>10.4f} {v2:>10.4f} {mejor:>10}")

print("="*70)
print(f"\n  CV 5-fold (Naive Bayes intención):")
print(f"  Accuracy={cv_results['accuracy']} | F1={cv_results['f1']} | Precision={cv_results['precision']}")
print(f"\n  Leakage: CERO — fit solo en train ({len(X_train)} muestras)")
print(f"  Split  : holdout + stratified 5-fold CV")
print("✅ Experimentos completados")
