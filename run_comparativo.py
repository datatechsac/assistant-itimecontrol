"""
Script temporal: ejecuta el análisis completo de comparativo de latencia.
Genera todos los artefactos en logs/.
"""
import os, sys, json, time, random, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

LOGS_DIR     = ROOT / "logs"
DATASETS_DIR = ROOT / "data" / "datasets"
LOGS_DIR.mkdir(exist_ok=True)

COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2", "#937860"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. Cargando métricas...")
print("=" * 60)

with open(LOGS_DIR / "baseline_metrics.json", encoding="utf-8") as f:
    baseline_data = json.load(f)

with open(LOGS_DIR / "rag_evaluation_results.json", encoding="utf-8") as f:
    rag_data = json.load(f)
rag_metrics = rag_data["summary"]
rag_model   = rag_data.get("model", "llama-3.3-70b-versatile")
rag_n       = rag_data["total_questions"]

print(f"  Baseline cargado: {list(baseline_data.keys())}")
print(f"  RAG cargado: {rag_model} | {rag_n} preguntas")
for k, v in rag_metrics.items():
    print(f"    {k:<20}: {v:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TABLA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Tabla comparativa baseline vs RAG")
print("=" * 60)

bl = baseline_data
METRIC_MAP = {
    "ROUGE-1": ("rouge1",  "rouge1"),
    "ROUGE-2": ("rouge2",  "rouge2"),
    "ROUGE-L": ("rougeL",  "rougeL"),
    "BLEU":    ("bleu",    "bleu"),
}

rows = []
for label, (bl_key, rag_key) in METRIC_MAP.items():
    tf  = bl["tfidf_coseno"].get(bl_key, 0)
    nb  = bl["naive_bayes"].get(bl_key, 0)
    knn = bl["knn"].get(bl_key, 0)
    rag = rag_metrics.get(rag_key, 0)
    delta = round((rag - tf) / tf * 100, 1) if tf > 0 else None
    rows.append({
        "Métrica":          label,
        "TF-IDF (Baseline)": round(tf, 4),
        "Naive Bayes":       round(nb, 4),
        "KNN":               round(knn, 4),
        "RAG (Propuesto)":   round(rag, 4),
        "Delta_vs_TFIDF":    f"{delta:+.1f}%" if delta is not None else "-",
    })

for label, rag_key in [("Hit Rate@K", "hit_rate"), ("Context Recall", "context_recall")]:
    rag = rag_metrics.get(rag_key, 0)
    rows.append({
        "Métrica":           label,
        "TF-IDF (Baseline)": "N/A",
        "Naive Bayes":       "N/A",
        "KNN":               "N/A",
        "RAG (Propuesto)":   round(rag, 4),
        "Delta_vs_TFIDF":    "N/A (RAG-only)",
    })

df_compare = pd.DataFrame(rows)
print("\n" + "=" * 75)
print("  COMPARATIVO TÉCNICO — BASELINE vs RAG (Propuesto)")
print("=" * 75)
print(df_compare.to_string(index=False))
print("=" * 75)
df_compare.to_csv(LOGS_DIR / "comparativo_baseline_rag.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 3. GRÁFICA COMPARATIVA
# ─────────────────────────────────────────────────────────────────────────────
print("\n3. Generando gráfica comparativa...")

numeric_metrics = ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]
df_num = df_compare[df_compare["Métrica"].isin(numeric_metrics)].copy()

fig = plt.figure(figsize=(16, 6))
fig.suptitle(
    f"Comparativo Técnico — Baseline vs RAG ({rag_model})\n"
    f"Asistente Conversacional iTimeControl",
    fontsize=13,
)

# Panel izquierdo: barras agrupadas
ax1 = fig.add_subplot(121)
models_plot = ["TF-IDF (Baseline)", "Naive Bayes", "KNN", "RAG (Propuesto)"]
model_colors = [COLORS[0], COLORS[2], COLORS[3], COLORS[1]]
x = np.arange(len(numeric_metrics))
width = 0.2

for i, (model, color) in enumerate(zip(models_plot, model_colors)):
    vals = [float(row[model]) if isinstance(row[model], (int, float)) else 0
            for _, row in df_num.iterrows()]
    hatch = "////" if model == "RAG (Propuesto)" else ""
    bars = ax1.bar(x + i * width, vals, width, label=model,
                   color=color, edgecolor="white", hatch=hatch, alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax1.set_xticks(x + width * 1.5)
ax1.set_xticklabels(numeric_metrics)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel("Score")
ax1.set_title("Métricas de calidad de respuesta")
ax1.legend(fontsize=8)

# Panel derecho: radar TF-IDF vs RAG
ax2 = fig.add_subplot(122, polar=True)
cats   = numeric_metrics
angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
angles_c = angles + angles[:1]

tfidf_vals = [float(bl["tfidf_coseno"].get(k, 0))
              for k in ["rouge1", "rouge2", "rougeL", "bleu"]]
rag_vals   = [float(rag_metrics.get(k, 0))
              for k in ["rouge1", "rouge2", "rougeL", "bleu"]]

for vals, color, label, ls in [
    (tfidf_vals, COLORS[0], "TF-IDF (Baseline)", "--"),
    (rag_vals,   COLORS[1], f"RAG ({rag_model[:20]})", "-"),
]:
    v = vals + [vals[0]]
    ax2.plot(angles_c, v, ls, linewidth=2, color=color, label=label)
    ax2.fill(angles_c, v, alpha=0.12, color=color)

ax2.set_xticks(angles)
ax2.set_xticklabels(cats, fontsize=9)
ax2.set_ylim(0, 1)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
ax2.set_title("Radar: TF-IDF vs RAG", pad=20, fontsize=11)
ax2.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8)

plt.tight_layout()
out = LOGS_DIR / "comparativo_tecnico.png"
plt.savefig(str(out), bbox_inches="tight")
plt.close()
print(f"  Guardado: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. BENCHMARK DE LATENCIA (N=20)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Benchmark de latencia (N=20, Groq)")
print("=" * 60)

from src.utils.helpers import load_config
from src.rag.pipeline import RAGPipeline

config = load_config()
rag_pipeline = RAGPipeline(config)

with open(DATASETS_DIR / "benchmark_questions.json", encoding="utf-8") as f:
    all_questions = json.load(f)

random.seed(42)
sample_qs = random.sample(all_questions, min(20, len(all_questions)))

latency_records = []
errors = 0

for i, item in enumerate(sample_qs, 1):
    question = item["question"]
    try:
        t0 = time.perf_counter()
        retrieved   = rag_pipeline.retriever.search(question)
        context     = rag_pipeline.retriever.format_context(retrieved)
        t_ret       = time.perf_counter() - t0

        t1 = time.perf_counter()
        msg = f"Contexto de iTimeControl:\n{context}\n\nPregunta: {question}" if context else question
        answer = rag_pipeline._call_llm(msg)
        t_gen  = time.perf_counter() - t1
        t_tot  = time.perf_counter() - t0

        latency_records.append({
            "query_id":     i,
            "question":     question[:55],
            "t_retrieval":  round(t_ret * 1000, 1),
            "t_generation": round(t_gen * 1000, 1),
            "t_total":      round(t_tot * 1000, 1),
            "status":       "ok",
        })
        print(f"  [{i:02d}/{len(sample_qs)}] {t_tot*1000:.0f}ms  "
              f"(ret={t_ret*1000:.0f}ms | gen={t_gen*1000:.0f}ms)")
    except Exception as e:
        errors += 1
        latency_records.append({
            "query_id": i, "question": question[:55],
            "t_retrieval": None, "t_generation": None, "t_total": None,
            "status": f"error: {str(e)[:50]}",
        })
        print(f"  [{i:02d}] ERROR: {e}")

df_lat = pd.DataFrame(latency_records)
df_ok  = df_lat[df_lat["status"] == "ok"].copy()

# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTADÍSTICAS DE LATENCIA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Estadísticas de latencia")
print("=" * 60)

stats = {}
for col in ["t_retrieval", "t_generation", "t_total"]:
    vals = df_ok[col].dropna().values
    stats[col] = {
        "p50":  round(float(np.percentile(vals, 50)), 1),
        "p95":  round(float(np.percentile(vals, 95)), 1),
        "p99":  round(float(np.percentile(vals, 99)), 1),
        "mean": round(float(vals.mean()), 1),
        "std":  round(float(vals.std()), 1),
        "min":  round(float(vals.min()), 1),
        "max":  round(float(vals.max()), 1),
    }

total_time_s  = df_ok["t_total"].sum() / 1000
throughput_qpm = round(len(df_ok) / total_time_s * 60, 2) if total_time_s > 0 else 0
error_rate     = round(errors / len(sample_qs) * 100, 1)

print(f"\n{'='*60}")
print(f"  INFORME DE LATENCIA — RAG iTimeControl ({rag_model})")
print(f"{'='*60}")
print(f"  Consultas : {len(sample_qs)} | OK: {len(df_ok)} | Errores: {errors}")
print(f"  Tasa error: {error_rate}%")
print(f"  Throughput: {throughput_qpm} queries/min")
print()
for col, label in [
    ("t_retrieval",  "Retrieval FAISS"),
    ("t_generation", f"Generación Groq"),
    ("t_total",      "Total"),
]:
    s = stats[col]
    print(f"  {label}:")
    print(f"    p50={s['p50']}ms  p95={s['p95']}ms  p99={s['p99']}ms")
    print(f"    media={s['mean']}ms  std={s['std']}ms  "
          f"min={s['min']}ms  max={s['max']}ms")
print(f"{'='*60}")

latency_report = {
    "n_queries": len(sample_qs),
    "n_ok": len(df_ok),
    "error_rate_pct": error_rate,
    "throughput_qpm": throughput_qpm,
    "provider": config.get("generation", {}).get("provider", "groq"),
    "model": rag_model,
    "stats_ms": stats,
}
with open(LOGS_DIR / "latency_report.json", "w", encoding="utf-8") as f:
    json.dump(latency_report, f, indent=2)
df_lat.to_csv(LOGS_DIR / "latency_records.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 6. GRÁFICAS DE LATENCIA
# ─────────────────────────────────────────────────────────────────────────────
print("\n6. Generando gráficas de latencia...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f"Informe de Latencia — RAG iTimeControl\n({rag_model} via Groq + FAISS CPU)", fontsize=13)

# 1. Histograma latencia total
ax = axes[0, 0]
ax.hist(df_ok["t_total"], bins=10, color=COLORS[1], edgecolor="white", alpha=0.85)
ax.axvline(stats["t_total"]["p50"], color="red", linestyle="--", linewidth=1.5,
           label=f"p50={stats['t_total']['p50']}ms")
ax.axvline(stats["t_total"]["p95"], color="orange", linestyle="--", linewidth=1.5,
           label=f"p95={stats['t_total']['p95']}ms")
ax.set_xlabel("Latencia total (ms)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de latencia total")
ax.legend(fontsize=9)

# 2. Boxplot por componente
ax = axes[0, 1]
boxdata = [df_ok["t_retrieval"].dropna().values,
           df_ok["t_generation"].dropna().values,
           df_ok["t_total"].dropna().values]
bp = ax.boxplot(boxdata, patch_artist=True, widths=0.5,
                labels=["Retrieval\n(FAISS)", f"Generación\n(Groq)", "Total"])
for patch, color in zip(bp["boxes"], [COLORS[0], COLORS[2], COLORS[1]]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Latencia (ms)")
ax.set_title("Distribución por componente")
ax.grid(True, alpha=0.4, axis="y")

# 3. Timeline de latencia por consulta (stacked bar)
ax = axes[1, 0]
x_ids = df_ok["query_id"].values
ax.bar(x_ids, df_ok["t_retrieval"], label="Retrieval", color=COLORS[0], alpha=0.85)
ax.bar(x_ids, df_ok["t_generation"], bottom=df_ok["t_retrieval"],
       label="Generación", color=COLORS[2], alpha=0.85)
ax.axhline(stats["t_total"]["p50"], color="red", linestyle="--", linewidth=1.2,
           label=f"p50={stats['t_total']['p50']}ms")
ax.axhline(stats["t_total"]["p95"], color="orange", linestyle="--", linewidth=1.2,
           label=f"p95={stats['t_total']['p95']}ms")
ax.set_xlabel("Consulta #")
ax.set_ylabel("Latencia (ms)")
ax.set_title("Desglose de latencia por consulta")
ax.legend(fontsize=8)

# 4. Tabla resumen visual
ax = axes[1, 1]
ax.axis("off")
table_data = [
    ["Métrica", "Retrieval", "Generación", "Total"],
    ["p50 (ms)",   str(stats["t_retrieval"]["p50"]),  str(stats["t_generation"]["p50"]),  str(stats["t_total"]["p50"])],
    ["p95 (ms)",   str(stats["t_retrieval"]["p95"]),  str(stats["t_generation"]["p95"]),  str(stats["t_total"]["p95"])],
    ["p99 (ms)",   str(stats["t_retrieval"]["p99"]),  str(stats["t_generation"]["p99"]),  str(stats["t_total"]["p99"])],
    ["Media (ms)", str(stats["t_retrieval"]["mean"]), str(stats["t_generation"]["mean"]), str(stats["t_total"]["mean"])],
    ["", "", "", ""],
    ["Throughput",  f"{throughput_qpm} q/min", "", ""],
    ["Tasa error",  f"{error_rate}%",          "", ""],
    ["Proveedor",   "Groq", "", ""],
]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")
ax.set_title("Resumen estadístico", fontweight="bold", pad=20)

plt.tight_layout()
out = LOGS_DIR / "informe_latencia.png"
plt.savefig(str(out), bbox_inches="tight")
plt.close()
print(f"  Guardado: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. OPTIMIZACIÓN top_k
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. Optimización top_k (N=10 consultas × 4 valores)")
print("=" * 60)

from src.evaluation.metrics import evaluate_single as _eval_single

TOP_K_VALUES = [3, 5, 7, 10]
opt_sample   = sample_qs[:10]
opt_results  = []

for top_k in TOP_K_VALUES:
    print(f"\n  top_k={top_k}...")
    times, rouge1s = [], []
    for item in opt_sample:
        try:
            t0 = time.perf_counter()
            ret = rag_pipeline.retriever.search(item["question"], top_k=top_k)
            ctx = rag_pipeline.retriever.format_context(ret)
            msg = f"Contexto de iTimeControl:\n{ctx}\n\nPregunta: {item['question']}" if ctx else item["question"]
            ans = rag_pipeline._call_llm(msg)
            t   = (time.perf_counter() - t0) * 1000
            sc  = _eval_single(ans, item["answer"])
            times.append(t)
            rouge1s.append(sc["rouge1"])
        except Exception as e:
            print(f"    error: {e}")

    r = {
        "top_k":       top_k,
        "lat_p50_ms":  round(float(np.percentile(times, 50)), 1) if times else None,
        "lat_p95_ms":  round(float(np.percentile(times, 95)), 1) if times else None,
        "lat_mean_ms": round(float(np.mean(times)),           1) if times else None,
        "rouge1_mean": round(float(np.mean(rouge1s)),         4) if rouge1s else None,
        "n_ok":        len(times),
    }
    opt_results.append(r)
    print(f"    p50={r['lat_p50_ms']}ms  p95={r['lat_p95_ms']}ms  ROUGE-1={r['rouge1_mean']}")

df_opt = pd.DataFrame(opt_results)
print("\n" + df_opt.to_string(index=False))
df_opt.to_csv(LOGS_DIR / "optimization_topk.csv", index=False)

# Gráfica optimización
df_opt_v = df_opt.dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Optimización: Impacto del top_k en Latencia y Calidad", fontsize=13)

ax = axes[0]
ax.plot(df_opt_v["top_k"], df_opt_v["lat_p50_ms"],  "o-", color=COLORS[0], linewidth=2, label="p50")
ax.plot(df_opt_v["top_k"], df_opt_v["lat_p95_ms"],  "s--", color=COLORS[2], linewidth=2, label="p95")
ax.plot(df_opt_v["top_k"], df_opt_v["lat_mean_ms"], "^:", color=COLORS[3], linewidth=2, label="media")
for _, row in df_opt_v.iterrows():
    ax.annotate(f"{row['lat_p50_ms']}ms", (row["top_k"], row["lat_p50_ms"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.set_xlabel("top_k")
ax.set_ylabel("Latencia (ms)")
ax.set_title("Latencia vs top_k")
ax.set_xticks(TOP_K_VALUES)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.4)

ax2 = axes[1]
bars = ax2.bar(df_opt_v["top_k"].astype(str), df_opt_v["rouge1_mean"],
               color=COLORS[1], edgecolor="white", alpha=0.85, width=0.5)
for bar, val in zip(bars, df_opt_v["rouge1_mean"]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
             f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_xlabel("top_k")
ax2.set_ylabel("ROUGE-1 (media)")
ax2.set_title("Calidad vs top_k")
ax2.set_ylim(0, max(df_opt_v["rouge1_mean"]) * 1.4 if len(df_opt_v) else 1)
ax2.grid(True, alpha=0.4, axis="y")

plt.tight_layout()
out = LOGS_DIR / "optimizacion_topk.png"
plt.savefig(str(out), bbox_inches="tight")
plt.close()
print(f"\n  Guardado: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN FINAL — SISTEMA RAG iTimeControl")
print("=" * 65)

best_topk = df_opt_v.loc[df_opt_v["rouge1_mean"].idxmax()] if len(df_opt_v) > 0 else None

print(f"\n  Modelo   : {rag_model}")
print(f"  Proveedor: {config.get('generation', {}).get('provider', 'groq')}")
print(f"  Eval N   : {rag_n} preguntas")

print("\n  [1] MÉTRICAS DE CALIDAD")
print(f"  {'Métrica':<20} {'TF-IDF':>10} {'RAG':>10} {'Δ':>10}")
print(f"  {'-'*52}")
for _, row in df_compare[df_compare["Métrica"].isin(["ROUGE-1","ROUGE-2","ROUGE-L","BLEU"])].iterrows():
    print(f"  {row['Métrica']:<20} {str(row['TF-IDF (Baseline)']):>10} "
          f"{str(row['RAG (Propuesto)']):>10} {str(row['Delta_vs_TFIDF']):>10}")
print(f"\n  Hit Rate@5      : {rag_metrics['hit_rate']:.4f}")
print(f"  Context Recall  : {rag_metrics['context_recall']:.4f}")

print("\n  [2] LATENCIA (Groq)")
print(f"  p50 total  : {stats['t_total']['p50']} ms")
print(f"  p95 total  : {stats['t_total']['p95']} ms")
print(f"  Throughput : {throughput_qpm} queries/min")
print(f"  Tasa error : {error_rate}%")
print(f"  Retrieval  : p50={stats['t_retrieval']['p50']}ms p95={stats['t_retrieval']['p95']}ms")
print(f"  Generación : p50={stats['t_generation']['p50']}ms p95={stats['t_generation']['p95']}ms")

if best_topk is not None:
    print(f"\n  [3] MEJOR top_k: {int(best_topk['top_k'])}")
    print(f"  ROUGE-1 = {best_topk['rouge1_mean']}  |  p50 = {best_topk['lat_p50_ms']}ms")

print("\n" + "=" * 65)

# Guardar resumen JSON final
import datetime
summary_out = {
    "fecha":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "modelo":       rag_model,
    "proveedor":    config.get("generation", {}).get("provider", "groq"),
    "eval_n":       rag_n,
    "metricas_rag": rag_metrics,
    "latencia": {
        "p50_ms":         stats["t_total"]["p50"],
        "p95_ms":         stats["t_total"]["p95"],
        "p99_ms":         stats["t_total"]["p99"],
        "throughput_qpm": throughput_qpm,
        "error_rate_pct": error_rate,
        "retrieval_p50":  stats["t_retrieval"]["p50"],
        "generation_p50": stats["t_generation"]["p50"],
    },
    "mejor_topk": int(best_topk["top_k"]) if best_topk is not None else None,
}
with open(LOGS_DIR / "resumen_semana13.json", "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

print("\nArtefactos generados:")
for art in [
    "comparativo_baseline_rag.csv",
    "comparativo_tecnico.png",
    "informe_latencia.png",
    "optimization_topk.csv",
    "optimizacion_topk.png",
    "latency_report.json",
    "latency_records.csv",
    "resumen_semana13.json",
]:
    path = LOGS_DIR / art
    mark = "OK" if path.exists() else "MISSING"
    print(f"  [{mark}] logs/{art}")
