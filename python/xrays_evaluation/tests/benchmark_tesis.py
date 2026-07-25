"""
Benchmark comparativo de los tres modelos entrenados (nano / medium / xlarge)
para el capitulo de resultados del trabajo de grado.

Diseno de la evaluacion
-----------------------
Se evalua sobre dos conjuntos independientes del entrenamiento:

  1. split_data/test  (900 img)  - particion reservada del dataset de
     entrenamiento; el modelo nunca la vio (no existe test.cache).
  2. benchmarking     (200 img)  - recoleccion posterior e independiente,
     balanceada 100/100.

La clase positiva es "Anomaly": en un contexto clinico el evento de interes
es la presencia de hallazgo patologico, de modo que la sensibilidad mide la
capacidad de no pasar por alto un caso anomalo (falso negativo = riesgo).

Las metricas se calculan de forma explicita con numpy en lugar de delegarlas
a una libreria, para que cada valor reportado sea trazable y verificable.

Uso:
    python tests/benchmark_tesis.py [--sin-hash]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

# --- Configuracion ---
RAIZ = Path(__file__).resolve().parent.parent
MODELOS = {
    "YOLO11n-cls (nano)":   RAIZ / "models/YOLO/xrays_evaluation_model_nano_v1.pt",
    "YOLO11m-cls (medium)": RAIZ / "models/YOLO/xrays_evaluation_model_medium_v1.pt",
    "YOLO11x-cls (xlarge)": RAIZ / "models/YOLO/xrays_evaluation_model_xlarge_v1.pt",
}

# Ubicacion del dataset. No se codifica una ruta local para que el script
# funcione en cualquier equipo: se indica con la variable de entorno
# XRAY_DATASET_DIR y, en su defecto, se busca en datasets/ en la raiz del
# repositorio, que es la convencion declarada en .gitignore.
DATOS = Path(os.getenv("XRAY_DATASET_DIR", RAIZ.parent.parent / "datasets"))
BASE_TRAIN = DATOS / "train/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121"
BASE_TEST = DATOS / "test/ingeniia_services_xrays_evaluation_img_v1.0.0_test_20251130"

CONJUNTOS = {
    "Prueba interna (split_data/test)": BASE_TRAIN / "split_data/test",
    "Validacion externa (benchmarking)": BASE_TEST / "benchmarking",
}
CONJUNTO_ENTRENAMIENTO = BASE_TRAIN / "split_data/train"

# anomaly = 0 (clase positiva), normal = 1
CARPETA_A_ETIQUETA = {"anomaly": 0, "normal": 1}
NOMBRES = ["Anomaly", "Normal"]
POSITIVA = 0

SALIDA = RAIZ / "tests/results/benchmark_tesis"
EXTENSIONES = ("*.jpg", "*.jpeg", "*.png")


# --- Carga de datos ---
def listar_imagenes(base: Path) -> pd.DataFrame:
    filas = []
    for carpeta, etiqueta in CARPETA_A_ETIQUETA.items():
        d = base / carpeta
        if not d.is_dir():
            raise FileNotFoundError(f"No se encontro la carpeta: {d}")
        for patron in EXTENSIONES:
            for p in d.glob(patron):
                filas.append({"ruta": str(p), "y_true": etiqueta})
    if not filas:
        raise RuntimeError(f"Sin imagenes en {base}")
    return pd.DataFrame(filas)


# --- Verificacion de fuga de datos ---
def hash_archivo(ruta: Path, bloque: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def hashes_de(base: Path) -> set:
    salida = set()
    for carpeta in CARPETA_A_ETIQUETA:
        for patron in EXTENSIONES:
            for p in (base / carpeta).glob(patron):
                salida.add(hash_archivo(p))
    return salida


def verificar_fuga() -> dict:
    """
    Comprueba que ninguna imagen de evaluacion aparezca en entrenamiento.
    Una coincidencia invalidaria las metricas por fuga de datos.
    """
    print("\n[1/3] Verificando ausencia de fuga de datos (hash MD5)...")
    t0 = time.time()
    h_train = hashes_de(CONJUNTO_ENTRENAMIENTO)
    print(f"      entrenamiento: {len(h_train)} imagenes unicas")

    informe = {"entrenamiento_unicas": len(h_train)}
    for nombre, base in CONJUNTOS.items():
        h_eval = hashes_de(base)
        solapamiento = h_train & h_eval
        informe[nombre] = {
            "unicas": len(h_eval),
            "coincidencias_con_entrenamiento": len(solapamiento),
        }
        estado = "LIMPIO" if not solapamiento else f"ALERTA: {len(solapamiento)} duplicadas"
        print(f"      {nombre}: {len(h_eval)} unicas -> {estado}")

    print(f"      ({time.time() - t0:.0f} s)")
    return informe


# --- Metricas ---
def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """cm[i, j] = casos de clase real i predichos como j."""
    cm = np.zeros((2, 2), dtype=int)
    for real, pred in zip(y_true, y_pred):
        cm[real, pred] += 1
    return cm


def metricas(cm: np.ndarray) -> dict:
    """Metricas tomando Anomaly (indice 0) como clase positiva."""
    vp = cm[POSITIVA, POSITIVA]          # anomalia detectada
    fn = cm[POSITIVA, 1 - POSITIVA]      # anomalia no detectada (grave)
    fp = cm[1 - POSITIVA, POSITIVA]      # falsa alarma
    vn = cm[1 - POSITIVA, 1 - POSITIVA]  # normal correcto
    total = vp + fn + fp + vn

    div = lambda a, b: a / b if b else 0.0
    sens = div(vp, vp + fn)
    esp = div(vn, vn + fp)
    prec = div(vp, vp + fp)

    return {
        "exactitud": div(vp + vn, total),
        "precision": prec,
        "sensibilidad": sens,
        "especificidad": esp,
        "f1": div(2 * prec * sens, prec + sens),
        "vpn": div(vn, vn + fn),
        "VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn),
    }


def curva_roc(y_true: np.ndarray, puntajes: np.ndarray):
    """Curva ROC y AUC por integracion trapezoidal."""
    es_pos = (y_true == POSITIVA).astype(int)
    orden = np.argsort(-puntajes)
    y = es_pos[orden]

    p, n = y.sum(), len(y) - y.sum()
    if p == 0 or n == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")

    tpr = np.concatenate([[0], np.cumsum(y) / p])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / n])
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


# --- Evaluacion ---
def evaluar(nombre: str, ruta_modelo: Path, df: pd.DataFrame) -> dict:
    modelo = YOLO(str(ruta_modelo))
    y_true = df["y_true"].to_numpy()
    y_pred, puntajes, latencias = [], [], []

    for i, ruta in enumerate(df["ruta"], 1):
        t0 = time.perf_counter()
        r = modelo.predict(ruta, verbose=False)[0]
        latencias.append((time.perf_counter() - t0) * 1000)

        y_pred.append(int(r.probs.top1))
        puntajes.append(float(r.probs.data[POSITIVA]))

        if i % 200 == 0:
            print(f"      {i}/{len(df)}...")

    y_pred = np.array(y_pred)
    puntajes = np.array(puntajes)

    cm = matriz_confusion(y_true, y_pred)
    m = metricas(cm)
    fpr, tpr, auc = curva_roc(y_true, puntajes)

    m.update({
        "modelo": nombre,
        "auc": auc,
        "latencia_media_ms": float(np.mean(latencias)),
        "latencia_p95_ms": float(np.percentile(latencias, 95)),
        "tamano_mb": ruta_modelo.stat().st_size / 1024 ** 2,
        "n_imagenes": len(df),
    })
    return {"metricas": m, "cm": cm, "roc": (fpr, tpr, auc)}


# --- Graficas ---
def fig_matrices(resultados: dict, conjunto: str, destino: Path):
    fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados), 4.5))
    axes = np.atleast_1d(axes)

    for ax, (nombre, res) in zip(axes, resultados.items()):
        cm = res["cm"]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=15, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], NOMBRES)
        ax.set_yticks([0, 1], NOMBRES)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Valor real")
        ax.set_title(f"{nombre}\nExactitud = {res['metricas']['exactitud']:.4f}", fontsize=10)

    fig.suptitle(f"Matrices de confusion - {conjunto}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_roc(resultados: dict, conjunto: str, destino: Path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for nombre, res in resultados.items():
        fpr, tpr, auc = res["roc"]
        ax.plot(fpr, tpr, lw=2, label=f"{nombre} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")
    ax.set_xlabel("Tasa de falsos positivos (1 - Especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)")
    ax.set_title(f"Curvas ROC - {conjunto}\nClase positiva: Anomaly", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_comparativa(df: pd.DataFrame, conjunto: str, destino: Path):
    campos = ["exactitud", "sensibilidad", "especificidad", "f1", "auc"]
    etiquetas = ["Exactitud", "Sensibilidad", "Especificidad", "F1", "AUC"]

    x = np.arange(len(campos))
    ancho = 0.8 / len(df)
    fig, ax = plt.subplots(figsize=(10, 5))

    for k, (_, fila) in enumerate(df.iterrows()):
        pos = x + k * ancho - 0.4 + ancho / 2
        barras = ax.bar(pos, [fila[c] for c in campos], ancho, label=fila["modelo"])
        for b, c in zip(barras, campos):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{fila[c]:.3f}", ha="center", fontsize=7, rotation=90)

    ax.set_xticks(x, etiquetas)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Valor")
    ax.set_title(f"Comparacion de metricas - {conjunto}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_compromiso(df: pd.DataFrame, conjunto: str, destino: Path):
    """Exactitud frente a latencia; el area del punto representa el tamano del modelo."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, f in df.iterrows():
        ax.scatter(f["latencia_media_ms"], f["exactitud"],
                   s=f["tamano_mb"] * 12, alpha=0.65, edgecolors="black", zorder=3)
        # Etiqueta debajo del punto: evita que choque con el titulo cuando el
        # modelo mas exacto queda en el borde superior del area de trazado.
        ax.annotate(f"{f['modelo']}\n{f['tamano_mb']:.0f} MB",
                    (f["latencia_media_ms"], f["exactitud"]),
                    textcoords="offset points", xytext=(0, -34),
                    ha="center", fontsize=8.5, zorder=4)

    # Margenes amplios para que ningun punto ni etiqueta toque los bordes.
    ax.margins(x=0.22, y=0.32)
    ax.set_xlabel("Latencia media por imagen (ms)")
    ax.set_ylabel("Exactitud")
    ax.set_title(f"Compromiso exactitud / latencia / tamano\n{conjunto}",
                 fontsize=12, fontweight="bold", pad=14)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --- Principal ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-hash", action="store_true",
                    help="omite la verificacion de fuga de datos")
    args = ap.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    informe = {"fuga_de_datos": None, "resultados": {}}

    if not args.sin_hash:
        informe["fuga_de_datos"] = verificar_fuga()
    else:
        print("\n[1/3] Verificacion de fuga omitida (--sin-hash)")

    print("\n[2/3] Evaluando modelos...")
    tablas = {}

    for conjunto, base in CONJUNTOS.items():
        df_img = listar_imagenes(base)
        print(f"\n  == {conjunto} == ({len(df_img)} imagenes)")

        resultados = {}
        for nombre, ruta in MODELOS.items():
            if not ruta.exists():
                print(f"    {nombre}: modelo no encontrado, se omite")
                continue
            print(f"    {nombre}...")
            resultados[nombre] = evaluar(nombre, ruta, df_img)
            m = resultados[nombre]["metricas"]
            print(f"      exactitud={m['exactitud']:.4f}  sens={m['sensibilidad']:.4f}  "
                  f"esp={m['especificidad']:.4f}  AUC={m['auc']:.4f}  "
                  f"lat={m['latencia_media_ms']:.1f} ms")

        tabla = pd.DataFrame([r["metricas"] for r in resultados.values()])
        tablas[conjunto] = tabla
        informe["resultados"][conjunto] = tabla.to_dict("records")

        slug = "interna" if "interna" in conjunto else "externa"
        fig_matrices(resultados, conjunto, SALIDA / f"matrices_confusion_{slug}.png")
        fig_roc(resultados, conjunto, SALIDA / f"curvas_roc_{slug}.png")
        fig_comparativa(tabla, conjunto, SALIDA / f"comparacion_metricas_{slug}.png")
        fig_compromiso(tabla, conjunto, SALIDA / f"compromiso_exactitud_latencia_{slug}.png")
        tabla.to_csv(SALIDA / f"metricas_{slug}.csv", index=False, encoding="utf-8")

    print("\n[3/3] Guardando resultados...")

    def a_nativo(v):
        """pandas/numpy devuelven np.float64 y np.int64, que json no serializa."""
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return str(v)

    (SALIDA / "informe_completo.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False, default=a_nativo),
        encoding="utf-8")

    print("\n" + "=" * 78)
    for conjunto, tabla in tablas.items():
        print(f"\n{conjunto}")
        cols = ["modelo", "exactitud", "sensibilidad", "especificidad", "f1", "auc",
                "latencia_media_ms", "tamano_mb"]
        print(tabla[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + "=" * 78)
    print(f"Figuras y tablas en: {SALIDA}")


if __name__ == "__main__":
    sys.exit(main())
