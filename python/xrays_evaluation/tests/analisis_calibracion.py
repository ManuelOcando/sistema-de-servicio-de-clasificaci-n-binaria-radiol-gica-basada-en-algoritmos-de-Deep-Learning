"""
Analisis de calibracion del modelo desplegado.

Responde a una pregunta que surge al ver que el servicio devuelve confianza
1.0000: se trata de sobreajuste?

Se distinguen dos fenomenos que suelen confundirse:

  - Sobreajuste: el modelo acierta sobre los datos de entrenamiento y falla
    sobre datos nuevos. Se detecta comparando el rendimiento entre ambos.
  - Sobreconfianza (mala calibracion): el modelo acierta sobre datos nuevos,
    pero las probabilidades que emite no reflejan su fiabilidad real. Es un
    comportamiento documentado en redes profundas modernas
    (Guo et al., 2017, "On Calibration of Modern Neural Networks").

El script mide:
  1. Si las imagenes de ejemplo del repositorio pertenecen al conjunto de
     entrenamiento, lo que explicaria una confianza de 1.0000 sobre ellas.
  2. La distribucion de confianza sobre el conjunto de prueba reservado,
     separando aciertos de errores.
  3. El Error de Calibracion Esperado (ECE) y el diagrama de fiabilidad.

Uso:
    python tests/analisis_calibracion.py
"""

import hashlib
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
MODELO = RAIZ / "models/YOLO/xrays_evaluation_model_medium_v1.pt"
EJEMPLOS = RAIZ / "examples/images"

# Ver nota sobre XRAY_DATASET_DIR en benchmark_tesis.py.
DATOS = Path(os.getenv("XRAY_DATASET_DIR", RAIZ.parent.parent / "datasets"))
BASE_TRAIN = DATOS / "train/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121"
ENTRENAMIENTO = BASE_TRAIN / "split_data/train"
PRUEBA = BASE_TRAIN / "split_data/test"

CARPETAS = {"anomaly": 0, "normal": 1}
EXTENSIONES = ("*.jpg", "*.jpeg", "*.png")
POSITIVA = 0

SALIDA = RAIZ / "tests/results/benchmark_tesis"


def hash_archivo(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def comprobar_ejemplos():
    """Verifica si las imagenes de ejemplo salieron del entrenamiento."""
    print("[1/3] Origen de las imagenes de ejemplo del repositorio\n")

    hashes_train = {}
    for carpeta in CARPETAS:
        for patron in EXTENSIONES:
            for p in (ENTRENAMIENTO / carpeta).glob(patron):
                hashes_train[hash_archivo(p)] = f"{carpeta}/{p.name}"
    print(f"      entrenamiento: {len(hashes_train)} imagenes")

    for ejemplo in sorted(EJEMPLOS.glob("*.jpeg")):
        h = hash_archivo(ejemplo)
        coincidencia = hashes_train.get(h)
        if coincidencia:
            print(f"      {ejemplo.name}: SI esta en entrenamiento -> {coincidencia}")
        else:
            print(f"      {ejemplo.name}: NO esta en entrenamiento")
    print()


def evaluar_prueba():
    """Recoge confianzas sobre el conjunto de prueba reservado."""
    print("[2/3] Evaluando el conjunto de prueba reservado...\n")

    rutas, y_true = [], []
    for carpeta, etiqueta in CARPETAS.items():
        for patron in EXTENSIONES:
            for p in (PRUEBA / carpeta).glob(patron):
                rutas.append(p)
                y_true.append(etiqueta)

    modelo = YOLO(str(MODELO))
    y_pred, confianzas = [], []
    for i, ruta in enumerate(rutas, 1):
        r = modelo.predict(str(ruta), verbose=False)[0]
        y_pred.append(int(r.probs.top1))
        confianzas.append(float(r.probs.top1conf))
        if i % 300 == 0:
            print(f"      {i}/{len(rutas)}...")

    return np.array(y_true), np.array(y_pred), np.array(confianzas)


def ece(aciertos: np.ndarray, confianzas: np.ndarray, n_bins: int = 10) -> tuple:
    """Error de Calibracion Esperado y datos del diagrama de fiabilidad."""
    bordes = np.linspace(0, 1, n_bins + 1)
    error, filas = 0.0, []
    for lo, hi in zip(bordes[:-1], bordes[1:]):
        en_bin = (confianzas > lo) & (confianzas <= hi)
        n = en_bin.sum()
        if n == 0:
            continue
        precision = aciertos[en_bin].mean()
        confianza_media = confianzas[en_bin].mean()
        error += (n / len(confianzas)) * abs(precision - confianza_media)
        filas.append((lo, hi, int(n), precision, confianza_media))
    return error, filas


def main():
    comprobar_ejemplos()

    y_true, y_pred, conf = evaluar_prueba()
    aciertos = (y_true == y_pred).astype(float)

    print("\n[3/3] Resultados\n")
    print(f"      Imagenes evaluadas    : {len(y_true)}")
    print(f"      Exactitud             : {aciertos.mean():.4f}")
    print()
    print(f"      Confianza media       : {conf.mean():.4f}")
    print(f"      Confianza en ACIERTOS : {conf[aciertos == 1].mean():.4f}")
    if (aciertos == 0).any():
        print(f"      Confianza en ERRORES  : {conf[aciertos == 0].mean():.4f}")
        print(f"      Errores con conf>0.99 : {int((conf[aciertos == 0] > 0.99).sum())}"
              f" de {int((aciertos == 0).sum())}")
    print()
    for umbral in (0.9999, 0.999, 0.99, 0.95):
        pct = 100 * (conf >= umbral).mean()
        print(f"      Predicciones con confianza >= {umbral:<7}: {pct:5.1f}%")

    error_cal, filas = ece(aciertos, conf)
    print(f"\n      Error de Calibracion Esperado (ECE): {error_cal:.4f}")
    print(f"      Brecha confianza - exactitud       : {conf.mean() - aciertos.mean():+.4f}")

    # --- Figura: histograma de confianza y diagrama de fiabilidad ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.hist(conf[aciertos == 1], bins=40, alpha=0.75, label="Aciertos", color="#2b7bba")
    if (aciertos == 0).any():
        ax1.hist(conf[aciertos == 0], bins=40, alpha=0.85, label="Errores", color="#d1495b")
    ax1.set_yscale("log")
    ax1.set_xlabel("Confianza de la prediccion")
    ax1.set_ylabel("Numero de imagenes (escala logaritmica)")
    ax1.set_title("Distribucion de la confianza", fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if filas:
        centros = [(lo + hi) / 2 for lo, hi, *_ in filas]
        precisiones = [f[3] for f in filas]
        ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Calibracion perfecta")
        ax2.plot(centros, precisiones, "o-", lw=2, color="#2b7bba", label="Modelo")
        for lo, hi, n, prec, _ in filas:
            ax2.annotate(f"n={n}", ((lo + hi) / 2, prec),
                         textcoords="offset points", xytext=(0, -14),
                         ha="center", fontsize=7)
    ax2.set_xlabel("Confianza declarada")
    ax2.set_ylabel("Exactitud observada")
    ax2.set_title(f"Diagrama de fiabilidad (ECE = {error_cal:.4f})",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.suptitle("Calibracion del modelo YOLO11m-cls sobre el conjunto de prueba reservado",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    destino = SALIDA / "calibracion_confianza.png"
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n      Figura guardada en: {destino}")


if __name__ == "__main__":
    main()
