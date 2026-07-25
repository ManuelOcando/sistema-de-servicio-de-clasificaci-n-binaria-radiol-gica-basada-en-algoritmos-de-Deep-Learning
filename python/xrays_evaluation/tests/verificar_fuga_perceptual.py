"""
Verificacion de fuga de datos mediante hashing perceptual, calibrada.

Motivacion
----------
El hash MD5 solo detecta archivos identicos byte a byte. Roboflow reprocesa
y redimensiona las imagenes al generar las particiones (raw_data esta a
resolucion completa y split_data/test a 224x224), de modo que una misma
radiografia produce hashes MD5 distintos y la comprobacion la da por
diferente. El MD5 es, por tanto, ciego al duplicado redimensionado.

El hash perceptual sobrevive al reescalado, pero introduce el problema
contrario: todas las radiografias de torax comparten anatomia, encuadre y
fondo, asi que dos pacientes distintos producen hashes muy parecidos. Un
umbral elegido a ojo marca como duplicadas imagenes que no lo son.

Por eso este script no fija un umbral arbitrario, sino que lo calibra:

  1. Usa un dHash de 256 bits (16x16), con mas resolucion que el habitual
     de 64 bits.
  2. Mide la DISTRIBUCION NULA: la distancia entre pares de imagenes que
     sabemos distintas. Indica cuanto se parecen dos radiografias sin
     relacion.
  3. Usa VERDAD CONOCIDA: las imagenes identicas por MD5 entre
     benchmarking y raw_data son duplicados reales y deben dar distancia 0.
  4. Solo entonces informa de coincidencias, a varios umbrales, para que la
     conclusion no dependa de un solo valor elegido de antemano.

Uso:
    python tests/verificar_fuga_perceptual.py
"""

import hashlib
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

# Ver nota sobre XRAY_DATASET_DIR en benchmark_tesis.py.
DATOS = Path(os.getenv("XRAY_DATASET_DIR",
                       Path(__file__).resolve().parents[3] / "datasets"))
TR = DATOS / "train/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121"
TE = DATOS / "test/ingeniia_services_xrays_evaluation_img_v1.0.0_test_20251130"

CONJUNTOS = {
    "raw_data": TR / "raw_data",
    "train": TR / "split_data/train",
    "valid": TR / "split_data/valid",
    "test": TR / "split_data/test",
    "benchmarking": TE / "benchmarking",
}

CLASES = ("anomaly", "normal")
EXTENSIONES = ("*.jpg", "*.jpeg", "*.png")
SALIDA = Path(__file__).resolve().parent / "results/benchmark_tesis"
LADO = 16  # dHash de 16x16 = 256 bits
UMBRALES = (0, 2, 5, 10, 15, 20)

random.seed(20260725)


def dhash(ruta: Path):
    img = cv2.imread(str(ruta), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    pequena = cv2.resize(img, (LADO + 1, LADO), interpolation=cv2.INTER_AREA)
    return np.packbits(pequena[:, 1:] > pequena[:, :-1])


def recorrer(base: Path):
    for clase in CLASES:
        for patron in EXTENSIONES:
            for p in (base / clase).glob(patron):
                yield clase, p


def indexar(base: Path) -> dict:
    salida = {}
    for clase, p in recorrer(base):
        h = dhash(p)
        if h is not None:
            salida[f"{clase}/{p.name}"] = {
                "dhash": h,
                "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            }
    return salida


def matriz(hashes: dict) -> np.ndarray:
    return np.unpackbits(np.array([v["dhash"] for v in hashes.values()]), axis=1)


def distancias_minimas(a: dict, b: dict) -> np.ndarray:
    """Para cada elemento de A, distancia de Hamming al mas cercano de B."""
    A, B = matriz(a), matriz(b)
    minimas = np.empty(len(A), dtype=np.int16)
    # Por bloques, para no construir una matriz de 18843 x 900 x 256 bits.
    for i in range(0, len(A), 256):
        bloque = A[i:i + 256]
        d = (bloque[:, None, :] != B[None, :, :]).sum(axis=2)
        minimas[i:i + 256] = d.min(axis=1)
    return minimas


def distribucion_nula(hashes: dict, n_pares: int = 4000) -> np.ndarray:
    """Distancias entre pares de imagenes distintas del mismo conjunto."""
    M = matriz(hashes)
    idx = range(len(M))
    distancias = []
    for _ in range(n_pares):
        i, j = random.sample(idx, 2)
        distancias.append(int((M[i] != M[j]).sum()))
    return np.array(distancias)


def main():
    print(f"Indexando con dHash de {LADO}x{LADO} = {LADO * LADO} bits...\n")
    idx = {}
    for nombre, base in CONJUNTOS.items():
        idx[nombre] = indexar(base)
        print(f"  {nombre:14s}: {len(idx[nombre]):5d} imagenes")

    # --- 1. Distribucion nula ---
    print("\n[1] Distribucion nula: distancia entre radiografias DISTINTAS")
    nula = distribucion_nula(idx["train"])
    p1 = int(np.percentile(nula, 1))
    print(f"    media={nula.mean():.1f}  min={nula.min()}  "
          f"percentil-1={p1}  percentil-5={int(np.percentile(nula, 5))}")
    print(f"    -> Dos radiografias sin relacion difieren tipicamente en "
          f"{nula.mean():.0f} de {LADO * LADO} bits.")
    print(f"    -> Un umbral valido debe quedar MUY por debajo de {p1}.")

    # --- 2. Verdad conocida ---
    print("\n[2] Verdad conocida: duplicados exactos por MD5")
    md5_raw = {v["md5"] for v in idx["raw_data"].values()}
    ciertos = [k for k, v in idx["benchmarking"].items() if v["md5"] in md5_raw]
    print(f"    benchmarking identicas por MD5 a raw_data: {len(ciertos)} de "
          f"{len(idx['benchmarking'])}")

    if ciertos:
        sub = {k: idx["benchmarking"][k] for k in ciertos}
        d = distancias_minimas(sub, idx["raw_data"])
        print(f"    distancia perceptual de esos duplicados reales: "
              f"max={d.max()}  media={d.mean():.2f}")
        print(f"    -> Los duplicados verdaderos se detectan con umbral {d.max()}.")
        umbral_sugerido = max(int(d.max()), 2)
    else:
        umbral_sugerido = 2

    # --- 3. Comparaciones a varios umbrales ---
    print(f"\n[3] Coincidencias segun umbral (sugerido: {umbral_sugerido})\n")
    comparaciones = [
        ("test", "train"), ("test", "valid"),
        ("benchmarking", "train"), ("benchmarking", "raw_data"),
    ]
    cabecera = "    " + f"{'comparacion':30s}" + "".join(f"{f'<={u}':>8s}" for u in UMBRALES)
    print(cabecera)
    print("    " + "-" * (30 + 8 * len(UMBRALES)))

    informe = []
    for a, b in comparaciones:
        d = distancias_minimas(idx[a], idx[b])
        fila = [int((d <= u).sum()) for u in UMBRALES]
        etiqueta = f"{a} en {b} (n={len(d)})"
        print(f"    {etiqueta:30s}" + "".join(f"{v:>8d}" for v in fila))
        informe.append({
            "origen": a, "contra": b, "total": len(d),
            "coincidencias_por_umbral": dict(zip(map(str, UMBRALES), fila)),
            "distancia_minima_observada": int(d.min()),
        })

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "verificacion_fuga_perceptual.json"
    destino.write_text(json.dumps({
        "bits": LADO * LADO,
        "distribucion_nula": {
            "media": float(nula.mean()), "minimo": int(nula.min()),
            "percentil_1": p1,
        },
        "duplicados_conocidos_md5": len(ciertos),
        "umbral_sugerido": umbral_sugerido,
        "comparaciones": informe,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n    Informe guardado en: {destino}")


if __name__ == "__main__":
    main()
