"""
Calibracion del filtro de plausibilidad radiologica.

Un clasificador binario entrenado solo con radiografias no puede responder
"esto no es una radiografia": la funcion softmax reparte siempre el total
entre sus dos clases. En la practica eso significa que una fotografia
cualquiera obtiene una prediccion con confianza altisima, por encima incluso
del umbral de derivacion a revision humana.

La solucion no es del modelo sino previa a el: comprobar que la entrada tenga
las propiedades de una radiografia antes de clasificarla. Este script deriva
los umbrales de esa comprobacion midiendo las radiografias reales del
conjunto de prueba, en lugar de fijarlos a ojo.

Se miden tres propiedades:

  - Saturacion cromatica. Una radiografia es una imagen en escala de grises,
    de modo que sus tres canales coinciden y la saturacion es practicamente
    nula. Cualquier fotografia en color la supera con holgura.
  - Desviacion tipica de la luminancia. Distingue las imagenes de tono
    uniforme, que no contienen estructura alguna.
  - Entropia del histograma. Mide la riqueza tonal: una radiografia real
    ocupa buena parte del rango, mientras que una imagen plana o de ruido
    puro se aparta de ese comportamiento.

Uso:
    python tests/calibrar_plausibilidad.py
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DATOS = Path(os.getenv("XRAY_DATASET_DIR", RAIZ.parent.parent / "datasets"))
PRUEBA = (
    DATOS
    / "train/ingeniia_services_xrays_evaluation_img_v1.0.0_training_20251121"
    / "split_data/test"
)
BENCH = (
    DATOS
    / "test/ingeniia_services_xrays_evaluation_img_v1.0.0_test_20251130"
    / "benchmarking"
)
SALIDA = RAIZ / "tests/results/benchmark_tesis"
CLASES = ("anomaly", "normal")
EXTENSIONES = ("*.jpg", "*.jpeg", "*.png")


def metricas(imagen: np.ndarray) -> dict:
    """Propiedades de plausibilidad de una imagen en BGR."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Saturacion: en una imagen en escala de grises los tres canales coinciden.
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    saturacion = float(hsv[:, :, 1].mean())

    desviacion = float(gris.std())

    histograma = cv2.calcHist([gris], [0], None, [256], [0, 256]).ravel()
    probabilidades = histograma / max(histograma.sum(), 1)
    no_nulas = probabilidades[probabilidades > 0]
    entropia = float(-(no_nulas * np.log2(no_nulas)).sum())

    return {"saturacion": saturacion, "desviacion": desviacion, "entropia": entropia}


def recorrer(base: Path):
    for clase in CLASES:
        for patron in EXTENSIONES:
            yield from (base / clase).glob(patron)


def medir_conjunto(base: Path, nombre: str) -> dict:
    valores = {"saturacion": [], "desviacion": [], "entropia": []}
    total = 0
    for ruta in recorrer(base):
        img = cv2.imread(str(ruta))
        if img is None:
            continue
        for clave, valor in metricas(img).items():
            valores[clave].append(valor)
        total += 1

    print(f"\n{nombre} ({total} imagenes)")
    resumen = {}
    for clave, lista in valores.items():
        a = np.array(lista)
        resumen[clave] = {
            "minimo": float(a.min()),
            "p1": float(np.percentile(a, 1)),
            "mediana": float(np.median(a)),
            "p99": float(np.percentile(a, 99)),
            "maximo": float(a.max()),
        }
        print(
            f"  {clave:12s} min={a.min():7.2f}  p1={np.percentile(a,1):7.2f}  "
            f"mediana={np.median(a):7.2f}  p99={np.percentile(a,99):7.2f}  "
            f"max={a.max():7.2f}"
        )
    return resumen


def casos_sinteticos() -> dict:
    rng = np.random.default_rng(7)
    casos = {
        "Ruido aleatorio a color": rng.integers(0, 255, (600, 600, 3), dtype=np.uint8),
        "Negro uniforme": np.zeros((600, 600, 3), dtype=np.uint8),
        "Blanco uniforme": np.full((600, 600, 3), 255, dtype=np.uint8),
    }

    grad = np.zeros((600, 800, 3), dtype=np.uint8)
    grad[:, :, 0] = np.linspace(200, 40, 800)[None, :]
    grad[:, :, 1] = np.linspace(120, 190, 600)[:, None]
    grad[:, :, 2] = 60
    casos["Degradado de color"] = grad

    formas = np.full((600, 600, 3), 250, dtype=np.uint8)
    cv2.circle(formas, (200, 200), 120, (0, 0, 220), -1)
    cv2.rectangle(formas, (320, 300), (520, 500), (30, 180, 30), -1)
    casos["Formas geometricas"] = formas

    # Fotografia en escala de grises simulada: sin color, pero con estructura
    # ajena a una radiografia.
    gris = rng.integers(90, 170, (400, 400), dtype=np.uint8)
    gris = cv2.GaussianBlur(gris, (31, 31), 0)
    casos["Textura gris difusa"] = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)

    return casos


def main():
    print("Midiendo radiografias reales para derivar los umbrales...")
    reales = medir_conjunto(PRUEBA, "Conjunto de prueba reservado")
    externas = medir_conjunto(BENCH, "Conjunto benchmarking")

    print("\nCasos que NO son radiografias")
    print(f"  {'entrada':<26} {'saturacion':>11} {'desviacion':>11} {'entropia':>10}")
    print("  " + "-" * 60)
    sinteticos = {}
    for nombre, img in casos_sinteticos().items():
        m = metricas(img)
        sinteticos[nombre] = m
        print(
            f"  {nombre:<26} {m['saturacion']:11.2f} {m['desviacion']:11.2f} "
            f"{m['entropia']:10.2f}"
        )

    # Los umbrales se fijan con margen respecto a lo observado en radiografias
    # reales: se toma el peor caso de los dos conjuntos y se deja holgura para
    # no rechazar imagenes legitimas algo atipicas.
    sat_max = max(reales["saturacion"]["maximo"], externas["saturacion"]["maximo"])
    des_min = min(reales["desviacion"]["minimo"], externas["desviacion"]["minimo"])
    ent_min = min(reales["entropia"]["minimo"], externas["entropia"]["minimo"])

    umbrales = {
        "saturacion_maxima": round(max(sat_max * 2, 12.0), 2),
        "desviacion_minima": round(des_min * 0.5, 2),
        "entropia_minima": round(ent_min * 0.7, 2),
    }

    print("\nUmbrales derivados (con margen sobre lo observado)")
    for clave, valor in umbrales.items():
        print(f"  {clave:22s} {valor}")

    print("\nComprobacion sobre los casos sinteticos")
    rechazados = 0
    for nombre, m in sinteticos.items():
        motivos = []
        if m["saturacion"] > umbrales["saturacion_maxima"]:
            motivos.append("color")
        if m["desviacion"] < umbrales["desviacion_minima"]:
            motivos.append("sin estructura")
        if m["entropia"] < umbrales["entropia_minima"]:
            motivos.append("tonalidad pobre")
        veredicto = f"RECHAZADA ({', '.join(motivos)})" if motivos else "aceptada"
        if motivos:
            rechazados += 1
        print(f"  {nombre:<26} {veredicto}")
    print(f"\n  {rechazados} de {len(sinteticos)} entradas invalidas serian rechazadas")

    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / "calibracion_plausibilidad.json"
    destino.write_text(
        json.dumps(
            {
                "radiografias_reales": {"prueba": reales, "benchmarking": externas},
                "casos_no_radiograficos": sinteticos,
                "umbrales": umbrales,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nInforme guardado en: {destino}")


if __name__ == "__main__":
    main()
